# Starts MoDL segmentation server locally for repeated API calls without
# reloading TensorFlow and U-RNet+ for every image set.
# The server will be available at: http://127.0.0.1:5002/segment

from pathlib import Path
import glob
import math
import os

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import tifffile

from segment_predict_flexible import (
    DEFAULT_WEIGHTS,
    PATCH_SIZE,
    crop_to_output,
    convert_to_uint8,
    extract_patches,
    predict_patches,
    read_grayscale_array,
    resize_array,
    resize_outputs_to_original,
    resolve_weights_path,
)

# If CUDNN crashes this may help:
# os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "0")


app = Flask(__name__)
prev_model_path = ""
model = None
tf.config.run_functions_eagerly(True)


def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
            tf.config.set_logical_device_configuration(
                gpus[0],
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=12000  # MB
                    )
                ],
            )
        except RuntimeError as exc:
            print(exc)
    else:
        tf.config.set_visible_devices([], "GPU")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_float(name, default):
    return float(request.form.get(name, default))


def get_int(name, default):
    return int(request.form.get(name, default))


def resolve_image_list(img_path):
    if any(ch in img_path for ch in "*?[]"):
        return sorted(glob.glob(img_path))

    if os.path.isdir(img_path):
        patterns = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp")
        image_list = []
        for pattern in patterns:
            image_list.extend(glob.glob(os.path.join(img_path, pattern)))
        return sorted(image_list)

    if os.path.isfile(img_path):
        return [img_path]

    return []


def load_cached_model(model_path):
    global model
    global prev_model_path

    model_path = str(resolve_weights_path(model_path).resolve())
    if prev_model_path != model_path or model is None:
        print(f"Loading MoDL model: {model_path}")
        model = load_model(model_path)
        prev_model_path = model_path
    return model


def save_png(path, image):
    Image.fromarray(image).save(path)


def output_path_for(image_path, suffix):
    path = Path(image_path)
    return path.with_name(f"{path.stem}{suffix}.tif")


def stitch_probabilities_trimmed(predictions, positions, output_shape, overlap):
    height, width = output_shape
    probability = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    max_y = max(y for y, _ in positions)
    max_x = max(x for _, x in positions)
    trim_before = overlap // 2
    trim_after = overlap - trim_before

    for prediction, (y, x) in zip(predictions, positions):
        top = trim_before if y > 0 else 0
        left = trim_before if x > 0 else 0
        bottom = trim_after if y < max_y else 0
        right = trim_after if x < max_x else 0

        dest_y0 = y + top
        dest_x0 = x + left
        dest_y1 = min(y + PATCH_SIZE - bottom, height)
        dest_x1 = min(x + PATCH_SIZE - right, width)
        if dest_y1 <= dest_y0 or dest_x1 <= dest_x0:
            continue

        src_y0 = top
        src_x0 = left
        src_y1 = src_y0 + (dest_y1 - dest_y0)
        src_x1 = src_x0 + (dest_x1 - dest_x0)

        probability[dest_y0:dest_y1, dest_x0:dest_x1] += prediction[src_y0:src_y1, src_x0:src_x1]
        weight[dest_y0:dest_y1, dest_x0:dest_x1] += 1.0

    return probability / np.maximum(weight, 1.0)


def make_blend_weights(y, x, max_y, max_x, overlap):
    weights = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    if overlap <= 0:
        return weights

    ramp_up = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
    ramp_down = np.linspace(1.0, 0.0, overlap, endpoint=False, dtype=np.float32)

    if y > 0:
        weights[:overlap, :] *= ramp_up[:, np.newaxis]
    if y < max_y:
        weights[-overlap:, :] *= ramp_down[:, np.newaxis]
    if x > 0:
        weights[:, :overlap] *= ramp_up[np.newaxis, :]
    if x < max_x:
        weights[:, -overlap:] *= ramp_down[np.newaxis, :]
    return weights


def stitch_probabilities_blended(predictions, positions, output_shape, overlap):
    height, width = output_shape
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    max_y = max(y for y, _ in positions)
    max_x = max(x for _, x in positions)

    for prediction, (y, x) in zip(predictions, positions):
        dest_y1 = min(y + PATCH_SIZE, height)
        dest_x1 = min(x + PATCH_SIZE, width)
        crop_height = dest_y1 - y
        crop_width = dest_x1 - x
        if crop_height <= 0 or crop_width <= 0:
            continue

        weights = make_blend_weights(y, x, max_y, max_x, overlap)
        weights = weights[:crop_height, :crop_width]
        prediction = prediction[:crop_height, :crop_width]

        probability_sum[y:dest_y1, x:dest_x1] += prediction * weights
        weight_sum[y:dest_y1, x:dest_x1] += weights

    return probability_sum / np.maximum(weight_sum, 1e-6)


def stitch_probabilities_refined(predictions, positions, output_shape, overlap, blend_overlap):
    if overlap <= 0:
        return stitch_probabilities_blended(predictions, positions, output_shape, overlap)
    if blend_overlap:
        return stitch_probabilities_blended(predictions, positions, output_shape, overlap)
    return stitch_probabilities_trimmed(predictions, positions, output_shape, overlap)


def segment_one(
    model_instance,
    image_path,
    overlap,
    scale,
    threshold,
    batch_size,
    output_original_size,
    percentile_low,
    percentile_high,
    save_png_outputs,
    blend_overlap,
):
    original_raw = read_grayscale_array(image_path)
    original_array = convert_to_uint8(original_raw, percentile_low, percentile_high)
    original_height, original_width = original_array.shape
    scaled_array = resize_array(original_array, scale)

    patches, positions, output_shape, padded_shape, crop_origin = extract_patches(scaled_array, overlap)
    predictions = predict_patches(model_instance, patches, batch_size)
    probability = stitch_probabilities_refined(predictions, positions, padded_shape, overlap, blend_overlap)
    probability = crop_to_output(probability, output_shape, crop_origin)

    mask = (probability > threshold).astype(np.uint8) * 255
    probability_image = np.clip(probability * 255.0, 0, 255).astype(np.uint8)

    if output_original_size:
        empty_overlay = np.zeros((*probability_image.shape, 3), dtype=np.uint8)
        mask, probability_image, _ = resize_outputs_to_original(
            mask, probability_image, empty_overlay, (original_width, original_height)
        )

    mask_path = output_path_for(image_path, "_masks")
    probability_path = output_path_for(image_path, "_prob")
    tifffile.imwrite(mask_path, mask)
    tifffile.imwrite(probability_path, probability_image)

    if save_png_outputs:
        save_png(mask_path.with_suffix(".png"), mask)
        save_png(probability_path.with_suffix(".png"), probability_image)

    return str(mask_path), str(probability_path)


@app.route("/segment", methods=["POST"])
def segment():
    print("Segmenting with MoDL")
    print(request.form)

    img_path = request.form["--dir"]
    requested_model = request.form.get("--weights", request.form.get("--model", DEFAULT_WEIGHTS))

    image_list = resolve_image_list(img_path)
    if not image_list:
        return jsonify({"status": "failed", "mask_path": "", "prob_path": "", "error": "No input images found"})

    try:
        overlap = get_int("--overlap", 0)
        scale = get_float("--scale", 1.0)
        threshold = get_float("--threshold", 0.7)
        batch_size = get_int("--batch-size", request.form.get("--batch_size", 1))
        output_original_size = str_to_bool(request.form.get("--output-original-size", False))
        percentile_low = get_float("--percentile-low", 0.1)
        percentile_high = get_float("--percentile-high", 99.9)
        save_png_outputs = str_to_bool(request.form.get("--save-png", False))
        blend_overlap = str_to_bool(request.form.get("--blend-overlap", False))

        if scale <= 0 or math.isnan(scale):
            raise ValueError("--scale must be greater than 0")

        model_instance = load_cached_model(requested_model)

        mask_paths = []
        probability_paths = []
        for image_path in image_list:
            mask_path, probability_path = segment_one(
                model_instance=model_instance,
                image_path=image_path,
                overlap=overlap,
                scale=scale,
                threshold=threshold,
                batch_size=batch_size,
                output_original_size=output_original_size,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                save_png_outputs=save_png_outputs,
                blend_overlap=blend_overlap,
            )
            mask_paths.append(mask_path)
            probability_paths.append(probability_path)

    except Exception as exc:
        print(exc)
        return jsonify({"status": "failed", "mask_path": "", "prob_path": "", "error": str(exc)})

    return jsonify(
        {
            "status": "done",
            "mask_path": mask_paths[-1],
            "prob_path": probability_paths[-1],
            "mask_paths": mask_paths,
            "prob_paths": probability_paths,
        }
    )


if __name__ == "__main__":
    configure_gpu()
    app.run(host="127.0.0.1", port=5002)
