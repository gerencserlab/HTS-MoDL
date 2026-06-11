import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import tifffile


PATCH_SIZE = 512
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_WEIGHTS = "U-RNet+.hdf5"


def get_resample_filter():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        except RuntimeError as exc:
            print(exc)
    else:
        tf.config.set_visible_devices([], "GPU")


def resolve_weights_path(weights):
    weights_path = Path(weights)
    if not weights_path.suffix:
        weights_path = weights_path.with_suffix(".hdf5")

    if weights_path.is_absolute() or weights_path.parent != Path("."):
        return weights_path
    return MODEL_DIR / weights_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment arbitrary-sized images with U-RNet+ using 512x512 tiled prediction."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "testraw"),
        help="Input image file or directory. Directories process common image files.",
    )
    parser.add_argument(
        "--weights",
        "--model",
        dest="weights",
        default=DEFAULT_WEIGHTS,
        help=(
            "Model weights filename or path. Filename-only values are loaded from "
            f"{MODEL_DIR}; .hdf5 is appended when omitted."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "final_results" / "flexible"),
        help="Output directory for masks and overlays.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Patch overlap in pixels. Must be from 0 to 511.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor applied before segmentation. Use <1 to downscale or >1 to upscale.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Probability threshold used to create the binary mask.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="TensorFlow prediction batch size.",
    )
    parser.add_argument(
        "--output-original-size",
        action="store_true",
        help="Resize the final mask and overlay back to the original input image size.",
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=0.1,
        help="Lower percentile for converting non-8-bit images to 8-bit.",
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.9,
        help="Upper percentile for converting non-8-bit images to 8-bit.",
    )
    parser.add_argument(
        "--save-debug-input",
        action="store_true",
        help="Save the normalized 8-bit input image used for tiling.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save PNG copies of outputs for viewers that display TIFFs incorrectly.",
    )
    return parser.parse_args()


def list_images(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]

    image_exts = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in image_exts)


def read_grayscale_array(image_path):
    path = Path(image_path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        image = tifffile.imread(path)
    else:
        image = np.asarray(Image.open(path))

    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim > 2:
        image = image.reshape((-1,) + image.shape[-2:])[0]

    return image


def convert_to_uint8(image, percentile_low, percentile_high):
    if image.dtype == np.uint8:
        return image

    if not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")

    image = image.astype(np.float32)
    low, high = np.percentile(image, [percentile_low, percentile_high])
    if high <= low:
        low = float(image.min())
        high = float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - low) / (high - low)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def save_png(path, image):
    Image.fromarray(image).save(path)


def resize_array(image_array, scale):
    if scale <= 0:
        raise ValueError("--scale must be greater than 0")
    if math.isclose(scale, 1.0):
        return image_array

    height, width = image_array.shape
    scaled_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.fromarray(image_array)
    return np.asarray(image.resize(scaled_size, get_resample_filter()), dtype=np.uint8)


def compute_starts(length, patch_size, stride):
    starts = [0]
    while starts[-1] + patch_size < length:
        starts.append(starts[-1] + stride)
    return starts


def extract_patches(image_array, overlap):
    if overlap < 0 or overlap >= PATCH_SIZE:
        raise ValueError("--overlap must be from 0 to 511")

    stride = PATCH_SIZE - overlap
    height, width = image_array.shape
    y_starts = compute_starts(height, PATCH_SIZE, stride)
    x_starts = compute_starts(width, PATCH_SIZE, stride)

    padded_height = y_starts[-1] + PATCH_SIZE
    padded_width = x_starts[-1] + PATCH_SIZE
    padded = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded[:height, :width] = image_array

    patches = []
    positions = []
    for y in y_starts:
        for x in x_starts:
            patches.append(padded[y:y + PATCH_SIZE, x:x + PATCH_SIZE])
            positions.append((y, x))

    patches = np.asarray(patches, dtype=np.uint8)[..., np.newaxis]
    return patches, positions, (height, width), (padded_height, padded_width)


def normalize_like_original_script(patches):
    patches = patches.astype("float32") / 255.0
    # patches -= patches.mean(axis=0)
    patches -= patches.mean()
    return patches


def predict_patches(model, patches, batch_size):
    normalized = normalize_like_original_script(patches)
    predictions = model.predict(normalized, batch_size=batch_size, verbose=1)
    return predictions[..., 0]


def stitch_probabilities(predictions, positions, output_shape, padded_shape):
    padded_height, padded_width = padded_shape
    probability_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)

    for prediction, (y, x) in zip(predictions, positions):
        y_end = y + PATCH_SIZE
        x_end = x + PATCH_SIZE
        probability_sum[y:y_end, x:x_end] += prediction
        weight_sum[y:y_end, x:x_end] += 1.0

    probability = probability_sum / np.maximum(weight_sum, 1.0)
    height, width = output_shape
    return probability[:height, :width]


def make_overlay(gray_array, mask):
    color = cv2.cvtColor(gray_array, cv2.COLOR_GRAY2RGB)
    overlay = color.copy()
    overlay[mask > 0] = (255, 255, 0)
    return cv2.addWeighted(color, 0.65, overlay, 0.35, 0)


def resize_outputs_to_original(mask, probability, overlay, original_size):
    width, height = original_size
    mask_image = Image.fromarray(mask)
    probability_image = Image.fromarray(probability)
    overlay_image = Image.fromarray(overlay)

    mask = np.asarray(mask_image.resize((width, height), Image.NEAREST), dtype=np.uint8)
    probability = np.asarray(probability_image.resize((width, height), get_resample_filter()), dtype=np.uint8)
    overlay = np.asarray(overlay_image.resize((width, height), get_resample_filter()), dtype=np.uint8)
    return mask, probability, overlay


def segment_image(
    model,
    image_path,
    output_dir,
    scale,
    overlap,
    threshold,
    batch_size,
    output_original_size,
    percentile_low,
    percentile_high,
    save_debug_input,
    save_png_outputs,
):
    original_raw = read_grayscale_array(image_path)
    original_array = convert_to_uint8(original_raw, percentile_low, percentile_high)
    original_height, original_width = original_array.shape
    scaled_array = resize_array(original_array, scale)

    patches, positions, output_shape, padded_shape = extract_patches(scaled_array, overlap)
    predictions = predict_patches(model, patches, batch_size)
    probability = stitch_probabilities(predictions, positions, output_shape, padded_shape)

    mask = (probability > threshold).astype(np.uint8) * 255
    probability_image = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    overlay = make_overlay(scaled_array, mask)

    if output_original_size:
        mask, probability_image, overlay = resize_outputs_to_original(
            mask, probability_image, overlay, (original_width, original_height)
        )

    stem = Path(image_path).stem
    tifffile.imwrite(output_dir / "bw" / f"{stem}_mask.tif", mask)
    tifffile.imwrite(output_dir / "probability" / f"{stem}_probability.tif", probability_image)
    tifffile.imwrite(output_dir / "overlay" / f"{stem}_overlay.tif", overlay, photometric="rgb")
    if save_debug_input:
        tifffile.imwrite(output_dir / "debug_input" / f"{stem}_input_8bit.tif", scaled_array)
    if save_png_outputs:
        save_png(output_dir / "bw" / f"{stem}_mask.png", mask)
        save_png(output_dir / "probability" / f"{stem}_probability.png", probability_image)
        save_png(output_dir / "overlay" / f"{stem}_overlay.png", overlay)
        if save_debug_input:
            save_png(output_dir / "debug_input" / f"{stem}_input_8bit.png", scaled_array)
    print(f"Saved results for {image_path.name}")


def main():
    args = parse_args()
    configure_gpu()

    output_dir = Path(args.output)
    for subdir in ("bw", "probability", "overlay"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    if args.save_debug_input:
        (output_dir / "debug_input").mkdir(parents=True, exist_ok=True)

    image_paths = list_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No input images found in {args.input}")

    weights_path = resolve_weights_path(args.weights)
    model = load_model(str(weights_path))
    for image_path in image_paths:
        segment_image(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            scale=args.scale,
            overlap=args.overlap,
            threshold=args.threshold,
            batch_size=args.batch_size,
            output_original_size=args.output_original_size,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            save_debug_input=args.save_debug_input,
            save_png_outputs=args.save_png,
        )

    print(f"All done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
