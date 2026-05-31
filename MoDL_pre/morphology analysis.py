import io
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import numpy as np
import cv2
from scipy.ndimage import morphology
from skimage import measure
from skimage.morphology import thin
from skimage.measure import regionprops
from scipy.stats import kurtosis
from scipy.stats import skew
from skimage.morphology import convex_hull_image
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPLOAD_PATH = PROJECT_ROOT / "final_results" / "bw"
DEFAULT_SAVE_FOLDER = PROJECT_ROOT / "results"
DEFAULT_PATCH_RESULTS_FOLDER = PROJECT_ROOT / "final_results" / "512x512_pixels"
WORKER_MITO_LABELS = None
WORKER_IMAGE_SHAPE = None


def init_mito_worker(mito_labels, image_shape):
    global WORKER_MITO_LABELS
    global WORKER_IMAGE_SHAPE
    WORKER_MITO_LABELS = mito_labels
    WORKER_IMAGE_SHAPE = image_shape


def print_progress_bar(current, total, prefix="", width=40):
    if total <= 0:
        return
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100 * current / total
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({percent:5.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def mito_prop_to_task(index, prop):
    return {
        "index": index,
        "label": prop.label,
        "area": prop.area,
        "eccentricity": prop.eccentricity,
        "equivalent_diameter": prop.equivalent_diameter,
        "euler_number": prop.euler_number,
        "extent": prop.extent,
        "major_axis_length": prop.major_axis_length,
        "minor_axis_length": prop.minor_axis_length,
        "orientation": prop.orientation,
        "perimeter": prop.perimeter,
        "solidity": prop.solidity,
        "centroid": prop.centroid,
    }


def process_mito_prop(task):
    label = task["label"]
    label_mask = np.zeros(WORKER_IMAGE_SHAPE, dtype="uint8")
    label_mask[WORKER_MITO_LABELS == label] = 255

    number_branches = 0
    total_branch_length = 0
    mean_branch_length = 0
    median_branch_length = 0
    std_branch_length = 0
    mean_branch_angle = 0
    median_branch_angle = 0
    std_branch_angle = 0
    total_density = 0
    average_density = 0
    median_density = 0

    try:
        skeleton = thin(label_mask)
        skeleton = 255 * skeleton
        branch_points = getSkeletonIntersection(skeleton)

        branch_point_mask = np.zeros(shape=WORKER_IMAGE_SHAPE, dtype=np.uint8)
        for x_pos, y_pos in branch_points:
            branch_point_mask[y_pos, x_pos] = 255

        kernel = np.ones((3, 3), np.uint8)
        dilated_branch_points = cv2.dilate(branch_point_mask, kernel, iterations=1)

        branch_length_matrix = skeleton - dilated_branch_points
        branch_matrix = branch_length_matrix > 0
        branch_labels = measure.label(np.array(branch_matrix), connectivity=2)
        number_branches = branch_labels.max()
        branch_props = regionprops(branch_labels)

        branch_length = []
        branch_angle = []
        dist_transform = cv2.distanceTransform(label_mask, cv2.DIST_L2, 5)
        positive_distances = dist_transform[dist_transform > 0]

        for branch_prop in branch_props:
            branch_length.append(branch_prop.area + 4)
            branch_angle.append(branch_prop.orientation)

        if len(branch_length) == 1:
            branch_length[0] = task["major_axis_length"]

        if branch_length:
            branch_angle = np.multiply(branch_angle, (180 / np.pi))
            total_branch_length = np.sum(branch_length)
            mean_branch_length = np.mean(branch_length)
            median_branch_length = np.median(branch_length)
            std_branch_length = np.std(branch_length)
            mean_branch_angle = np.mean(branch_angle)
            median_branch_angle = np.median(branch_angle)
            std_branch_angle = np.std(branch_angle)

        if positive_distances.size:
            total_density = np.sum(positive_distances)
            average_density = np.mean(positive_distances)
            median_density = np.median(positive_distances)

    except Exception:
        pass

    centroid = task["centroid"]
    return {
        "index": task["index"],
        "area": task["area"],
        "eccentricity": task["eccentricity"],
        "equivalent_diameter": task["equivalent_diameter"],
        "euler_number": task["euler_number"],
        "extent": task["extent"],
        "major_axis_length": task["major_axis_length"],
        "minor_axis_length": task["minor_axis_length"],
        "orientation": task["orientation"],
        "perimeter": task["perimeter"],
        "solidity": task["solidity"],
        "centroid": centroid,
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
        "branch_count": number_branches,
        "total_branch_length": total_branch_length,
        "mean_branch_length": mean_branch_length,
        "median_branch_length": median_branch_length,
        "std_branch_length": std_branch_length,
        "mean_branch_angle": mean_branch_angle,
        "median_branch_angle": median_branch_angle,
        "std_branch_angle": std_branch_angle,
        "total_density": total_density,
        "average_density": average_density,
        "median_density": median_density,
    }


def process_mito_props_parallel(file_name, img, mito_labels, mito_props):
    tasks = [mito_prop_to_task(index, prop) for index, prop in enumerate(mito_props) if prop.area > 16]
    if not tasks:
        return []

    results = []
    max_workers = os.cpu_count() or 1
    progress_prefix = f"Mitochondria {file_name}"
    print_progress_bar(0, len(tasks), progress_prefix)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_mito_worker,
        initargs=(mito_labels, img.shape),
    ) as executor:
        futures = [executor.submit(process_mito_prop, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            print_progress_bar(completed, len(tasks), progress_prefix)

    return sorted(results, key=lambda result: result["index"])


def load_and_skeletonize_image(file_path):
    image = io.imread(file_path)
    binary_image = image > 0
    skeleton = morphology.skeletonize(binary_image)
    return skeleton


def eight_neighbors(x, y, image):
    VIII_neighbors = [image[x, y - 1], image[x - 1, y - 1], image[x - 1, y], image[x - 1, y + 1],
                      image[x, y + 1], image[x + 1, y + 1], image[x + 1, y], image[x + 1, y - 1]]
    return VIII_neighbors


def getSkeletonIntersection(skeleton):
    validIntersection = [[0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1],
                         [0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1],
                         [1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0],
                         [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1],
                         [1, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0],
                         [1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1],
                         [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1],
                         [0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0],
                         [0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0],
                         [1, 0, 1, 1, 0, 1, 0, 0]];
    image = skeleton.copy();
    image = image / 255;
    intersections = list();
    for x in range(1, len(image) - 1):
        for y in range(1, len(image[x]) - 1):
            if image[x][y] == 1:
                neighbours = eight_neighbors(x, y, image);
                if neighbours in validIntersection:
                    intersections.append((y, x));

    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < 10 ** 2) and (point1 != point2):
                intersections.remove(point2);

    intersections = list(set(intersections));
    return intersections;


def measurement(directory_path, save_path):
    directory_path = Path(directory_path).resolve()
    save_path = Path(save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    DEFAULT_PATCH_RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    if not directory_path.is_dir():
        raise FileNotFoundError(f"Input folder not found: {directory_path}")

    database = pd.DataFrame([[0] * 103],
                            columns=['cell_name', 'cell_mean_mito_area_(pixels_squared)',
                                     'cell_median_mito_area_(pixels_squared)',
                                     'cell_std_mito_area_(pixels_squared)', 'cell_mean_mito_eccentricity',
                                     'cell_median_mito_eccentricity', 'cell_std_mito_eccentricity',
                                     'cell_mean_mito_equi_diameter_(pixels)', 'cell_median_mito_equi_diameter_(pixels)',
                                     'cell_std_mito_equi_diameter_(pixels)', 'cell_mean_mito_euler_number',
                                     'cell_std_mito_euler_number',
                                     'cell_mean_mito_extent',
                                     'cell_median_mito_extent', 'cell_std_mito_extent',
                                     'cell_mean_mito_major_axis_(pixels)',
                                     'cell_median_mito_major_axis_(pixels)', 'cell_std_mito_major_axis_(pixels)',
                                     'cell_mean_mito_minor_axis_(pixels)',
                                     'cell_median_mito_minor_axis_(pixels)', 'cell_std_mito_minor_axis_(pixels)',
                                     'cell_mean_mito_orientation_(degrees)',
                                     'cell_median_mito_orientation_(degrees)', 'cell_std_mito_orientation_(degrees)',
                                     'cell_mean_mito_perimeter_(pixels)',
                                     'cell_median_mito_perimeter_(pixels)', 'cell_std_mito_perimeter_(pixels)',
                                     'cell_mean_mito_solidity',
                                     'cell_median_mito_solidity', 'cell_std_mito_solidity',
                                     'cell_mean_mito_centroid_x_(pixels)',
                                     'cell_median_mito_centroid_x_(pixels)', 'cell_std_mito_centroid_x_(pixels)',
                                     'cell_mean_mito_centroid_y_(pixels)',
                                     'cell_median_mito_centroid_y_(pixels)', 'cell_std_mito_centroid_y_(pixels)',
                                     'cell_mean_mito_distance_(pixels)',
                                     'cell_median_mito_distance_(pixels)', 'cell_std_mito_distance_(pixels)',
                                     'cell_mean_mito_weighted_cent_x_(pixels)',
                                     'cell_median_mito_weighted_cent_x_(pixels)',
                                     'cell_std_mito_weighted_cent_x_(pixels)',
                                     'cell_mean_mito_weighted_cent_y_(pixels)',
                                     'cell_median_mito_weighted_cent_y_(pixels)',
                                     'cell_std_mito_weighted_cent_y_(pixels)',
                                     'cell_mean_mito_weighted_distance_(pixels)',
                                     'cell_median_mito_weighted_distance_(pixels)',
                                     'cell_std_mito_weighted_distance_(pixels)',
                                     'cell_mean_mito_form_factor', 'cell_median_mito_form_factor',
                                     'cell_std_mito_form_factor', 'cell_mean_mito_roundness',
                                     'cell_median_mito_roundness',
                                     'cell_std_mito_roundness', 'cell_mean_mito_branch_count',
                                     'cell_std_mito_branch_count', 'cell_mean_mito_mean_branch_length_(pixels)',
                                     'cell_median_mito_mean_branch_length_(pixels)',
                                     'cell_std_mito_mean_branch_length_(pixels)',
                                     'cell_mean_mito_total_branch_length_(pixels)',
                                     'cell_median_mito_total_branch_length_(pixels)',
                                     'cell_std_mito_total_branch_length_(pixels)',
                                     'cell_mean_mito_median_branch_length_(pixels)',
                                     'cell_median_mito_median_branch_length_(pixels)',
                                     'cell_std_mito_median_branch_length_(pixels)',
                                     'cell_mean_mito_std_branch_length_(pixels)',
                                     'cell_std_mito_std_branch_length_(degrees)',
                                     'cell_mean_mito_mean_branch_angle_(degrees)',
                                     'cell_median_mito_mean_branch_angle_(degrees)',
                                     'cell_std_mito_mean_branch_angle_(degrees)',
                                     'cell_mean_mito_median_branch_angle_(degrees)',
                                     'cell_median_mito_median_branch_angle_(degrees)',
                                     'cell_std_mito_median_branch_angle_(degrees)',
                                     'cell_mean_mito_std_branch_angle_(degrees)',
                                     'cell_std_mito_std_branch_angle_(degrees)',
                                     'cell_mean_mito_total_density_(pixels)', 'cell_median_mito_total_density_(pixels)',
                                     'cell_std_mito_total_density', 'cell_mean_mito_average_density_(pixels)',
                                     'cell_median_mito_average_density_(pixels)',
                                     'cell_std_mito_average_density_(pixels)',
                                     'cell_mean_mito_median_density_(pixels)', 'cell_median_mito_median_density',
                                     'cell_std_mito_median_density_(pixels)', 'cell_kurtosis_x',
                                     'cell_weighted_kurtosis_x',
                                     'cell_kurtosis_y', 'cell_weighted_kurtosis_y', 'cell_kurtosis_squared',
                                     'cell_weighted_kurtosis_squared', 'cell_skewness_x', 'cell_weighted_skewness_x',
                                     'cell_skewness_y', 'cell_weighted_skewness_y', 'cell_skewness_squared',
                                     'cell_weighted_skewness_squared', 'cell_network_orientation_(degrees)',
                                     'cell_network_major_axis_(pixels)',
                                     'cell_network_minor_axis_(pixels)', 'cell_network_eccentricity',
                                     'cell_network_effective_extent', 'cell_network_effective_solidity',
                                     'cell_network_fractal_dimension'])

    database_raw = pd.DataFrame([[0] * 39], columns=['cell_name', 'resize_factor', 'mito_area', 'mito_centroid',
                                                     'mito_eccentricity',
                                                     'mito_equi_diameter', 'mito_euler_number', 'mito_extent',
                                                     'mito_major_axis',
                                                     'mito_minor_axis', 'mito_orientation', 'mito_perimeter',
                                                     'mito_solidity',
                                                     'mito_centroid_x', 'mito_centroid_y', 'mito_distance',
                                                     'mito_weighted_cent_x',
                                                     'mito_weighted_cent_y', 'mito_weighted_distance',
                                                     'mito_form_factor',
                                                     'mito_roundness', 'mito_branch_count', 'mito_total_branch_length',
                                                     'mito_mean_branch_length', 'mito_median_branch_length',
                                                     'mito_std_branch_length',
                                                     'mito_mean_branch_angle', 'mito_median_branch_angle',
                                                     'mito_std_branch_angle',
                                                     'mito_total_density', 'mito_average_density',
                                                     'mito_median_density',
                                                     'mito_branch_count', 'mito_distance', 'mito_weighted_cent_x',
                                                     'mito_weighted_cent_y',
                                                     'mito_weighted_distance', 'mito_form_factor', 'mito_roundness'])

    test_num = 0
    files = os.listdir(directory_path)  # Get the file list from the directory
    total_file_count = len(files)
    image_extensions = ['.jpg', '.png', '.tif', '.tiff']

    for file in files:
        test_num += 1
        if any(file.lower().endswith(ext) for ext in image_extensions):
            try:
                file_path = directory_path / file
                img = cv2.imread(str(file_path))
                img = img[:, :, 0]
                print("Test", file, f'Test [{np.round(100 * (test_num / total_file_count), 2)}%]')
                scale = 1

                # Individual mitochondria analysis
                mito_labels = measure.label(np.array(img), connectivity=2)
                mito_props = regionprops(mito_labels)

                mito_area = []
                mito_centroid = []
                mito_eccentricity = []
                mito_equi_diameter = []
                mito_euler_number = []
                mito_extent = []
                mito_major_axis = []
                mito_minor_axis = []
                mito_orientation = []
                mito_perimeter = []
                mito_solidity = []
                mito_centroid_x = []
                mito_centroid_y = []
                mito_distance = []
                mito_weighted_cent_x = []
                mito_weighted_cent_y = []
                mito_weighted_distance = []
                mito_form_factor = []
                mito_roundness = []
                mito_branch_count = []
                mito_total_branch_length = []
                mito_mean_branch_length = []
                mito_median_branch_length = []
                mito_std_branch_length = []
                mito_mean_branch_angle = []
                mito_median_branch_angle = []
                mito_std_branch_angle = []
                mito_total_density = []
                mito_average_density = []
                mito_median_density = []
                mito_branch_count = []

                mito_results = process_mito_props_parallel(file, img, mito_labels, mito_props)
                for mito_result in mito_results:
                    mito_area.append(mito_result["area"])
                    mito_eccentricity.append(mito_result["eccentricity"])
                    mito_equi_diameter.append(mito_result["equivalent_diameter"])
                    mito_euler_number.append(mito_result["euler_number"])
                    mito_extent.append(mito_result["extent"])
                    mito_major_axis.append(mito_result["major_axis_length"])
                    mito_minor_axis.append(mito_result["minor_axis_length"])
                    mito_orientation.append(mito_result["orientation"])
                    mito_perimeter.append(mito_result["perimeter"])
                    mito_solidity.append(mito_result["solidity"])
                    mito_centroid.append(mito_result["centroid"])
                    mito_centroid_x.append(mito_result["centroid_x"])
                    mito_centroid_y.append(mito_result["centroid_y"])
                    mito_branch_count.append(mito_result["branch_count"])
                    mito_total_branch_length.append(mito_result["total_branch_length"])
                    mito_mean_branch_length.append(mito_result["mean_branch_length"])
                    mito_median_branch_length.append(mito_result["median_branch_length"])
                    mito_std_branch_length.append(mito_result["std_branch_length"])
                    mito_mean_branch_angle.append(mito_result["mean_branch_angle"])
                    mito_median_branch_angle.append(mito_result["median_branch_angle"])
                    mito_std_branch_angle.append(mito_result["std_branch_angle"])
                    mito_total_density.append(mito_result["total_density"])
                    mito_average_density.append(mito_result["average_density"])
                    mito_median_density.append(mito_result["median_density"])

                mito_area = np.multiply(np.power(scale, 2), mito_area)
                mito_equi_diameter = np.multiply(scale, mito_equi_diameter)
                mito_major_axis = np.multiply(scale, mito_major_axis)
                mito_minor_axis = np.multiply(scale, mito_minor_axis)
                mito_perimeter = np.multiply(scale, mito_perimeter)
                mito_centroid_x = np.multiply(scale, mito_centroid_x)
                mito_centroid_y = np.multiply(scale, mito_centroid_y)

                mito_distance = np.sqrt(np.power(mito_centroid_x, 2) + np.power(mito_centroid_y, 2))
                mito_weighted_cent_x = np.divide(np.multiply(mito_centroid_x, mito_area), np.sum(mito_area))
                mito_weighted_cent_y = np.divide(np.multiply(mito_centroid_y, mito_area), np.sum(mito_area))
                mito_weighted_distance = np.sqrt(np.power(mito_weighted_cent_x, 2) + np.power(mito_weighted_cent_y, 2))
                mito_form_factor = (np.divide(np.power(mito_perimeter, 2), mito_area)) / (4 * np.pi)
                mito_roundness = ((4 / np.pi) * np.divide(mito_area, np.power(mito_major_axis, 2)))

                mito_total_branch_length = np.multiply(scale, mito_total_branch_length)
                mito_mean_branch_length = np.multiply(scale, mito_mean_branch_length)
                mito_median_branch_length = np.multiply(scale, mito_median_branch_length)
                mito_std_branch_length = np.multiply(scale, mito_std_branch_length)
                mito_total_density = np.multiply(scale, mito_total_density)
                mito_average_density = np.multiply(scale, mito_average_density)
                mito_median_density = np.multiply(scale, mito_median_density)

                # Per-image summary
                cell_mito_count = len(mito_area)
                cell_total_mito_area = np.sum(mito_area)
                cell_mean_mito_area = np.mean(mito_area)
                cell_median_mito_area = np.median(mito_area)
                cell_std_mito_area = np.std(mito_area)
                cell_mean_mito_eccentricity = np.mean(mito_eccentricity)
                cell_median_mito_eccentricity = np.median(mito_eccentricity)
                cell_std_mito_eccentricity = np.std(mito_eccentricity)
                cell_mean_mito_equi_diameter = np.mean(mito_equi_diameter)
                cell_median_mito_equi_diameter = np.median(mito_equi_diameter)
                cell_std_mito_equi_diameter = np.std(mito_equi_diameter)
                cell_mean_mito_euler_number = np.mean(mito_euler_number)
                cell_median_mito_euler_number = np.median(mito_euler_number)
                cell_std_mito_euler_number = np.std(mito_euler_number)
                cell_mean_mito_extent = np.mean(mito_extent)
                cell_median_mito_extent = np.median(mito_extent)
                cell_std_mito_extent = np.std(mito_extent)
                cell_mean_mito_major_axis = np.mean(mito_major_axis)
                cell_median_mito_major_axis = np.median(mito_major_axis)
                cell_std_mito_major_axis = np.std(mito_major_axis)
                cell_mean_mito_minor_axis = np.mean(mito_minor_axis)
                cell_median_mito_minor_axis = np.median(mito_minor_axis)
                cell_std_mito_minor_axis = np.std(mito_minor_axis)
                cell_mean_mito_orientation = np.mean(mito_orientation)
                cell_median_mito_orientation = np.median(mito_orientation)
                cell_std_mito_orientation = np.std(mito_orientation)
                cell_mean_mito_perimeter = np.mean(mito_perimeter)
                cell_median_mito_perimeter = np.median(mito_perimeter)
                cell_std_mito_perimeter = np.std(mito_perimeter)
                cell_mean_mito_solidity = np.mean(mito_solidity)
                cell_median_mito_solidity = np.median(mito_solidity)
                cell_std_mito_solidity = np.std(mito_solidity)
                cell_mean_mito_centroid_x = np.mean(mito_centroid_x)
                cell_median_mito_centroid_x = np.median(mito_centroid_x)
                cell_std_mito_centroid_x = np.std(mito_centroid_x)
                cell_mean_mito_centroid_y = np.mean(mito_centroid_y)
                cell_median_mito_centroid_y = np.median(mito_centroid_y)
                cell_std_mito_centroid_y = np.std(mito_centroid_y)
                cell_mean_mito_distance = np.mean(mito_distance)
                cell_median_mito_distance = np.median(mito_distance)
                cell_std_mito_distance = np.std(mito_distance)
                cell_mean_mito_weighted_cent_x = np.mean(mito_weighted_cent_x)
                cell_median_mito_weighted_cent_x = np.median(mito_weighted_cent_x)
                cell_std_mito_weighted_cent_x = np.std(mito_weighted_cent_x)
                cell_mean_mito_weighted_cent_y = np.mean(mito_weighted_cent_y)
                cell_median_mito_weighted_cent_y = np.median(mito_weighted_cent_y)
                cell_std_mito_weighted_cent_y = np.std(mito_weighted_cent_y)
                cell_mean_mito_weighted_distance = np.mean(mito_weighted_distance)
                cell_median_mito_weighted_distance = np.median(mito_weighted_distance)
                cell_std_mito_weighted_distance = np.std(mito_weighted_distance)
                cell_mean_mito_form_factor = np.mean(mito_form_factor)
                cell_median_mito_form_factor = np.median(mito_form_factor)
                cell_std_mito_form_factor = np.std(mito_form_factor)
                cell_mean_mito_roundness = np.mean(mito_roundness)
                cell_median_mito_roundness = np.median(mito_roundness)
                cell_std_mito_roundness = np.std(mito_roundness)
                cell_mean_mito_branch_count = np.mean(mito_branch_count)
                cell_median_mito_branch_count = np.median(mito_branch_count)
                cell_std_mito_branch_count = np.std(mito_branch_count)
                cell_mean_mito_mean_branch_length = np.mean(mito_mean_branch_length)
                cell_median_mito_mean_branch_length = np.median(mito_mean_branch_length)
                cell_std_mito_mean_branch_length = np.std(mito_mean_branch_length)
                cell_mean_mito_total_branch_length = np.mean(mito_total_branch_length)
                cell_median_mito_total_branch_length = np.median(mito_total_branch_length)
                cell_std_mito_total_branch_length = np.std(mito_total_branch_length)
                cell_mean_mito_median_branch_length = np.mean(mito_median_branch_length)
                cell_median_mito_median_branch_length = np.median(mito_median_branch_length)
                cell_std_mito_median_branch_length = np.std(mito_median_branch_length)
                cell_mean_mito_std_branch_length = np.mean(mito_std_branch_length)
                cell_median_mito_std_branch_length = np.median(mito_std_branch_length)
                cell_std_mito_std_branch_length = np.std(mito_std_branch_length)
                cell_mean_mito_mean_branch_angle = np.mean(mito_mean_branch_angle)
                cell_median_mito_mean_branch_angle = np.median(mito_mean_branch_angle)
                cell_std_mito_mean_branch_angle = np.std(mito_mean_branch_angle)
                cell_mean_mito_median_branch_angle = np.mean(mito_median_branch_angle)
                cell_median_mito_median_branch_angle = np.median(mito_median_branch_angle)
                cell_std_mito_median_branch_angle = np.std(mito_median_branch_angle)
                cell_mean_mito_std_branch_angle = np.mean(mito_std_branch_angle)
                cell_median_mito_std_branch_angle = np.median(mito_std_branch_angle)
                cell_std_mito_std_branch_angle = np.std(mito_std_branch_angle)
                cell_mean_mito_total_density = np.mean(mito_total_density)
                cell_median_mito_total_density = np.median(mito_total_density)
                cell_std_mito_total_density = np.std(mito_total_density)
                cell_mean_mito_average_density = np.mean(mito_average_density)
                cell_median_mito_average_density = np.median(mito_average_density)
                cell_std_mito_average_density = np.std(mito_average_density)
                cell_mean_mito_median_density = np.mean(mito_median_density)
                cell_median_mito_median_density = np.median(mito_median_density)
                cell_std_mito_median_density = np.std(mito_median_density)
                cell_kurtosis_x = kurtosis(mito_centroid_x)
                cell_weighted_kurtosis_x = kurtosis(mito_weighted_cent_x)
                cell_kurtosis_y = kurtosis(mito_centroid_y)
                cell_weighted_kurtosis_y = kurtosis(mito_weighted_cent_y)
                cell_kurtosis_squared = np.add(np.power(cell_kurtosis_x, 2), np.power(cell_kurtosis_y, 2))
                cell_weighted_kurtosis_squared = np.add(np.power(cell_weighted_kurtosis_x, 2),
                                                        np.power(cell_weighted_kurtosis_y, 2))
                cell_skewness_x = skew(mito_centroid_x)
                cell_weighted_skewness_x = skew(mito_weighted_cent_x)
                cell_skewness_y = skew(mito_centroid_y)
                cell_weighted_skewness_y = skew(mito_weighted_cent_y)
                cell_skewness_squared = np.add(np.power(cell_skewness_x, 2), np.power(cell_skewness_y, 2))
                cell_weighted_skewness_squared = np.add(np.power(cell_weighted_skewness_x, 2),
                                                        np.power(cell_weighted_skewness_y, 2))

                chull = convex_hull_image(img)

                cell_labels = measure.label(np.array(chull), connectivity=2)
                cell_props = regionprops(cell_labels)
                cell_network_orientation = cell_props[0].orientation * 180 / np.pi
                cell_network_major_axis = cell_props[0].major_axis_length
                cell_network_minor_axis = cell_props[0].minor_axis_length
                cell_network_eccentricity = cell_props[0].eccentricity

                cell_scaled_area = np.multiply(np.power(scale, 2), cell_props[0].area)
                cell_network_effective_extent = (np.sum(mito_area) / cell_scaled_area) * cell_props[0].extent
                cell_network_effective_solidity = np.sum(mito_area) / cell_scaled_area

                cell_network_major_axis = np.multiply(scale, cell_network_major_axis)
                cell_network_minor_axis = np.multiply(scale, cell_network_minor_axis)

                pixels = []

                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        if img[i, j] > 0:
                            pixels.append((i, j))

                Lx = img.shape[1]
                Ly = img.shape[0]

                pixels = np.array(pixels)

                scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
                Ns = []
                for scale1 in scales:
                    H, edges = np.histogramdd(pixels, bins=(np.arange(0, Lx, scale1), np.arange(0, Ly, scale1)))
                    Ns.append(np.sum(H > 0))

                coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
                cell_network_fractal_dimension = -coeffs[0]

                temp_dataset = pd.DataFrame([[file, cell_mean_mito_area,
                                              cell_median_mito_area, cell_std_mito_area, cell_mean_mito_eccentricity,
                                              cell_median_mito_eccentricity, cell_std_mito_eccentricity,
                                              cell_mean_mito_equi_diameter, cell_median_mito_equi_diameter,
                                              cell_std_mito_equi_diameter, cell_mean_mito_euler_number,
                                              cell_std_mito_euler_number,
                                              cell_mean_mito_extent, cell_median_mito_extent, cell_std_mito_extent,
                                              cell_mean_mito_major_axis, cell_median_mito_major_axis,
                                              cell_std_mito_major_axis,
                                              cell_mean_mito_minor_axis, cell_median_mito_minor_axis,
                                              cell_std_mito_minor_axis,
                                              cell_mean_mito_orientation, cell_median_mito_orientation,
                                              cell_std_mito_orientation,
                                              cell_mean_mito_perimeter, cell_median_mito_perimeter,
                                              cell_std_mito_perimeter,
                                              cell_mean_mito_solidity, cell_median_mito_solidity,
                                              cell_std_mito_solidity,
                                              cell_mean_mito_centroid_x, cell_median_mito_centroid_x,
                                              cell_std_mito_centroid_x,
                                              cell_mean_mito_centroid_y, cell_median_mito_centroid_y,
                                              cell_std_mito_centroid_y,
                                              cell_mean_mito_distance, cell_median_mito_distance,
                                              cell_std_mito_distance,
                                              cell_mean_mito_weighted_cent_x, cell_median_mito_weighted_cent_x,
                                              cell_std_mito_weighted_cent_x, cell_mean_mito_weighted_cent_y,
                                              cell_median_mito_weighted_cent_y, cell_std_mito_weighted_cent_y,
                                              cell_mean_mito_weighted_distance, cell_median_mito_weighted_distance,
                                              cell_std_mito_weighted_distance, cell_mean_mito_form_factor,
                                              cell_median_mito_form_factor, cell_std_mito_form_factor,
                                              cell_mean_mito_roundness, cell_median_mito_roundness,
                                              cell_std_mito_roundness,
                                              cell_mean_mito_branch_count,
                                              cell_std_mito_branch_count, cell_mean_mito_mean_branch_length,
                                              cell_median_mito_mean_branch_length, cell_std_mito_mean_branch_length,
                                              cell_mean_mito_total_branch_length, cell_median_mito_total_branch_length,
                                              cell_std_mito_total_branch_length, cell_mean_mito_median_branch_length,
                                              cell_median_mito_median_branch_length, cell_std_mito_median_branch_length,
                                              cell_mean_mito_std_branch_length,
                                              cell_std_mito_std_branch_length, cell_mean_mito_mean_branch_angle,
                                              cell_median_mito_mean_branch_angle, cell_std_mito_mean_branch_angle,
                                              cell_mean_mito_median_branch_angle, cell_median_mito_median_branch_angle,
                                              cell_std_mito_median_branch_angle, cell_mean_mito_std_branch_angle,
                                              cell_std_mito_std_branch_angle,
                                              cell_mean_mito_total_density, cell_median_mito_total_density,
                                              cell_std_mito_total_density, cell_mean_mito_average_density,
                                              cell_median_mito_average_density, cell_std_mito_average_density,
                                              cell_mean_mito_median_density, cell_median_mito_median_density,
                                              cell_std_mito_median_density, cell_kurtosis_x, cell_weighted_kurtosis_x,
                                              cell_kurtosis_y, cell_weighted_kurtosis_y, cell_kurtosis_squared,
                                              cell_weighted_kurtosis_squared, cell_skewness_x, cell_weighted_skewness_x,
                                              cell_skewness_y, cell_weighted_skewness_y, cell_skewness_squared,
                                              cell_weighted_skewness_squared, cell_network_orientation,
                                              cell_network_major_axis,
                                              cell_network_minor_axis, cell_network_eccentricity,
                                              cell_network_effective_extent,
                                              cell_network_effective_solidity, cell_network_fractal_dimension]],
                                            columns=['cell_name', 'cell_mean_mito_area_(pixels_squared)',
                                                     'cell_median_mito_area_(pixels_squared)',
                                                     'cell_std_mito_area_(pixels_squared)',
                                                     'cell_mean_mito_eccentricity',
                                                     'cell_median_mito_eccentricity', 'cell_std_mito_eccentricity',
                                                     'cell_mean_mito_equi_diameter_(pixels)',
                                                     'cell_median_mito_equi_diameter_(pixels)',
                                                     'cell_std_mito_equi_diameter_(pixels)',
                                                     'cell_mean_mito_euler_number',
                                                     'cell_std_mito_euler_number',
                                                     'cell_mean_mito_extent',
                                                     'cell_median_mito_extent', 'cell_std_mito_extent',
                                                     'cell_mean_mito_major_axis_(pixels)',
                                                     'cell_median_mito_major_axis_(pixels)',
                                                     'cell_std_mito_major_axis_(pixels)',
                                                     'cell_mean_mito_minor_axis_(pixels)',
                                                     'cell_median_mito_minor_axis_(pixels)',
                                                     'cell_std_mito_minor_axis_(pixels)',
                                                     'cell_mean_mito_orientation_(degrees)',
                                                     'cell_median_mito_orientation_(degrees)',
                                                     'cell_std_mito_orientation_(degrees)',
                                                     'cell_mean_mito_perimeter_(pixels)',
                                                     'cell_median_mito_perimeter_(pixels)',
                                                     'cell_std_mito_perimeter_(pixels)', 'cell_mean_mito_solidity',
                                                     'cell_median_mito_solidity', 'cell_std_mito_solidity',
                                                     'cell_mean_mito_centroid_x_(pixels)',
                                                     'cell_median_mito_centroid_x_(pixels)',
                                                     'cell_std_mito_centroid_x_(pixels)',
                                                     'cell_mean_mito_centroid_y_(pixels)',
                                                     'cell_median_mito_centroid_y_(pixels)',
                                                     'cell_std_mito_centroid_y_(pixels)',
                                                     'cell_mean_mito_distance_(pixels)',
                                                     'cell_median_mito_distance_(pixels)',
                                                     'cell_std_mito_distance_(pixels)',
                                                     'cell_mean_mito_weighted_cent_x_(pixels)',
                                                     'cell_median_mito_weighted_cent_x_(pixels)',
                                                     'cell_std_mito_weighted_cent_x_(pixels)',
                                                     'cell_mean_mito_weighted_cent_y_(pixels)',
                                                     'cell_median_mito_weighted_cent_y_(pixels)',
                                                     'cell_std_mito_weighted_cent_y_(pixels)',
                                                     'cell_mean_mito_weighted_distance_(pixels)',
                                                     'cell_median_mito_weighted_distance_(pixels)',
                                                     'cell_std_mito_weighted_distance_(pixels)',
                                                     'cell_mean_mito_form_factor', 'cell_median_mito_form_factor',
                                                     'cell_std_mito_form_factor', 'cell_mean_mito_roundness',
                                                     'cell_median_mito_roundness',
                                                     'cell_std_mito_roundness', 'cell_mean_mito_branch_count',
                                                     'cell_std_mito_branch_count',
                                                     'cell_mean_mito_mean_branch_length_(pixels)',
                                                     'cell_median_mito_mean_branch_length_(pixels)',
                                                     'cell_std_mito_mean_branch_length_(pixels)',
                                                     'cell_mean_mito_total_branch_length_(pixels)',
                                                     'cell_median_mito_total_branch_length_(pixels)',
                                                     'cell_std_mito_total_branch_length_(pixels)',
                                                     'cell_mean_mito_median_branch_length_(pixels)',
                                                     'cell_median_mito_median_branch_length_(pixels)',
                                                     'cell_std_mito_median_branch_length_(pixels)',
                                                     'cell_mean_mito_std_branch_length_(pixels)',
                                                     'cell_std_mito_std_branch_length_(degrees)',
                                                     'cell_mean_mito_mean_branch_angle_(degrees)',
                                                     'cell_median_mito_mean_branch_angle_(degrees)',
                                                     'cell_std_mito_mean_branch_angle_(degrees)',
                                                     'cell_mean_mito_median_branch_angle_(degrees)',
                                                     'cell_median_mito_median_branch_angle_(degrees)',
                                                     'cell_std_mito_median_branch_angle_(degrees)',
                                                     'cell_mean_mito_std_branch_angle_(degrees)',
                                                     'cell_std_mito_std_branch_angle_(degrees)',
                                                     'cell_mean_mito_total_density_(pixels)',
                                                     'cell_median_mito_total_density_(pixels)',
                                                     'cell_std_mito_total_density',
                                                     'cell_mean_mito_average_density_(pixels)',
                                                     'cell_median_mito_average_density_(pixels)',
                                                     'cell_std_mito_average_density_(pixels)',
                                                     'cell_mean_mito_median_density_(pixels)',
                                                     'cell_median_mito_median_density',
                                                     'cell_std_mito_median_density_(pixels)', 'cell_kurtosis_x',
                                                     'cell_weighted_kurtosis_x',
                                                     'cell_kurtosis_y', 'cell_weighted_kurtosis_y',
                                                     'cell_kurtosis_squared',
                                                     'cell_weighted_kurtosis_squared', 'cell_skewness_x',
                                                     'cell_weighted_skewness_x',
                                                     'cell_skewness_y', 'cell_weighted_skewness_y',
                                                     'cell_skewness_squared',
                                                     'cell_weighted_skewness_squared',
                                                     'cell_network_orientation_(degrees)',
                                                     'cell_network_major_axis_(pixels)',
                                                     'cell_network_minor_axis_(pixels)', 'cell_network_eccentricity',
                                                     'cell_network_effective_extent', 'cell_network_effective_solidity',
                                                     'cell_network_fractal_dimension'])

                temp_dataset_raw = pd.DataFrame([[file, scale, mito_area, mito_centroid,
                                                  mito_eccentricity, mito_equi_diameter, mito_euler_number, mito_extent,
                                                  mito_major_axis, mito_minor_axis, mito_orientation, mito_perimeter,
                                                  mito_solidity, mito_centroid_x, mito_centroid_y, mito_distance,
                                                  mito_weighted_cent_x, mito_weighted_cent_y, mito_weighted_distance,
                                                  mito_form_factor, mito_roundness, mito_branch_count,
                                                  mito_total_branch_length,
                                                  mito_mean_branch_length, mito_median_branch_length,
                                                  mito_std_branch_length,
                                                  mito_mean_branch_angle, mito_median_branch_angle,
                                                  mito_std_branch_angle,
                                                  mito_total_density, mito_average_density, mito_median_density,
                                                  mito_branch_count,
                                                  mito_distance, mito_weighted_cent_x, mito_weighted_cent_y,
                                                  mito_weighted_distance,
                                                  mito_form_factor, mito_roundness]],
                                                columns=['cell_name', 'resize_factor', 'mito_area', 'mito_centroid',
                                                         'mito_eccentricity',
                                                         'mito_equi_diameter', 'mito_euler_number', 'mito_extent',
                                                         'mito_major_axis',
                                                         'mito_minor_axis', 'mito_orientation', 'mito_perimeter',
                                                         'mito_solidity',
                                                         'mito_centroid_x', 'mito_centroid_y', 'mito_distance',
                                                         'mito_weighted_cent_x',
                                                         'mito_weighted_cent_y', 'mito_weighted_distance',
                                                         'mito_form_factor',
                                                         'mito_roundness', 'mito_branch_count',
                                                         'mito_total_branch_length',
                                                         'mito_mean_branch_length', 'mito_median_branch_length',
                                                         'mito_std_branch_length',
                                                         'mito_mean_branch_angle', 'mito_median_branch_angle',
                                                         'mito_std_branch_angle',
                                                         'mito_total_density', 'mito_average_density',
                                                         'mito_median_density',
                                                         'mito_branch_count', 'mito_distance', 'mito_weighted_cent_x',
                                                         'mito_weighted_cent_y',
                                                         'mito_weighted_distance', 'mito_form_factor',
                                                         'mito_roundness'])

                database = database.append(temp_dataset, ignore_index=True)
                database_raw = database_raw.append(temp_dataset_raw, ignore_index=True)

            except:
                print('Cann\'t test {0}'.format(file))
    database.drop(database.index[0], inplace=True)
    database.to_csv(save_path / "Distinct image testv.csv", sep=',', index=False)
    database.to_csv(DEFAULT_PATCH_RESULTS_FOLDER / "Distinct image testv.csv", sep=',', index=False)

    database_raw.drop(database_raw.index[0], inplace=True)
    database_raw.to_csv(save_path / "Distinct mitochondria test.tsv", sep='\t', index=False)
    database_raw.to_csv(DEFAULT_PATCH_RESULTS_FOLDER / "Distinct mitochondria test.tsv", sep='\t',
                        index=False)

    print('Test has been completed')


if __name__ == "__main__":
    upload_path = DEFAULT_UPLOAD_PATH
    save_floder = DEFAULT_SAVE_FOLDER
    measurement(upload_path, save_floder)
