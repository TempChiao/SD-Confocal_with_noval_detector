import os
import numpy as np
import pandas as pd
import tifffile
import scipy.io
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from PIL import Image
import matplotlib.pyplot as plt

# skimage
from skimage.io import imread, imsave
from skimage import filters, measure
from skimage.filters import threshold_local, threshold_otsu
from skimage.measure import label, regionprops, ransac
from skimage.transform import AffineTransform, warp

# Parallelization
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

##############################################################################
# Global variables (adjust as needed)
##############################################################################
matlab_path = '/Users/tempchiao/Desktop/Test_SD_confocal'  # Example path to your main data folder
variable_name = 'pixxx_reduction'


##############################################################################
# Function definitions
##############################################################################

def generate_df(root, frame, variable_name):
    """
    Load .mat files from each frame-folder under 'root', extracting a variable
    named 'variable_name' from each .mat file, stacking them along axis 0.
    """
    image_list = []
    for i in range(1, frame + 1):
        mat_filepath = os.path.join(root, f'Frame_{i}', 'worksapce.mat')

        # Load the .mat file
        mat_data = scipy.io.loadmat(mat_filepath)

        # Check if the variable exists
        if variable_name in mat_data:
            specific_data = mat_data[variable_name]
            image_list.append(specific_data)

    # Convert to NumPy array
    image_array = np.array(image_list)
    return image_array


def save_image(path, image, Type):
    """
    Save image in various modes:
        Type=0 -> 'origin.tif'
        Type=1 -> 'binary_mask.tif'
        Type=2 -> 'masked.tif'
    """
    if Type == 0:
        output_file = os.path.join(path, 'origin.tif')
        tifffile.imwrite(output_file, image)
    elif Type == 1:
        output_file = os.path.join(path, 'binary_mask.tif')
        # For a simple binary mask, convert to uint8 {0, 255}:
        imsr = Image.fromarray((image > 0).astype(np.uint8) * 255)
        imsr.save(output_file)
    elif Type == 2:
        output_file = os.path.join(path, 'masked.tif')
        tifffile.imwrite(output_file, image)


def split_tiff_stack(input_tiff, output_dir, name):
    """
    Split a multi-frame TIFF into left (red) and right (green) halves along width,
    and save them as separate TIFF stacks.
    """
    os.makedirs(output_dir, exist_ok=True)
    data = tifffile.imread(input_tiff)  # shape: (num_frames, height, width)

    # Split along width
    num_frames, height, width = data.shape
    half_w = width // 2
    left_stack = data[:, :, :half_w]    # Red
    right_stack = data[:, :, half_w:]   # Green

    # Construct filenames with .tif extension
    red_filename = os.path.join(output_dir, f'Red_{name}.tif')
    green_filename = os.path.join(output_dir, f'Green_{name}.tif')

    # Write them out
    tifffile.imwrite(red_filename, left_stack)
    tifffile.imwrite(green_filename, right_stack)


def threshold_image(image, threshold='otsu'):
    """
    Threshold an image using either 'otsu' or a numeric value.
    Returns a binary (boolean) mask.
    """
    if threshold == 'otsu':
        t = threshold_otsu(image)
        print("Otsu's threshold value:", t)
    else:
        # Ensure threshold is numeric
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be 'otsu' or a numeric value.")
        t = float(threshold)
    binary_image = image > t
    return binary_image


def calculate_pinhole_matrix(green_mask, red_mask, output_path):
    """
    Calculate the pinhole matrix using matched pinholes between red and green masks,
    using Hungarian assignment + RANSAC for robust affine transform fitting.

    Writes:
      - 'generated_green_mask.tif' (aligned red->green mask)
      - 'matched_pinholes.csv'
    into 'output_path'.
    """
    print("Calculating pinhole matrix...")
    print('Red mask shape:',   red_mask.shape)
    print('Green mask shape:', green_mask.shape)

    red_labels = measure.label(red_mask, connectivity=2)
    green_labels = measure.label(green_mask, connectivity=2)
    red_props = regionprops(red_labels)
    green_props = regionprops(green_labels)

    red_centroids = np.array([p.centroid for p in red_props])
    green_centroids = np.array([p.centroid for p in green_props])

    print('Number of pinholes in red:',   len(red_centroids))
    print('Number of pinholes in green:', len(green_centroids))

    # Hungarian assignment (linear_sum_assignment) of pinholes to minimize distance
    cost_mat = distance_matrix(red_centroids, green_centroids)
    row_ind, col_ind = linear_sum_assignment(cost_mat)

    matched_red_points = red_centroids[row_ind]
    matched_green_points = green_centroids[col_ind]

    # RANSAC to find robust affine transform
    (model_robust, inliers) = ransac(
        data=(matched_red_points, matched_green_points),
        model_class=AffineTransform,
        min_samples=3,
        residual_threshold=2.0,
        max_trials=5000
    )

    inlier_indices = np.where(inliers)[0]  # array of True/False
    inlier_red_points   = matched_red_points[inliers]
    inlier_green_points = matched_green_points[inliers]

    n_inliers = np.sum(inliers)
    n_matches = len(inliers)

    print(f"RANSAC: {n_inliers}/{n_matches} inliers after robust fitting.")
    print("Estimated robust affine transform:\n", model_robust.params)

    # Warp the red channel to the green channel
    aligned_red_mask_float = warp(
        red_mask.astype(float),
        inverse_map=model_robust.inverse,  # transform is from red→green, so use inverse
        output_shape=green_mask.shape,
        order=0  # nearest-neighbor (preserve binary)
    )
    aligned_red_mask = (aligned_red_mask_float > 0.5)

    # Save the warped red mask
    aligned_mask_path = os.path.join(output_path, 'generated_green_mask.tif')
    imsave(aligned_mask_path, (aligned_red_mask.astype(np.uint8)*255))

    # Build a DataFrame of matched data
    matched_data = []
    red_props = regionprops(label(red_mask, connectivity=2))    # Recompute just in case
    green_props = regionprops(label(green_mask, connectivity=2))
    for idx in inlier_indices:
        red_i   = row_ind[idx]
        green_j = col_ind[idx]

        # Get regionprops label IDs
        red_label_id   = red_props[red_i].label
        green_label_id = green_props[green_j].label

        # Get centroids if you want to store them
        # (We've already matched them by index, but let's store them.)
        red_centroid   = matched_red_points[idx]
        green_centroid = matched_green_points[idx]

        matched_data.append({
            "RedLabelID": red_label_id,
            "GreenLabelID": green_label_id,
            "RedY": red_centroid[0],
            "RedX": red_centroid[1],
            "GreenY": green_centroid[0],
            "GreenX": green_centroid[1]
        })

    df = pd.DataFrame(matched_data)
    csv_path = os.path.join(output_path, 'matched_pinholes.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved inlier matches to: {csv_path}")

    # (Optional) Visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(red_mask, cmap='Reds')
    axes[0].set_title("Original Red Mask")
    axes[1].imshow(green_mask, cmap='Greens')
    axes[1].set_title("Green Mask")
    axes[2].imshow(aligned_red_mask, cmap='Reds')
    axes[2].set_title("RANSAC-Aligned Red→Green")

    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def analysis_image(green_image, red_image, matched_pinholes, green_mask, red_mask,
                   output_path, experiment_name):
    """
    Analyze (green_image, red_image) time-lapse data, measuring intensities
    in matched pinholes (regions). Writes a CSV to output_path.
    """
    # Dimensions of the TIFF stack
    num_frames, H, W = red_image.shape

    # Label each mask
    green_labels = label(green_mask, connectivity=2)
    red_labels   = label(red_mask,   connectivity=2)

    red_label_ids = matched_pinholes['RedLabelID'].values
    green_label_ids = matched_pinholes['GreenLabelID'].values
    n_matched_pinholes = len(red_label_ids)
    print(f"Loaded {n_matched_pinholes} matched pinholes from CSV file.")

    all_data = []
    for t in range(num_frames):
        red_props_t   = regionprops(red_labels,   intensity_image=red_image[t])
        green_props_t = regionprops(green_labels, intensity_image=green_image[t])

        for (r_label_id, g_label_id) in zip(red_label_ids, green_label_ids):
            r_index = r_label_id - 1
            g_index = g_label_id - 1
            # regionprops is usually sorted by label but ensure consecutive labeling
            if r_index < len(red_props_t) and g_index < len(green_props_t):
                red_area = red_props_t[r_index].area
                green_area = green_props_t[g_index].area
                red_mean_intensity = red_props_t[r_index].mean_intensity
                green_mean_intensity = green_props_t[g_index].mean_intensity
                red_max_intensity = red_props_t[r_index].max_intensity
                green_max_intensity = green_props_t[g_index].max_intensity
                red_centroid   = red_props_t[r_index].centroid
                green_centroid = green_props_t[g_index].centroid

                all_data.append({
                    "Frame": t,
                    "RedLabelID": r_label_id,
                    "GreenLabelID": g_label_id,
                    "RedArea": red_area,
                    "GreenArea": green_area,
                    "RedMeanIntensity": red_mean_intensity,
                    "GreenMeanIntensity": green_mean_intensity,
                    "RedMaxIntensity": red_max_intensity,
                    "GreenMaxIntensity": green_max_intensity,
                    "RedCentroidY": red_centroid[0],
                    "RedCentroidX": red_centroid[1],
                    "GreenCentroidY": green_centroid[0],
                    "GreenCentroidX": green_centroid[1]
                })
            else:
                pass

    # Create and save the DataFrame
    df = pd.DataFrame(all_data)
    print("Head of the DataFrame:")
    print(df.head())

    csv_filename = f"Timelapse_intensity_{experiment_name}.csv"
    csv_fullpath = os.path.join(output_path, csv_filename)
    df.to_csv(csv_fullpath, index=False)
    print(f"Saved analysis results to: {csv_fullpath}")


def process_experiment(experiment,
                       splitted_dir,
                       matched_pinholes,
                       green_mask,
                       red_mask,
                       variable_name,
                       matlab_path):
    """
    Function to be called in parallel for each experiment.
    """
    try:
        print("Processing experiment:", experiment)
        root_exp = os.path.join(matlab_path, experiment)

        # Gather frame folders
        frame_folders = [d for d in os.listdir(root_exp)
                         if os.path.isdir(os.path.join(root_exp, d))]
        frame_count = len(frame_folders)
        print("Frame count:", frame_count)

        # Generate the raw image array
        origin_image = generate_df(root_exp, frame_count, variable_name)
        print("Image shape:", origin_image.shape)

        # Save the stacked array as a TIFF
        save_image(root_exp, origin_image, 0)

        # Split the new origin.tif into Red_xxx.tif and Green_xxx.tif
        origin_path = os.path.join(root_exp, 'origin.tif')
        split_tiff_stack(input_tiff=origin_path,
                         output_dir=splitted_dir,
                         name=experiment)

        # Read them back in for analysis
        green_file = os.path.join(splitted_dir, f"Green_{experiment}.tif")
        red_file   = os.path.join(splitted_dir, f"Red_{experiment}.tif")
        green_image = imread(green_file)
        red_image   = imread(red_file)

        # Do the analysis
        analysis_image(green_image,
                       red_image,
                       matched_pinholes,
                       green_mask,
                       red_mask,
                       output_path=matlab_path,
                       experiment_name=experiment)

        print("Finished:", experiment)

    except Exception as e:
        print(f"Error in experiment {experiment}: {e}")


##############################################################################
# Main script logic (with parallelization)
##############################################################################
if __name__ == "__main__":

    # 1) Generate the combined mask from the pinholes
    mask_path = '/Users/tempchiao/Desktop/Test_SD_confocal/PinHole'
    mask_folders = [d for d in os.listdir(mask_path) if os.path.isdir(os.path.join(mask_path, d))]
    mask_frame = len(mask_folders)

    # Load the data (all frames in mask_path)
    total_mask = generate_df(mask_path, mask_frame, 'pixxx_reduction')
    # Max-project and threshold
    max_total_mask = np.max(total_mask, axis=0)
    binary_total_mask = threshold_image(max_total_mask, threshold='otsu')

    # Save the binary mask
    save_image(matlab_path, binary_total_mask, 1)

    # 2) Split the mask into green_mask and red_mask
    #    (Here, we assume your image width is 128. Adjust indices as needed.)
    green_mask = binary_total_mask[:, 64:]
    red_mask   = binary_total_mask[:, :64]

    # 3) Calculate the pinhole matrix & produce matched pinholes
    calculate_pinhole_matrix(green_mask, red_mask, output_path=matlab_path)
    matched_pinholes_csv = os.path.join(matlab_path, 'matched_pinholes.csv')
    matched_pinholes = pd.read_csv(matched_pinholes_csv)

    # 4) Process experiments in parallel
    experiment_path_list = [
        d for d in os.listdir(matlab_path)
        if os.path.isdir(os.path.join(matlab_path, d)) and d not in ["splitted_tiff"]
    ]

    # Create (or reuse) a subdir for splitted TIFFs
    splitted_dir = os.path.join(matlab_path, 'splitted_tiff')
    os.makedirs(splitted_dir, exist_ok=True)

    # Number of workers to use; you can also pick a smaller number if you don't
    # want to saturate all cores.
    num_workers = multiprocessing.cpu_count()
    print(f"Running in parallel with up to {num_workers} workers...")

    # We employ ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for experiment in experiment_path_list:
            if experiment == "splitted_tiff":
                continue  # skip the splitted directory

            # Submit a parallel task
            futures.append(executor.submit(
                process_experiment,
                experiment,
                splitted_dir,
                matched_pinholes,
                green_mask,
                red_mask,
                variable_name,
                matlab_path
            ))

        # Optionally wait for all tasks to complete
        for future in futures:
            future.result()

    print("All experiments processed.")
