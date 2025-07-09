#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scipy.io
import tifffile
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, ransac
from skimage.transform import AffineTransform
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import concurrent.futures


#############################
# Helper / Utility Functions
#############################

def convert_to_uint8(img):
    """
    Convert any NumPy array to uint8 by min-max scaling to [0, 255].
    If the image is already uint8, return it directly.
    """
    if img is None:
        raise ValueError("Image is None, cannot convert to uint8.")
    if img.dtype == np.uint8:
        return img  # Already uint8

    if img.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    mi, ma = img.min(), img.max()
    if mi == ma:
        # Constant image => avoid division by zero
        return np.zeros_like(img, dtype=np.uint8)

    scale = 255.0 / (ma - mi)
    return ((img - mi) * scale).astype(np.uint8)


def save_image(path, image, Type=0):
    """
    Save image in various modes:
        Type=0 -> grayscale/stack 'origin.tif'
        Type=1 -> 'binary_mask.tif' (0/255)
        Type=2 -> 'masked.tif'
    """
    if Type == 0:
        tifffile.imwrite(path, image, photometric='minisblack')
    elif Type == 1:
        # Convert bool => 0/255
        imsr = Image.fromarray((image > 0).astype(np.uint8) * 255)
        imsr.save(path)
    elif Type == 2:
        tifffile.imwrite(path, image, photometric='minisblack')


def threshold_image(image, threshold='otsu'):
    """
    Threshold an image using either 'otsu' or a numeric value.
    Returns a binary (boolean) mask.
    """
    if threshold == 'otsu':
        t = threshold_otsu(image)
        print("Otsu's threshold value:", t)
    else:
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be 'otsu' or a numeric value.")
        t = float(threshold)
    return image > t


def generate_df(root, variable_name):
    """
    Collect .mat files from folder 'root', load the variable 'variable_name',
    and stack them along axis=0 => (T, H, W).
    """
    image_list = []
    for file in os.listdir(root):
        if file.endswith('.mat') and not file.startswith('.'):
            mat_data = scipy.io.loadmat(os.path.join(root, file))
            if variable_name in mat_data:
                specific_data = mat_data[variable_name]
                image_list.append(specific_data)
    return np.array(image_list)  # shape (n, H, W) if consistent


def remove_hot_pixel(image, threshold=500):
    """
    Zero out any pixel above 'threshold' in-place.
    """
    image[image > threshold] = 0
    return image


# def remove_hot_pixel(image, percentile = 99.5):
#     """
#     Removes the top (100 - percentile)% brightest pixels by setting them to 0.
#     e.g. if percentile = 99, it removes the top 1% brightest pixels.
#     """
#     # Make a copy if you don't want to modify 'image' in place
#     new_image = image.copy()
#
#     # Find the intensity value at the specified percentile
#     threshold_val = np.percentile(new_image, percentile)
#
#     # Zero out anything above this percentile threshold
#     new_image[new_image > threshold_val] = 0
#
#     return new_image


#########################
# Registration Functions
#########################

def calculate_transform_green_to_red(green_image, red_image):
    """
    Estimate an affine transform that maps coordinates from the GREEN image
    to the RED image, i.e.  (x_g, y_g) -> (x_r, y_r).

    We'll call the resulting matrix 'matrix_gr'.
    """
    # Ensure uint8
    green_8u = convert_to_uint8(green_image)
    red_8u = convert_to_uint8(red_image)

    # Detect SIFT features
    sift = cv2.SIFT_create()
    kp_g, des_g = sift.detectAndCompute(green_8u, None)
    kp_r, des_r = sift.detectAndCompute(red_8u, None)

    if des_g is None or des_r is None:
        raise ValueError("No descriptors found in green or red image.")

    # BFMatcher + ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des_g, des_r, k=2)

    good = []
    ratio_thresh = 0.7
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 3:
        raise ValueError(f"Not enough matches found! Found {len(good)} good matches.")

    pts_green = np.float32([kp_g[m.queryIdx].pt for m in good])
    pts_red = np.float32([kp_r[m.trainIdx].pt for m in good])

    # Calculate transform via RANSAC
    matrix_gr, inliers = cv2.estimateAffine2D(
        pts_green,  # source
        pts_red,  # destination
        method=cv2.RANSAC,
        ransacReprojThreshold=3
    )
    if matrix_gr is None:
        raise ValueError("Could not estimate an affine transform.")

    return matrix_gr


def invert_affine_transform(matrix_2x3):
    """
    Given a 2x3 affine transform M that does (x_src, y_src) -> (x_dst, y_dst),
    compute the inverse transform that goes from (x_dst, y_dst) -> (x_src, y_src).
    We can use 'cv2.invertAffineTransform'.
    """
    # matrix_2x3 is shape (2,3)
    inv = cv2.invertAffineTransform(matrix_2x3)
    return inv


def apply_transform(image_to_transform, reference_image, matrix_2x3):
    """
    Warp 'image_to_transform' using the given 2x3 matrix so that
    it is in the coordinate space of 'reference_image'.

    The matrix_2x3 is assumed to map from 'image_to_transform' coords to 'reference_image' coords.
    """
    h, w = reference_image.shape[:2]
    return cv2.warpAffine(image_to_transform, matrix_2x3, (w, h))


def forward_transform_points(ptsYX, matrix_2x3):
    """
    Forward transform points in (y, x) format by the 2x3 matrix.
    We'll treat them as row vectors [x, y, 1] * M^T due to warpAffine convention.

    But be mindful that cv2 generally expects (x, y) ordering.
    If 'ptsYX' is shape (N, 2) with (y, x), we reorder to (x, y).
    """
    # Reorder to (x, y)
    pts_xy = ptsYX[:, [1, 0]]
    ones = np.ones((pts_xy.shape[0], 1), dtype=np.float32)
    pts_homo = np.hstack([pts_xy, ones])  # shape (N, 3)

    # Multiply
    transformed_xy = pts_homo @ matrix_2x3.T  # shape (N, 2)

    # Return as (y, x) by reversing
    transformed_yx = transformed_xy[:, [1, 0]]
    return transformed_yx


def backward_transform_points(ptsYX, matrix_2x3):
    """
    If matrix_2x3 maps from G->R, but you have points in the R space you want
    to map back to G, you'd invert the matrix. Or vice versa. This function
    is just a semantically named alias. Typically you'd do:

    backward_matrix = invert_affine_transform(matrix_2x3)
    forward_transform_points(ptsYX, backward_matrix)

    For convenience, let's define it explicitly if you like, but it's
    basically the same as forward_transform_points with the inverted matrix.
    """
    # optional implementation, if needed
    raise NotImplementedError("Define as needed, using invert_affine_transform + forward_transform_points.")


#########################
#  Pinhole Matching, in "aligned" (green) space
#########################

# def match_pinholes_in_aligned_space(
#         green_pinhole_3d,
#         red_pinhole_3d,
#         matrix_gr,
#         output_path
# ):
#     """
#     GOAL:
#       1) We'll keep the green pinhole image in its original space.
#       2) We'll transform (warp) the RED pinhole image into the GREEN coordinate space.
#
#       Then, we do threshold+label in the green image (unwarped)
#       and the red image (warped), match pinholes in that aligned space,
#       and finally map each red pinhole's aligned coordinate back to the
#       original red coordinate system.
#
#     Incoming:
#       matrix_gr : a 2x3 affine transform mapping green->red
#                   (i.e., if we have [x_g, y_g, 1], we get [x_r, y_r]).
#     Steps:
#       - invert that matrix => matrix_rg to warp red->green
#       - label green, label aligned red
#       - Hungarian => RANSAC
#       - for each matched 'red' pinhole in the aligned space =>
#            transform it back to original red coords using matrix_gr
#     """
#     # 1) Project the 3D stacks
#     proj_green = np.mean(green_pinhole_3d, axis=0)
#     proj_red = np.mean(red_pinhole_3d, axis=0)
#
#     # remove hot pixels or saturations
#     remove_hot_pixel(proj_green, 500)
#     remove_hot_pixel(proj_red, 500)
#
#     # convert to uint8
#     proj_green_8u = convert_to_uint8(proj_green)
#     proj_red_8u = convert_to_uint8(proj_red)
#
#     # 2) We have matrix_gr (green->red). We want to warp red->green => invert
#     matrix_rg = invert_affine_transform(matrix_gr)
#
#     # 3) Warp red image into green coords
#     aligned_red_8u = apply_transform(proj_red_8u, proj_green_8u, matrix_rg)
#
#     # (Optional) Save the aligned red image for debugging
#     out_aligned_red = os.path.join(output_path, "pinhole_red_aligned_to_green.tif")
#     save_image(out_aligned_red, aligned_red_8u, Type=0)
#
#     # 4) Threshold + label in green coords
#     green_bin = threshold_image(proj_green_8u, threshold='otsu')
#     red_aligned_bin = threshold_image(aligned_red_8u, threshold='otsu')
#
#     green_labels = label(green_bin, connectivity=2)
#     red_aligned_labels = label(red_aligned_bin, connectivity=2)
#
#     green_props = regionprops(green_labels)
#     red_aligned_props = regionprops(red_aligned_labels)
#
#     green_centroids = np.array([p.centroid for p in green_props])  # (N,2) => (y,x)
#     red_aligned_centroids = np.array([p.centroid for p in red_aligned_props])  # (M,2)
#
#     if len(green_centroids) == 0 or len(red_aligned_centroids) == 0:
#         raise ValueError("No pinholes found in green or aligned red. Check thresholding.")
#
#     # 5) Hungarian assignment in ALIGNED space
#     cost_mat = distance_matrix(red_aligned_centroids, green_centroids)
#     row_ind, col_ind = linear_sum_assignment(cost_mat)
#
#     # matched pairs => (red_aligned_centroids[i], green_centroids[j])
#     matched_red_aligned = red_aligned_centroids[row_ind]
#     matched_green = green_centroids[col_ind]
#
#     # 6) RANSAC to refine matches (ensuring consistent transform in aligned space)
#     #    We'll do an AffineTransform from red_aligned -> green, or vice versa.
#     #    regionprops gives (y, x). skimage.transform prefers (x, y).
#     #    We'll reorder but be consistent about residual_threshold.
#
#     matched_red_aligned_xy = matched_red_aligned[:, ::-1]  # (x, y)
#     matched_green_xy = matched_green[:, ::-1]
#
#     model_robust, inliers = ransac(
#         data=(matched_red_aligned_xy, matched_green_xy),
#         model_class=AffineTransform,
#         min_samples=3,
#         residual_threshold=2.0,
#         max_trials=5000
#     )
#     inlier_indices = np.where(inliers)[0]
#
#     inlier_red_aligned = matched_red_aligned[inliers]  # (y, x)
#     inlier_green = matched_green[inliers]  # (y, x)
#
#     # 7) For each inlier in red ALIGNED coords, map it back
#     #    to the original red coords using the forward transform 'matrix_gr'.
#     #    Because matrix_gr does: (x_g, y_g) -> (x_r, y_r).
#     #    But now we have (y_g, x_g)? Actually, we're in green coords.
#     #    We want to treat them as (y, x) => reorder => (x, y) => apply matrix_gr => (x_r, y_r), reorder => (y_r, x_r).
#
#     # We'll define a helper: forward_transform_points(inlier_point_yx, matrix_gr)
#     # But the tricky part: matrix_gr uses green->red.
#     # So if we have a point in green coords = (y_g, x_g),
#     # then applying matrix_gr "directly" with forward_transform_points should yield the original red coords.
#
#     inlier_red_original = forward_transform_points(inlier_red_aligned, matrix_gr)
#
#     matched_data = []
#     for i, idx in enumerate(inlier_indices):
#         # green pinhole in green coords
#         gy, gx = inlier_green[i]
#         # red pinhole in aligned red coords
#         ry_aligned, rx_aligned = inlier_red_aligned[i]
#         # red pinhole in original red coords
#         ry_orig, rx_orig = inlier_red_original[i]
#
#         matched_data.append({
#             "GreenY": gy,
#             "GreenX": gx,
#             "AlignedRedY": ry_aligned,
#             "AlignedRedX": rx_aligned,
#             "OriginalRedY": ry_orig,
#             "OriginalRedX": rx_orig
#         })
#
#     df_matches = pd.DataFrame(matched_data)
#     # Save
#     csv_file = os.path.join(output_path, "matched_pinholes_aligned_space.csv")
#     df_matches.to_csv(csv_file, index=False)
#     print(f"Saved {len(df_matches)} matched pinholes (inliers) to {csv_file}")
#     return df_matches
def match_pinholes_in_aligned_space(
    green_pinhole_3d,
    red_pinhole_3d,
    matrix_gr,
    output_path
):
    """
    1) Warp red pinhole data into green coords using the inverted matrix (matrix_rg).
    2) Label pinholes in (a) green and (b) aligned-red images.
    3) Match with Hungarian + RANSAC.
    4) Visualize matched pinholes in aligned coords (for sanity check).
    5) Compute original red pinhole coords => also visualize.
    6) Return a DataFrame with matched pinholes in BOTH green coords and original red coords.
    """
    import matplotlib.pyplot as plt

    # 1) Project the 3D stacks
    proj_green = np.mean(green_pinhole_3d, axis=0)
    proj_red   = np.mean(red_pinhole_3d,   axis=0)

    # remove hot pixels or saturations
    remove_hot_pixel(proj_green,1000)
    remove_hot_pixel(proj_red,1000)

    # convert to uint8
    proj_green_8u = convert_to_uint8(proj_green)
    proj_red_8u   = convert_to_uint8(proj_red)

    # 2) Invert the green->red matrix so we can warp red->green
    matrix_rg = invert_affine_transform(matrix_gr)

    # Warp red image into green coords
    aligned_red_8u = apply_transform(proj_red_8u, proj_green_8u, matrix_rg)

    # 3) Threshold + label in aligned space
    green_bin = threshold_image(proj_green_8u, threshold='otsu')
    red_aligned_bin = threshold_image(aligned_red_8u, threshold='otsu')

    green_labels = label(green_bin, connectivity=2)
    red_aligned_labels = label(red_aligned_bin, connectivity=2)

    green_props = regionprops(green_labels)
    red_aligned_props = regionprops(red_aligned_labels)

    green_centroids = np.array([p.centroid for p in green_props])             # (N,2) => (y,x)
    red_aligned_centroids = np.array([p.centroid for p in red_aligned_props]) # (M,2)

    # 4) Hungarian matching in aligned space
    cost_mat = distance_matrix(red_aligned_centroids, green_centroids)
    row_ind, col_ind = linear_sum_assignment(cost_mat)

    matched_red_aligned = red_aligned_centroids[row_ind]
    matched_green       = green_centroids[col_ind]

    # 5) RANSAC in aligned space
    from skimage.transform import AffineTransform
    from skimage.measure import ransac

    matched_red_aligned_xy = matched_red_aligned[:, ::-1] # (x,y)
    matched_green_xy       = matched_green[:, ::-1]       # (x,y)

    model_robust, inliers = ransac(
        data=(matched_red_aligned_xy, matched_green_xy),
        model_class=AffineTransform,
        min_samples=3,
        residual_threshold=2.0,
        max_trials=5000
    )
    inlier_indices = np.where(inliers)[0]

    # Keep only inliers
    inlier_red_aligned = matched_red_aligned[inliers]  # (y,x in aligned space)
    inlier_green       = matched_green[inliers]        # (y,x in green space)

    # 6) Now map the aligned red pinhole coords back to the original red coords
    #    Using matrix_gr (green->red) because our aligned red coords are effectively
    #    in "green" coordinates.
    #    forward_transform_points expects (y,x) shape => returns (y,x).
    inlier_red_original = forward_transform_points(inlier_red_aligned, matrix_gr)

    # 7) Create a matched-pinholes DataFrame
    matched_data = []
    for i, idx in enumerate(inlier_indices):
        gy, gx = inlier_green[i]                     # green space
        ry_aligned, rx_aligned = inlier_red_aligned[i] # aligned/red in green coords
        ry_orig, rx_orig       = inlier_red_original[i] # original red coords

        matched_data.append({
            "GreenY":        gy,
            "GreenX":        gx,
            "AlignedRedY":   ry_aligned,
            "AlignedRedX":   rx_aligned,
            "OriginalRedY":  ry_orig,
            "OriginalRedX":  rx_orig
        })

    df_matches = pd.DataFrame(matched_data)
    csv_file = os.path.join(output_path, "matched_pinholes_aligned_space.csv")
    df_matches.to_csv(csv_file, index=False)
    print(f"Saved {len(df_matches)} matched pinholes (inliers) to {csv_file}")

    ###############################
    # Visualization of Matched Pinholes
    ###############################

    # A) show green vs. ALIGNED red in the same coordinate space
    fig1, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Show original green
    axes[0].imshow(proj_green_8u, cmap='gray')
    axes[0].scatter(inlier_green[:, 1], inlier_green[:, 0],
                    c='r', s=10, marker='o', label='Inlier Green')
    axes[0].set_title("Green pinholes (Green coords)")
    axes[0].invert_yaxis()  # if you prefer the origin at top-left

    # Show aligned red
    axes[1].imshow(aligned_red_8u, cmap='gray')
    axes[1].scatter(inlier_red_aligned[:, 1], inlier_red_aligned[:, 0],
                    c='g', s=10, marker='o', label='Inlier Red (aligned)')
    axes[1].set_title("Red pinholes warped -> Green coords")
    axes[1].invert_yaxis()

    for ax in axes:
        ax.legend(loc='upper right')

    out_fig1 = os.path.join(output_path, "matched_pinhole_aligned_space.png")
    plt.savefig(out_fig1, dpi=200)
    plt.show()

    # B) show original red image with the matched pinholes in their original red coords
    fig2, ax2 = plt.subplots(1, 1, figsize=(6,6))
    ax2.imshow(proj_red_8u, cmap='gray')
    ax2.scatter(inlier_red_original[:,1], inlier_red_original[:,0],
                c='y', s=10, marker='o', label='Inlier Red (original coords)')
    ax2.set_title("Inlier Red pinholes in Original Red coords")
    ax2.invert_yaxis()

    out_fig2 = os.path.join(output_path, "matched_pinhole_original_red_space.png")
    plt.savefig(out_fig2, dpi=200)
    plt.show()

    return df_matches


#########################
#    Analysis Functions
#########################

def analysis_image_with_matched_pinholes(
        green_stack,
        red_stack,
        matched_pinholes_df,
        output_path,
        experiment_name
):
    """
    - We have 'matched_pinholes_df' with columns:
        GreenY, GreenX, AlignedRedY, AlignedRedX, OriginalRedY, OriginalRedX
    - We want to measure intensities in the original green coords
      and the original red coords (not aligned).

    So we can do:
      (gY, gX) from the DataFrame  => measure in green_stack
      (ry, rx) from 'OriginalRedY', 'OriginalRedX' => measure in red_stack

    We'll create a small ROI around each pinhole to extract intensity stats.
    """
    num_frames, H, W = red_stack.shape
    half_window = 2
    results = []

    for t in range(num_frames):
        green_frame = green_stack[t]
        red_frame = red_stack[t]

        for _, row in matched_pinholes_df.iterrows():
            gY, gX = row["GreenY"], row["GreenX"]
            rY, rX = row["OriginalRedY"], row["OriginalRedX"]

            # Round to nearest int
            gY_i, gX_i = int(round(gY)), int(round(gX))
            rY_i, rX_i = int(round(rY)), int(round(rX))

            # Safe boundaries
            g_ymin, g_ymax = max(0, gY_i - half_window), min(H, gY_i + half_window + 1)
            g_xmin, g_xmax = max(0, gX_i - half_window), min(W, gX_i + half_window + 1)

            r_ymin, r_ymax = max(0, rY_i - half_window), min(H, rY_i + half_window + 1)
            r_xmin, r_xmax = max(0, rX_i - half_window), min(W, rX_i + half_window + 1)

            green_roi = green_frame[g_ymin:g_ymax, g_xmin:g_xmax]
            red_roi = red_frame[r_ymin:r_ymax, r_xmin:r_xmax]

            if green_roi.size == 0 or red_roi.size == 0:
                continue

            data_dict = {
                "Frame": t,
                "GreenY": gY,
                "GreenX": gX,
                "RedY": rY,
                "RedX": rX,
                "GreenMean": green_roi.mean(),
                "GreenMax": green_roi.max(),
                "GreenSum": green_roi.sum(),
                "RedMean": red_roi.mean(),
                "RedMax": red_roi.max(),
                "RedSum": red_roi.sum(),
            }
            results.append(data_dict)

    df = pd.DataFrame(results)
    out_csv = os.path.join(output_path, f"timelapse_intensity_{experiment_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[{experiment_name}] => Saved per-frame intensity results to: {out_csv}")


def process_experiment(
        experiment,
        working_path,
        red_var_name,
        green_var_name,
        result_folder,
        matched_pinholes_df
):
    """
    Loads the real data for 'experiment', then measures intensities in
    the green coords vs. the original red coords.
    """
    exp_path = os.path.join(working_path, experiment)
    if not os.path.isdir(exp_path):
        print(f"Skipping '{experiment}' => Not a folder.")
        return

    red_stack = generate_df(exp_path, red_var_name)
    green_stack = generate_df(exp_path, green_var_name)
    if red_stack.size == 0 or green_stack.size == 0:
        print(f"[{experiment}] => No data found for Red/Green.")
        return

    # Convert to uint8 if needed
    red_stack_8u = convert_to_uint8(red_stack)
    green_stack_8u = convert_to_uint8(green_stack)

    # Save a reference TIF
    save_image(os.path.join(result_folder, f"{experiment}_red.tif"), red_stack_8u, Type=0)
    save_image(os.path.join(result_folder, f"{experiment}_green.tif"), green_stack_8u, Type=0)

    # Run analysis
    analysis_image_with_matched_pinholes(
        green_stack_8u,
        red_stack_8u,
        matched_pinholes_df,
        result_folder,
        experiment
    )


#############
# main script
#############

def main():
    # Example user-defined paths
    working_path = Path("/Volumes/T7/20250619")
    to_align_path = "/Volumes/T7/20250619/AlignChannel"
    pinhole_path = "/Volumes/T7/20250619/PinHole"

    # Make result folder
    result_folder_name = working_path.name + "_results"
    result_folder = os.path.join(os.path.dirname(working_path), result_folder_name)
    os.makedirs(result_folder, exist_ok=True)

    # Variables
    # For the "dual-fluorescence" alignment sample:
    red_align_var = "pixxx_reduction_red_final"
    green_align_var = "pixxx_reduction_green_final"

    # For the Pinhole data:
    red_pinhole_var = "pixxx_reduction_new_final"
    green_pinhole_var = "pixxx_reduction_old_final"

    # 1) Load alignment data => compute matrix (green->red)
    red_align_stack = generate_df(to_align_path, red_align_var)
    green_align_stack = generate_df(to_align_path, green_align_var)

    # Project them
    proj_red = np.mean(red_align_stack, axis=0)
    proj_green = np.mean(green_align_stack, axis=0)

    # Remove hot pixel, convert to 8-bit
    remove_hot_pixel(proj_red)
    remove_hot_pixel(proj_green)
    proj_red_8u = convert_to_uint8(proj_red)
    proj_green_8u = convert_to_uint8(proj_green)

    # Compute transform: green->red
    matrix_gr = calculate_transform_green_to_red(proj_green_8u, proj_red_8u)
    print("matrix_gr (green->red) =\n", matrix_gr)

    # 2) Load pinhole data, match in aligned (green) space
    red_pinhole_3d = generate_df(pinhole_path, red_pinhole_var)
    green_pinhole_3d = generate_df(pinhole_path, green_pinhole_var)

    # Save reference
    save_image(os.path.join(result_folder, "red_pinhole_stack.tif"), red_pinhole_3d, Type=0)
    save_image(os.path.join(result_folder, "green_pinhole_stack.tif"), green_pinhole_3d, Type=0)

    # Match pinholes
    matched_pinholes_df = match_pinholes_in_aligned_space(
        green_pinhole_3d,
        red_pinhole_3d,
        matrix_gr,
        result_folder
    )
    # This returns columns: ["GreenY", "GreenX", "AlignedRedY", "AlignedRedX", "OriginalRedY", "OriginalRedX"]

    # 3) Use matched pinholes for each experiment
    experiments = os.listdir(working_path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for experiment in experiments:
            if experiment.startswith("."):
                continue
            if not os.path.isdir(os.path.join(working_path, experiment)):
                continue

            futures.append(
                executor.submit(
                    process_experiment,
                    experiment,
                    str(working_path),
                    red_pinhole_var,
                    green_pinhole_var,
                    result_folder,
                    matched_pinholes_df
                )
            )

        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print("[ERROR in experiment loop]", e)

    print("All experiments processed successfully.")


if __name__ == '__main__':
    main()
