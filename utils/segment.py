import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time # Idea: try different configurations for each method, and for each configuration compute the time spent to obtain the result

from sklearn.cluster import KMeans
from skimage.filters import threshold_multiotsu
from sklearn.mixture import GaussianMixture
from utils.display import getSlice

# ----------------------------------------------------------------- #
# This function plots a slice of choice, segmented with K-Means     #
# with some configurations decided by the user. Then at the end     #
# a table containing all the usefull information is displayed.      #
# At the end, also the segmented slices are returned so that the    #
# user can use them for comparisons.                                #
# ----------------------------------------------------------------- #
def kMeansSegmentation(t1_data: np.ndarray, brain_data: np.ndarray, bool_mask: np.ndarray, 
                       configs: list, slice_idx: int = None, axis:int = 2) -> np.ndarray:

    # If slice is not specified, I take the slice in the middle for higher probability to have many details
    if slice_idx is None:
        slice_idx = t1_data.shape[axis] // 2

    results = [] # it will be a list of dictionaries, each one containing the results of a specific configuration
    n_configs = len(configs)

    fig, axes = plt.subplots(2, n_configs, figsize=(5 * n_configs, 10), squeeze=False)    
    
    slices = []

    for i, config in enumerate(configs):
        # Start the timer
        start_time = time.time()

        # Initialize KMeans object and fit the data
        kmeans = KMeans(**config)
        kmeans.fit(brain_data)
        
        # Stop the timer and compute the elapsed_time
        elapsed_time = time.time() - start_time
        
        t1_data_segmented = np.zeros_like(t1_data)
        t1_data_segmented[bool_mask] = kmeans.labels_ + 1 # attribute that contains the assigned labels of pixels

        # Ensure the equality in the shapes
        assert len(t1_data_segmented[bool_mask]) == len(kmeans.labels_), "Mismatch in the number of segmented values"
        
        # Count pixels for each label
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        pixels_per_label = dict(zip(unique, counts))
        
        # Save all the results in a dicitonary and append it to results
        results.append({
            'config_id': i + 1,
            'n_clusters': config.get('n_clusters', 8),
            'init': config.get('init', 'k-means++'),
            'algorithm': config.get('algorithm', 'lloyd'),
            'time_sec': round(elapsed_time, 3),
            'n_iter': kmeans.n_iter_,
            'inertia': round(kmeans.inertia_, 2),
            'n_features': kmeans.n_features_in_,
            'pixels_per_label': pixels_per_label
        })
        
        if axis == 0:
            slice_orig = t1_data[slice_idx, :, :]
            slice_img = t1_data_segmented[slice_idx, :, :]
        elif axis == 1:
            slice_orig = t1_data[:, slice_idx, :]
            slice_img = t1_data_segmented[:, slice_idx, :]
        else:
            slice_orig = t1_data[:, :, slice_idx]
            slice_img = t1_data_segmented[:, :, slice_idx]

        if i == 0:
            slices.append(slice_orig)
        slices.append(slice_img)
            
        ax_orig = axes[0, i]
        ax_orig.imshow(slice_orig, cmap='gray')
        ax_orig.set_title(f"Originale - Slice {slice_idx}\n(Config {i+1})")
        ax_orig.axis('off')

        ax_seg = axes[1, i]
        ax_seg.imshow(slice_img, cmap='gray')
        ax_seg.set_title(f"Segmentata - Config {i+1}\nIter: {kmeans.n_iter_} | Time: {elapsed_time:.2f}s")
        ax_seg.axis('off')
        
    plt.tight_layout()
    plt.show()

    df_summary = pd.DataFrame(results)
    return df_summary, slices

# ----------------------------------------------------------------- #
# This function plots a slice of choice, segmented with Otsu        #
# with some configurations decided by the user. Then at the end     #
# a table containing all the usefull information is displayed.      #
# At the end, also the segmented slices are returned so that the    #
# user can use them for comparisons.                                #
# ----------------------------------------------------------------- #
def otsuSegmentation(t1_data: np.ndarray, brain_data: np.ndarray, bool_mask: np.ndarray, 
                               configs: list, slice_idx: int = None, axis: int = 2):
    
    if slice_idx is None:
        slice_idx = t1_data.shape[axis] // 2
        
    risultati = []
    n_configs = len(configs)
    
    fig, axes = plt.subplots(2, n_configs, figsize=(5 * n_configs, 10), squeeze=False)
    slices = []
        
    for i, config in enumerate(configs):
        n_classes = config.get('classes', 3) # get the number of classe, if not set, use 3 as default
        
        start_time = time.time()
        thresholds = threshold_multiotsu(brain_data, classes=n_classes)
        labels = np.digitize(brain_data, bins=thresholds)
        elapsed_time = time.time() - start_time
        
        t1_data_segmented = np.zeros_like(t1_data)
        t1_data_segmented[bool_mask] = labels + 1
        
        assert len(t1_data_segmented[bool_mask]) == len(labels), "Qualcosa non torna con le dimensioni"
        
        unique, counts = np.unique(labels, return_counts=True)
        pixels_per_label = dict(zip(unique, counts))
        
        risultati.append({
            'config_id': i + 1,
            'classes': n_classes,
            'time_sec': round(elapsed_time, 3),
            'thresholds': str(np.round(thresholds, 2).tolist()),
            'pixels_per_label': pixels_per_label
        })
        
        if axis == 0:
            slice_orig, slice_seg = t1_data[slice_idx, :, :], t1_data_segmented[slice_idx, :, :]
        elif axis == 1:
            slice_orig, slice_seg = t1_data[:, slice_idx, :], t1_data_segmented[:, slice_idx, :]
        else:
            slice_orig, slice_seg = t1_data[:, :, slice_idx], t1_data_segmented[:, :, slice_idx]

        if i == 0:
            slices.append(slice_orig)

        slices.append(slice_seg)

        axes[0, i].imshow(slice_orig, cmap='gray')
        axes[0, i].set_title(f"Originale - Slice {slice_idx}\n(MultiOtsu {n_classes} classi)")
        axes[0, i].axis('off')
        axes[1, i].imshow(slice_seg, cmap='gray')
        axes[1, i].set_title(f"Segmentata - Config {i+1}\nTime: {elapsed_time:.2f}s")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(risultati), slices

# ----------------------------------------------------------------- #
# This function plots a slice of choice, segmented with Gaussian    #
# Mixture Models (GMM) with configurations specified by the user.   #
# Useful convergence metrics (AIC, BIC, Log-Likelihood) are tracked #
# and the segmented slices are returned for comparisons.            #
# ----------------------------------------------------------------- #
def gmmSegmentation(t1_data: np.ndarray, brain_data: np.ndarray, bool_mask: np.ndarray,
                    configs: list, slice_idx: int = None, axis: int = 2):

    # Pick middle slice if not specified
    if slice_idx is None:
        slice_idx = t1_data.shape[axis] // 2

    # GMM requires 2D input (n_samples, n_features)
    if brain_data.ndim == 1:
        brain_features = brain_data.reshape(-1, 1)
    else:
        brain_features = brain_data

    results = []
    n_configs = len(configs)

    fig, axes = plt.subplots(2, n_configs, figsize=(5 * n_configs, 10), squeeze=False)
    slices = []

    for i, config in enumerate(configs):
        start_time = time.time()

        # Initialize GMM and fit to the brain voxels
        gmm = GaussianMixture(**config)
        labels = gmm.fit_predict(brain_features)

        elapsed_time = time.time() - start_time

        t1_data_segmented = np.zeros_like(t1_data)
        t1_data_segmented[bool_mask] = labels + 1

        # Ensure dimensional consistency
        assert len(t1_data_segmented[bool_mask]) == len(
            labels
        ), "Mismatch in the number of segmented values"

        # Count pixels per tissue class
        unique, counts = np.unique(labels, return_counts=True)
        pixels_per_label = dict(zip(unique, counts))

        # Store parameters and GMM evaluation metrics
        results.append({
            "config_id": i + 1,
            "n_components": config.get("n_components", 3),
            "covariance_type": config.get("covariance_type", "full"),
            "time_sec": round(elapsed_time, 3),
            "n_iter": gmm.n_iter_,
            "converged": gmm.converged_,
            "bic": round(gmm.bic(brain_features), 2),
            "aic": round(gmm.aic(brain_features), 2),
            "pixels_per_label": pixels_per_label,
        })

        if axis == 0:
            slice_orig, slice_seg = (
                t1_data[slice_idx, :, :],
                t1_data_segmented[slice_idx, :, :],
            )
        elif axis == 1:
            slice_orig, slice_seg = (
                t1_data[:, slice_idx, :],
                t1_data_segmented[:, slice_idx, :],
            )
        else:
            slice_orig, slice_seg = (
                t1_data[:, :, slice_idx],
                t1_data_segmented[:, :, slice_idx],
            )

        if i == 0:
            slices.append(slice_orig)

        slices.append(slice_seg)

        axes[0, i].imshow(slice_orig, cmap="gray")
        axes[0, i].set_title(
            f"Originale - Slice {slice_idx}\n(GMM"
            f" {config.get('n_components', 3)} comp)"
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(slice_seg, cmap="gray")
        axes[1, i].set_title(
            f"Segmentata - Config {i+1}\nIter: {gmm.n_iter_} | Time:"
            f" {elapsed_time:.2f}s"
        )
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results), slices

# ------------------------------------------------------------------- #
# Comparison between slices obtained with three different             #
# segmentation methods (K-Means, Multi-Otsu, GMM), visually and       #
# statistically.                                                      #
# ------------------------------------------------------------------- #
def compareSegmentationMethods(slice_orig, slice_kmeans, slice_otsu, slice_gmm,
                               stats_kmeans, stats_otsu, stats_gmm):

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Original Plot
    axes[0].imshow(slice_orig, cmap="gray")
    axes[0].set_title("Immagine Originale (T1w)")
    axes[0].axis("off")

    # 2. K-Means Plot
    axes[1].imshow(slice_kmeans, cmap="gray")
    axes[1].set_title(
        f"K-Means ({stats_kmeans.get('n_clusters', 'N')} Cluster)\nTempo:"
        f" {stats_kmeans.get('time_sec', 0)}s"
    )
    axes[1].axis("off")

    # 3. Multi-Otsu Plot
    axes[2].imshow(slice_otsu, cmap="gray")
    axes[2].set_title(
        f"Multi-Otsu ({stats_otsu.get('classes', 'N')} Classi)\nTempo:"
        f" {stats_otsu.get('time_sec', 0)}s"
    )
    axes[2].axis("off")

    # 4. GMM Plot
    axes[3].imshow(slice_gmm, cmap="gray")
    axes[3].set_title(
        f"GMM ({stats_gmm.get('n_components', 'N')} Componenti)\nTempo:"
        f" {stats_gmm.get('time_sec', 0)}s"
    )
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

    # Collect stats in a unified comparison DataFrame
    df_compare = pd.DataFrame(
        [stats_kmeans, stats_otsu, stats_gmm],
        index=["K-Means", "Multi-Otsu", "GMM"],
    )
    return df_compare





