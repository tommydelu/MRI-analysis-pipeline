import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from skimage.filters import threshold_multiotsu

from utils.display import getSlice



def kmean_segmentation(t1_data, brain_data, mask, n_cluster, random_state=0, init='auto'):

    kmeans = KMeans(n_clusters=n_cluster, random_state=random_state, n_init=init)
    brain_data = brain_data.reshape(brain_data.shape[0],-1)

    kmeans.fit(brain_data)

    segmented = np.zeros_like(t1_data)
    segmented[mask] = kmeans.labels_
    assert len(segmented[mask]) == len(kmeans.labels_), "Il numero di valori non coincide"

    return segmented


def otsu_segmentation(brain_data):

    thresh = threshold_multiotsu(brain_data)
    return thresh


# ------------------------------------------------------------------- #
# Given a NiFti data structure, display a group of slices.            #
# You can give a predefined list of indexes or sample them randomnly, #
# or set a starting idx with a step                                   #
# ------------------------------------------------------------------- #
def compare_methods(t1_data, segmented1, segmented2, slice_idx, axis="axial"):

    original_slice, _ = getSlice(t1_data, slice_idx=slice_idx, axis=axis)
    slice1, _ = getSlice(segmented1, slice_idx=slice_idx, axis=axis)
    slice2, _ = getSlice(segmented2, slice_idx=slice_idx, axis=axis)

    fig, axs = plt.subplots(1,3,figsize=(10,6))
    axs[0].imshow(original_slice, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title("Original Image")
    axs[1].imshow(slice1, cmap='gray')
    axs[1].set_axis_off()
    axs[1].set_title("KMeans result")
    axs[2].imshow(slice2, cmap='gray')
    axs[2].set_axis_off()
    axs[2].set_title("Otsu Result")


