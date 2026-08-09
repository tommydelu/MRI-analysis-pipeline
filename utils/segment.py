import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from skimage.filters import threshold_multiotsu

from utils.display import _get_slice



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
