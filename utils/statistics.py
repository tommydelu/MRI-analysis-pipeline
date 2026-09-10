import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from utils.display import getSlice


# ----------------------------------------------------------------- #
# Given a single slice this function plots the histogram of the t1  #
# weighted image against the same slice but masked. Also other      #
# characteristics are displayed such as the maximum and minimum     #
# value, and the mean value along the chosen axis                   #
# ----------------------------------------------------------------- #
def sliceHistAndStatistics(nii_data: np.ndarray, mask_data: np.ndarray, 
                           slice_idx: int, axis: str = "axial") -> None:

    """
    Input:
    - nii_data: expected 3D array of t1 data
    - mask_data: expected a 3D mask data associated to t1
    - slice_idx: slice's index to print statistics and plot the histogram
    - axis: point of view

    Output:
    - None
    """

    slice_data, _ = getSlice(nii_data, slice_idx, axis=axis)
    mask_slice, _ = getSlice(mask_data, slice_idx, axis=axis)

    mask_bool = mask_slice.astype(bool)

    if not np.any(mask_bool):
        print(f"Attention: slice number {slice_idx} does not contain any voxel inside the mask.")
        return

    masked_voxels = slice_data[mask_bool]

    print(f"Max:  {np.max(masked_voxels):.2f}")
    print(f"Min:  {np.min(masked_voxels):.2f}")
    print(f"Mean: {np.mean(masked_voxels):.2f}")
    print(f"Std:  {np.std(masked_voxels):.2f}")

    all_voxels = slice_data.ravel()
    hist_range = (float(np.min(all_voxels)), float(np.max(all_voxels)))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    axs[0].hist(all_voxels, bins=100, range=hist_range, density=True, color='gray', alpha=0.7)
    axs[0].set_title(f'Slice {slice_idx} — No mask')
    axs[0].set_xlabel('Intensity')
    axs[0].set_ylabel('Density')

    axs[1].hist(masked_voxels, bins=100, range=hist_range, density=True, color='crimson', alpha=0.7)
    axs[1].set_title(f'Slice {slice_idx} — Masked')
    axs[1].set_xlabel('Intensity')

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------- #
# 
# ----------------------------------------------------------------- #
def volumeHistAndStatistics(nii_data: np.ndarray, mask_data: np.ndarray | None = None, 
                            bins = 100) -> None:
    """
    Input:
    - nii_data: 3D volume (or 4D, in this case a time instant must be chosen)
    - mask_data: if provided, the histogram is computed only on the masked voxels (default is None)
    - bins: plot parameters

    Output:
    - None
    """

    assert len(nii_data.shape) == 3, "volume_hist expects a 3D volume. If 4D, please select a time instant."
    
    data_flat = nii_data.ravel() # use .ravel to avoid copy in memories!
    hist_range = (float(np.min(data_flat)), float(np.max(data_flat)))

    # 1) No provided mask
    if mask_data is None:
        print(f"Max:  {np.max(data_flat):.2f}")
        print(f"Min:  {np.min(data_flat):.2f}")
        print(f"Mean: {np.mean(data_flat):.2f}")
        print(f"Std:  {np.std(data_flat):.2f}")

        plt.figure(figsize=(6, 4))
        plt.hist(data_flat, bins=bins, range=hist_range, density=True, color='gray', alpha=0.7)
        plt.title('Volume histogram — no mask')
        plt.xlabel('Intensity')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.show()
        return

    # 2:) Mask provided -> verifiche di validità
    assert mask_data.shape == nii_data.shape, (
        f"Shape mismatch: nii_data {nii_data.shape} vs mask_data {mask_data.shape}"
    )

    mask_bool = mask_data.astype(bool)
    if not np.any(mask_bool):
        print("Attention: the 3D volume does not contain any voxel inside the mask.")
        return
    
    masked_voxels = nii_data[mask_bool]

    print(f"Max:  {np.max(masked_voxels):.2f}")
    print(f"Min:  {np.min(masked_voxels):.2f}")
    print(f"Mean: {np.mean(masked_voxels):.2f}")
    print(f"Std:  {np.std(masked_voxels):.2f}")
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    axs[0].hist(data_flat, bins=bins, range=hist_range, density=True, color='gray', alpha=0.7)
    axs[0].set_title('Volume histogram — No mask')
    axs[0].set_xlabel('Intensity')
    axs[0].set_ylabel('Density')

    axs[1].hist(masked_voxels, bins=bins, range=hist_range, density=True, color='crimson', alpha=0.7)
    axs[1].set_title('Volume histogram — Masked')
    axs[1].set_xlabel('Intensity')

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------- #
# 
# ----------------------------------------------------------------- #
def avg_intensity_along_axis(nii_data, mask_data=None, axis="axial"):
    """
    Input:
    - nii_data: volume 3D
    - mask_data: se fornita, la media di ogni slice è calcolata solo sui voxel dentro la maschera (default None)
    - axis: piano lungo cui scorrere (default "axial")

    Return: 1D numpy array con l'intensità media per ogni slice lungo l'asse scelto;
            mostra anche il plot del profilo
    """
    axis_to_shape_idx = {"axial": 2, "sagittal": 1, "coronal": 0}
    if axis not in axis_to_shape_idx:
        raise ValueError('axis must be "axial", "sagittal" or "coronal"')
    
    n_slices = nii_data.shape[axis_to_shape_idx[axis]]
    means = np.zeros(n_slices)

    for i in range(n_slices):
        slice_2d, _ = _get_slice(nii_data, i, axis=axis)
        
        if mask_data is None:
            means[i] = np.mean(slice_2d)
        else:
            mask_slice, _ = _get_slice(mask_data, i, axis=axis)
            mask_bool = mask_slice.astype(bool)
            means[i] = np.mean(slice_2d[mask_bool]) if mask_bool.sum() > 0 else np.nan

    plt.figure(figsize=(8, 4))
    plt.plot(means)
    masked_str = "masked" if mask_data is not None else "unmasked"
    plt.title(f'Average intensity along {axis} axis ({masked_str})')
    plt.xlabel('Slice index')
    plt.ylabel('Mean intensity')
    plt.show()

    return means

# ----------------------------------------------------------------- #
# 
# ----------------------------------------------------------------- #
def avg_signal_along_time(data_4d, TR=None):
    """
    Input:
    - data_4d: volume fMRI 4D (x, y, z, t)
    - TR: repetition time in secondi, se fornito l'asse x è in secondi, altrimenti in indice di volume (default None)

    Return: 1D numpy array con la media globale del segnale per ogni istante temporale;
            mostra anche il plot del segnale nel tempo
    """
    n_timepoints = data_4d.shape[3]
    means = np.zeros(n_timepoints)

    for t in range(n_timepoints):
        means[t] = np.mean(data_4d[:, :, :, t])

    if TR is not None:
        time_pts = np.arange(n_timepoints) * TR
        xlabel = 'Time (s)'
    else:
        time_pts = np.arange(n_timepoints)
        xlabel = 'Time point index'

    plt.figure(figsize=(10, 6))
    plt.plot(time_pts, means)
    plt.xlabel(xlabel)
    plt.ylabel('Mean global signal')
    plt.title('Average signal over time')
    plt.show()

# ----------------------------------------------------------------- #
# 
# ----------------------------------------------------------------- #
def avgFmriTemporalVolume(fmri_data):

    avg_volume = np.mean(fmri_data,axis=3)
    return avg_volume











    


    

    
   


















