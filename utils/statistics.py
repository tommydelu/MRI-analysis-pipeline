import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


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


def _get_slice(nii_data, slice_idx, axis="axial", time_instant=0):
    """
    Auxiliary function: extracts a 2D slice from 3D or 4D nii_data,
    given the axis and slice index. Handles bounds checking via assert.

    Return: 2D slice, and the size of the volume along the chosen axis
            (useful for titles / labels).
    """
    axis_to_shape_idx = {"axial": 2, "sagittal": 1, "coronal": 0}
    
    if axis not in axis_to_shape_idx:
        raise ValueError('axis must be "axial", "sagittal" or "coronal"')
    
    shape_idx = axis_to_shape_idx[axis]
    n_slices = nii_data.shape[shape_idx]
    
    is_4d = len(nii_data.shape) > 3
    if is_4d:
        assert slice_idx < n_slices and time_instant < nii_data.shape[3], \
            "You need to choose a slice index < {} or a time instant < {}.".format(n_slices, nii_data.shape[3])
    else:
        assert slice_idx < n_slices, \
            "You need to choose a slice index < {}.".format(n_slices)
    
    if axis == "axial":
        slice_2d = nii_data[:, :, slice_idx, time_instant] if is_4d else nii_data[:, :, slice_idx]
    elif axis == "sagittal":
        slice_2d = nii_data[:, slice_idx, :, time_instant] if is_4d else nii_data[:, slice_idx, :]
    else:  # coronal
        slice_2d = nii_data[slice_idx, :, :, time_instant] if is_4d else nii_data[slice_idx, :, :]
    
    return slice_2d, n_slices


def single_slice_hist(nii_data, mask_data, slice_idx, axis = "axial"):

    slice_2d, _ = _get_slice(nii_data, slice_idx, axis=axis)
    slice_1d = slice_2d.flatten()

    mask_slice, _ = _get_slice(mask_data, slice_idx, axis=axis)
    mask_slice = mask_slice.astype(bool)
    mask_slice_flattened = mask_slice.flatten()

    if mask_slice_flattened.sum() == 0:
        print(f"Attenzione: lo slice {slice_idx} non contiene voxel dentro la maschera.")
        return

    fig, axs = plt.subplots(1,2,figsize=(10,6))
    axs[0].hist(slice_1d, density=True, bins=255, range=(0,255))
    axs[0].set_title('Slice {} — no mask'.format(slice_idx))

    axs[1].hist(slice_1d[mask_slice_flattened], density=True, bins=255, range=(0,255))
    axs[1].set_title('Slice {} — masked'.format(slice_idx))

    plt.show()


def volume_hist(nii_data, mask_data=None, bins=255, hist_range=(0, 255)):
    """
    Input:
    - nii_data: volume 3D (o 4D, in tal caso serve specificare un singolo volume prima di chiamare la funzione)
    - mask_data: se fornita, l'istogramma viene calcolato solo sui voxel dentro la maschera (default None)
    - bins, hist_range: parametri passati a plt.hist

    Return: void, mostra l'istogramma del volume intero (masked vs unmasked se mask_data è fornita)
    """
    assert len(nii_data.shape) == 3, "volume_hist si aspetta un volume 3D. Se hai un 4D, seleziona prima un time point."
    
    data_flat = nii_data.flatten()

    if mask_data is None:
        plt.figure(figsize=(6, 4))
        plt.hist(data_flat, density=True, bins=bins, range=hist_range)
        plt.title('Volume histogram — no mask')
        plt.show()
        return

    assert nii_data.shape == mask_data.shape, "nii_data e mask_data devono avere la stessa shape."
    
    mask_flat = mask_data.astype(bool).flatten()
    
    if mask_flat.sum() == 0:
        print("Attenzione: la maschera fornita non contiene voxel True.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].hist(data_flat, density=True, bins=bins, range=hist_range)
    axs[0].set_title('Volume histogram — no mask')

    axs[1].hist(data_flat[mask_flat], density=True, bins=bins, range=hist_range)
    axs[1].set_title('Volume histogram — masked')
    
    plt.show()


def avg_intensity(nii_data, mask_data=None, single_slice=True, slice_idx=None, axis="axial"):
    """
    Input:
    - nii_data: volume 3D
    - mask_data: se fornita, la media è calcolata solo sui voxel dentro la maschera (default None)
    - single_slice: se True calcola la media su un singolo slice, se False su tutto il volume (default True)
    - slice_idx: indice dello slice, richiesto se single_slice=True
    - axis: piano di riferimento se single_slice=True (default "axial")

    Return: float, intensità media
    """
    if single_slice:
        assert slice_idx is not None, "slice_idx è richiesto quando single_slice=True."
        data, _ = _get_slice(nii_data, slice_idx, axis=axis)
        if mask_data is not None:
            mask, _ = _get_slice(mask_data, slice_idx, axis=axis)
    else:
        data = nii_data
        mask = mask_data

    if mask_data is None:
        mean_val = np.mean(data)
    else:
        mask_bool = mask.astype(bool)
        if mask_bool.sum() == 0:
            print("Attenzione: la maschera non contiene voxel True in questa regione.")
            return None
        mean_val = np.mean(data[mask_bool])

    scope = f"slice {slice_idx} ({axis})" if single_slice else "intero volume"
    masked_str = "masked" if mask_data is not None else "unmasked"
    print(f"Intensità media ({masked_str}) — {scope}: {mean_val:.3f}")
    
    return mean_val


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