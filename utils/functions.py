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


def load_and_display(nii_path: str, name: str, print_info = True):
    """
    Input:
    - nii_path: path to the .nii file
    - name: descriptive name for the display of info

    Return: dict data, header data, img data and dimension of the img
    """

    nii_img = nib.load(nii_path)
    nii_hdr = nii_img.header
    nii_data = nii_img.get_fdata()
    nii_dim = nii_hdr['dim'][:4]

    voxel_size = tuple(round(float(v),3) for v in nii_hdr.get_zooms())
    
    if print_info:
        print(f"\nINFORMATION ABOUT {name} NIFTI FILE:")
        print("\n"+"-"*100)
        
        print(f"Type of variable nii_img: {type(nii_img)}")
        print(f"Type of variable nii_hdr: {type(nii_hdr)}")
        print(f"Shape of the NIfTI image: {nii_img.shape}")
        print(f"Dimensions of the NIfTI image: {nii_dim}")
        print(f"Dimensions of NIfTI img data: {nii_data.shape}")
        print(f"Voxel size (mm): {voxel_size}")
    
    return nii_img, nii_hdr, nii_data, nii_dim


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


def display_single_slice(nii_data, slice_idx, axis="axial", time_instant=0):
    """
    Input: 
    - img data of some .nii file
    - slice_idx: index for a slice
    - axis: reference plane, choices are "axial" (default), "sagittal" and "coronal"
    - time_instant: to choose a time instant when nii_data.shape is > 3 (fMRI for example), default is 0

    Return: void, it visualizes the chosen slice
    """
    slice_2d, n_slices = _get_slice(nii_data, slice_idx, axis, time_instant)
    
    plt.figure(figsize=(6, 4))
    plt.imshow(np.rot90(slice_2d, 1), cmap='gray', vmin=np.min(slice_2d))
    plt.grid(False)
    plt.title('Single slice, n. {} of {}'.format(slice_idx, n_slices))
    plt.axis('off')
    plt.show()


def display_group_of_slices(nii_data, num_to_disp, starting_idx=0, sparse=False,
                              idxs_list=None, disp_step=1, axis="axial", time_instant=0,
                              max_cols=4):
    """
    Input: 
    - nii_data: data structure to display
    - num_to_disp: number of slices to display
    - starting_idx: first idx to display (used when sparse=False)
    - sparse: boolean, if True you can provide a list of idxs to display, if False starting idx and step are used (default = False)
    - idxs_list: list of indexes to display in case of sparse = True (default = None)
    - disp_step: step between one displayed slice and the next, used when sparse=False (default = 1)
    - axis: reference plane, "axial" (default), "sagittal" or "coronal"
    - time_instant: time point to use if nii_data is 4D (default = 0)
    - max_cols: maximum number of images per row (default = 4)

    Return: void, displays a grid of slices
    """
    axis_to_shape_idx = {"axial": 2, "sagittal": 1, "coronal": 0}
    if axis not in axis_to_shape_idx:
        raise ValueError('axis must be "axial", "sagittal" or "coronal"')
    n_slices_total = nii_data.shape[axis_to_shape_idx[axis]]

    # --- build the list of indices to display ---
    if sparse:
        if idxs_list is None:
            print("Nessuna lista fornita: scelgo io {} indici casuali.".format(num_to_disp))
            idxs_list = sorted(np.random.choice(n_slices_total, size=num_to_disp, replace=False))
        else:
            assert max(idxs_list) < n_slices_total, \
                "All indices in idxs_list must be < {}.".format(n_slices_total)
            num_to_disp = len(idxs_list)
        indices = idxs_list
    else:
        indices = [starting_idx + i * disp_step for i in range(num_to_disp)]
        assert max(indices) < n_slices_total, \
            "starting_idx + num_to_disp * disp_step exceeds the number of slices ({}).".format(n_slices_total)

    # --- build the grid ---
    n_cols = min(max_cols, num_to_disp)
    n_rows = int(np.ceil(num_to_disp / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()  # flatten anche se n_rows=1 o n_cols=1

    for ax, idx in zip(axes, indices):
        slice_2d, n_slices = _get_slice(nii_data, idx, axis, time_instant)
        ax.imshow(np.rot90(slice_2d, 1), cmap='gray', vmin=np.min(slice_2d))
        ax.set_title('Slice {} of {}'.format(idx, n_slices))
        ax.axis('off')

    # nascondi eventuali subplot vuoti in eccesso (griglia non piena)
    for ax in axes[len(indices):]:
        ax.axis('off')

    plt.suptitle('{} view — group of {} slices'.format(axis.capitalize(), num_to_disp))
    plt.tight_layout()
    plt.show()


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