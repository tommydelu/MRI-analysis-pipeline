import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from utils.statistics import _get_slice


def compare_methods(t1_data, segmented1, segmented2, slice_idx, axis="axial"):

    original_slice, _ = _get_slice(t1_data, slice_idx=slice_idx, axis=axis)
    slice1, _ = _get_slice(segmented1, slice_idx=slice_idx, axis=axis)
    slice2, _ = _get_slice(segmented2, slice_idx=slice_idx, axis=axis)

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




