import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# ----------------------------------------------------------------- #
# Load a file in the .nii format and display a summary of its       #
# characteristics                                                   #
# ----------------------------------------------------------------- #
def loadAndDisplay(nii_path: str, name: str, print_summary = True) -> dict:

    """
    Input:
    - nii_path: path that leads to the .nii file of interest
    - name: name assigned in the summary
    - print_summary: set to True to display the summary

    Output:
    - nii_dict: dictionary containing the img, heder, data, and dims of the NiFti file
    """
    nii_dict = {"img": None, "hdr": None, "data": None, "dims": None}

    nii_dict['img'] = nib.load(nii_path)
    nii_dict['hdr'] = nii_dict['img'].header
    nii_dict['data'] = nii_dict['img'].get_fdata()
    nii_dict['dims'] = nii_dict['hdr']['dim'][:4]

    voxel_size = tuple(round(float(v),3) for v in nii_dict['hdr'].get_zooms())

    if print_summary:
        print(f"\nINFORMATION ABOUT {name} NIFTI FILE:")
        print("\n"+"-"*100)
        
        print(f"Type of variable nii_img: {type(nii_dict['img'])}")
        print(f"Type of variable nii_hdr: {type(nii_dict['hdr'])}")
        print(f"Shape of the NIfTI image: {nii_dict['img'].shape}")
        print(f"Dimensions of the NIfTI image: {nii_dict['dims']}")
        print(f"Dimensions of NIfTI img data: {nii_dict['data'].shape}")
        print(f"Voxel size (mm): {voxel_size}")
    
    return nii_dict

# ----------------------------------------------------------------- #
# Display only one slice of the structure that is given in input.   #
# You can decide to display an axial slice, coronal, or sagittal    #           
# ----------------------------------------------------------------- #
def displaySingleSlice(nii_data: np.memmap, slice_idx: int, 
                       axis: str = "axial", time_instant: int = 0, 
                       save_figure: bool = False, figure_name: str = '') -> None:
    """
    Input: 
    - nii_data: data structure of a NifTi file
    - slice_idx: index for a slice to display
    - axis: reference plane, choices are "axial" (default), "sagittal" and "coronal"
    - time_instant: to choose a time instant when nii_data.shape is > 3 (fMRI for example), default is 0.
    - save_figure: if True, save the plot in the figures directory
    - figure_name: give a name to the saved img

    Output:
    - None
    """
    # 1) Get the slice, for this purpose I use an auxiliary function.
    slice, n_slices = getSlice(nii_data, slice_idx, axis, time_instant)

    # 2) Plot the extracted slice.
    plt.figure(figsize=(6, 4))
    plt.imshow(np.rot90(slice, 1), cmap='gray', vmin=np.min(slice)) # Rotate the slice for a better visualization
    plt.grid(False)
    plt.title(f'Slice n. {slice_idx} of {n_slices}')
    plt.axis('off')

    if save_figure and len(nii_data.shape) == 3:
        plt.savefig(f'figures/{figure_name}_single_slice_{axis}_idx_{slice_idx}.png')
    elif save_figure and len(nii_data.shape) == 4:
        plt.savefig(f'figures/{figure_name}_single_slice_{axis}_idx_{slice_idx}_time_{time_instant}.png')

    plt.show()

# ----------------------------------------------------------------- #
# Extract a 2D slice from the data structure given in input.        #
# All the boundaries are checked to ensure a correct extraction     #
# ----------------------------------------------------------------- #
def getSlice(nii_data: np.memmap, slice_idx: int, axis: str = "axial", time_instant: int = 0) -> tuple[np.memmap, int]:
    
    """
    Input:
    - nii_data: data structure of a NifTi file
    - slice_idx: index for a slice to display
    - axis: reference plane, choices are "axial" (default), "sagittal" and "coronal"
    - time_instant: to choose a time instant when nii_data.shape is > 3 (fMRI for example), default is 0.
    
    Output:
    - slice: the 2D structure containing the slice
    - n_slices: number of slices along the chosen axis (usefull for titles/labels)
    """

    axis_to_shape_idx = {"axial": 2, "sagittal": 1, "coronal": 0} # axial represents the shape 2, sagittal shape 1, coronal shape 0
    
    if axis not in axis_to_shape_idx:
        raise ValueError('Axis must be axial, sagittal or coronal')
    
    shape_idx = axis_to_shape_idx[axis]
    n_slices = nii_data.shape[shape_idx]
    
    is_4d = len(nii_data.shape) > 3
    if is_4d:
        assert slice_idx < n_slices and time_instant < nii_data.shape[3], \
            f"You need to choose a slice index < {n_slices} or a time instant < {nii_data.shape[3]}."
    else:
        assert slice_idx < n_slices, \
            f"You need to choose a slice index < {n_slices}."
    
    if axis == "axial":
        slice = nii_data[:, :, slice_idx, time_instant] if is_4d else nii_data[:, :, slice_idx]
    elif axis == "sagittal":
        slice = nii_data[:, slice_idx, :, time_instant] if is_4d else nii_data[:, slice_idx, :]
    else:  # coronal
        slice = nii_data[slice_idx, :, :, time_instant] if is_4d else nii_data[slice_idx, :, :]
    
    return slice, n_slices

# ------------------------------------------------------------------- #
# Given a NiFti data structure, display a group of slices.            #
# You can give a predefined list of indexes or sample them randomnly, #
# or set a starting idx with a step                                   #
# ------------------------------------------------------------------- #
def displayGroupOfSlices(nii_data: np.memmap, count, starting_idx: int = 0, sparse: bool = False,
                         idxs_list: list = None, disp_step: int = 1, axis: str = "axial", time_instant: int = 0,
                         max_cols: int = 4) -> None:
    """
    Input: 
    - nii_data: data structure to display
    - count: number of slices to display
    - starting_idx: first idx to display (used when sparse=False)
    - sparse: boolean, if True you can provide a list of idxs to display, if False starting idx and step are used (default = False)
    - idxs_list: list of indexes to display in case of sparse = True (default = None)
    - disp_step: step between one displayed slice and the next, used when sparse=False (default = 1)
    - axis: reference plane, "axial" (default), "sagittal" or "coronal"
    - time_instant: time point to use if nii_data is 4D (default = 0)
    - max_cols: maximum number of images per row (default = 4)

    Output:
    - None
    """

    axis_to_shape_idx = {"axial": 2, "sagittal": 1, "coronal": 0}
    if axis not in axis_to_shape_idx:
        raise ValueError('Axis must be axial, sagittal or coronal')
    
    n_slices = nii_data.shape[axis_to_shape_idx[axis]]

    # --- Build the list of indices to display ---
    if sparse:
        if idxs_list is None: # Choose idxs randomnly
            print(f"No list provided: choosing {count} random indexes.")
            idxs_list = sorted(np.random.choice(n_slices, size = count, replace = False))
        else:
            assert max(idxs_list) < n_slices, \
                f"All indices in idxs_list must be < {n_slices}."
            count = len(idxs_list)
        indices = idxs_list
    else:
        indices = [starting_idx + i * disp_step for i in range(count)]
        assert max(indices) < n_slices, \
            f"starting_idx + count * disp_step exceeds the number of slices ({n_slices})."

    # --- Build the grid ---
    n_cols = min(max_cols, count)
    n_rows = int(np.ceil(count / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize = (4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, indices):
        slice, num_slices = getSlice(nii_data, idx, axis, time_instant)
        ax.imshow(np.rot90(slice, 1), cmap='gray', vmin=np.min(slice))
        ax.set_title(f'Slice {idx} of {num_slices}')
        ax.axis('off')

    # Hide empty subplots
    for ax in axes[len(indices):]:
        ax.axis('off')

    plt.suptitle(f'{axis.capitalize()} view — group of {count} slices')
    plt.tight_layout()
    plt.show()

































