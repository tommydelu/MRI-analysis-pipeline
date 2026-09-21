import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from fsl.wrappers import flirt, LOAD


def flirtRegistration(src, ref, configs, slice_idx=45, axis=2, out_mat_path=None):

    ref_data = ref.get_fdata() if hasattr(ref, 'get_fdata') else ref.data
    risultati = []
    n_configs = len(configs)
    
    fig, axes = plt.subplots(n_configs, 2, figsize=(10, 4.2 * n_configs), squeeze=False)

    best_nmi = -1
    best_mat_file = None
    temp_mats = []

    for i, config in enumerate(configs):
        temp_mat = f"temp_flirt_mat_{i}.mat"
        temp_mats.append(temp_mat)
        
        start_time = time.time()
        res = flirt(src, ref, out=LOAD, omat=temp_mat, **config)
        elapsed_time = time.time() - start_time
        
        aligned_img = res['out']
        aligned_data = aligned_img.get_fdata() if hasattr(aligned_img, 'get_fdata') else aligned_img.data
        
        # NMI computation: count how many times bin X of original img fall spatially on the bin Y of the aligned img
        # I cannot use continuous values for the intensity, the NMI would produce too many tissues --> I need to discretize
        bins = 50 # divide gray levels in 50 values
        ref_binned = np.digitize(ref_data.ravel(), np.linspace(ref_data.min(), ref_data.max(), bins)) # convert intensity values into these 50 levels, discretizing the img
        aligned_binned = np.digitize(aligned_data.ravel(), np.linspace(aligned_data.min(), aligned_data.max(), bins))
        nmi = normalized_mutual_info_score(ref_binned, aligned_binned)
        
        if nmi > best_nmi:
            best_nmi = nmi
            best_mat_file = temp_mat

        risultati.append({
            'config_id': i + 1,
            'dof': config.get('dof', 6),
            'cost': config.get('cost', 'corratio'),
            'interp': config.get('interp', 'trilinear'),
            'time_sec': round(elapsed_time, 2),
            'NMI_score': round(nmi, 4)
        })
        
        if axis == 0: s_ref, s_alig = ref_data[slice_idx, :, :], aligned_data[slice_idx, :, :]
        elif axis == 1: s_ref, s_alig = ref_data[:, slice_idx, :], aligned_data[:, slice_idx, :]
        else: s_ref, s_alig = ref_data[:, :, slice_idx], aligned_data[:, :, slice_idx]
            
        axes[i, 0].imshow(s_ref, cmap='gray', origin='lower')
        axes[i, 0].set_title(f"Reference - Slice {slice_idx}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(s_alig, cmap='gray', origin='lower')
        axes[i, 1].set_title(f"Aligned - Config {i+1}\nNMI: {nmi:.4f} | Time: {elapsed_time:.1f}s")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.show()

    # Saving the optimal matrix
    if out_mat_path and best_mat_file:
        out_mat_path = Path(out_mat_path)
        out_mat_path.parent.mkdir(parents=True, exist_ok=True)
        best_mat = np.loadtxt(best_mat_file)
        np.savetxt(str(out_mat_path), best_mat, fmt='%0.10f')
        print(f"Optimal matrix saved in: {out_mat_path}")

    # Remove files that are not optimal
    for f in temp_mats:
        Path(f).unlink(missing_ok=True)
        
    return pd.DataFrame(risultati)