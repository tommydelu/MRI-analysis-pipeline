import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.stats import zscore, spearmanr
from sklearn.decomposition import PCA, FastICA

# ----------------------------------------------------------------- #
# This function, given a matrix (samples x variables) computes      #
# a similarity matrix according to the similarity measure given     #
# in input.                                                         #
# ----------------------------------------------------------------- #
def computeSimilarityMatrix(time_series: np.ndarray, similarity_measure: str = 'corr', plot: bool = True) -> np.ndarray:
    """
    Input
    - time_series: matrix of shape (num_time_pts x labels).
    - distance_measure: set to the distance measure you want to apply.
    - plot: set to True to see the matrix

    Output:
    - mat: similarity matrix
    """

    is_correlation = False # correlation measures quantify similarity (corr, spearman), distance measures quantify dissimilarity in a multi-dimensional space

    if similarity_measure == 'corr': # [-1,1]
        mat = np.corrcoef(time_series.T) # Because np.corrcoef wants (variables x features)
        is_correlation = True
    elif similarity_measure == 'spearman': # [-1,1]
        mat = spearmanr(time_series).statistic
        is_correlation = True
    elif similarity_measure == 'euclidean': # [0, inf], low value = more similar
        ts_z = zscore(time_series, axis=0)
        mat = euclidean_distances(ts_z.T, ts_z.T)
    elif similarity_measure == 'cheby':
        ts_z = zscore(time_series, axis=0)
        mat = cdist(ts_z.T, ts_z.T, metric='chebyshev')
    elif similarity_measure == 'mink':
        ts_z = zscore(time_series, axis=0)
        mat = cdist(ts_z.T, ts_z.T, metric='minkowski', p=3)
    else:
        raise ValueError(f"This metric is not supported: {similarity_measure}")

    if plot:
        plt.figure(figsize=(8, 7))
        vmin, vmax = (-1.0, 1.0) if is_correlation else (None, None)
        cbar_label = "Correlation" if is_correlation else "Distance"

        sns.heatmap(
            mat,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            annot=False,
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': cbar_label}
        )
        plt.title(f"Connectivity Matrix (200x200) - Metric: {similarity_measure}")
        plt.xlabel("Parcel ID")
        plt.ylabel("Parcel ID")
        plt.show()

    return mat

# ----------------------------------------------------------------- #
# This function extract the name of the region given it's index     #    
# in the csv.                                                       #
# ----------------------------------------------------------------- #
def extractRegionNameFromLabel(label_idx: int, df: pd.DataFrame) -> str:

    """
    Input
    - label_idx: index of the region whose name we want to know.
    - df: csv containing the mapping from idx to region

    Output:
    - region: contains the name of the region
    """
    try:
        region = df.iloc[label_idx - 1, 1]
        return region
    except Exception: # if error --> return the idx
        return f"ROI_{label_idx}"

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
def getTopKPairs(mat: np.ndarray, k: int = 10, is_distance: bool = False) -> list:
    """
    Input
    - mat:
    - k:
    - is_distance:

    Output
    - pairs:
    """
    n = mat.shape[0] # 200 in this case
    tri_indices = np.triu_indices(n, k=1) # k=1 to remove the main diagonal
    values = mat[tri_indices] # extract the valid values from mat
    
    if is_distance:
        sorted_order = np.argsort(values)[:k]  # Distanza minima = massima similarità
    else:
        sorted_order = np.argsort(values)[::-1][:k]  # Correlazione massima

    pairs = []
    for idx in sorted_order:
        i = tri_indices[0][idx] + 1 # to be coherent with the mapping csv
        j = tri_indices[1][idx] + 1
        val = values[idx]
        pairs.append((i, j, val))
    return pairs

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
def compareSimilarityMethods(matrices_dict: dict, df_regions: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Input
    - matrices_dict:
    - df_regions:
    - top_k:

    Output
    - summary_df: 
    """

    summary_rows = [] # placeholder to append each method results
    tri_idxs = np.triu_indices(200, k=1)
    
    pearson_vec = matrices_dict['corr'][tri_idxs] # I use as reference 'corr' because it is the most used in the literature
    top_pairs_dict = {}

    for name, mat in matrices_dict.items():
        # Riconosce correttamente le distanze in base alla lista dei metodi
        is_dist = name in ['euclidean', 'cheby', 'mink']
        vec = mat[tri_idxs]
        
        global_rank_agreement, _ = spearmanr(pearson_vec, vec)
        
        top_pairs = getTopKPairs(mat, k=top_k, is_distance=is_dist)

        unique_pairs_set = set()
        for pair in top_pairs:
            i = pair[0]
            j = pair[1]
            ordered_pair = (min(i, j), max(i, j)) # create a set so that duplicates are removed
            unique_pairs_set.add(ordered_pair)

        top_pairs_dict[name] = unique_pairs_set

        top3_labels_list = []
        for pair in top_pairs[:3]:
            i = pair[0]
            j = pair[1]
            val = pair[2]
            
            name_i = extractRegionNameFromLabel(i, df_regions)
            name_j = extractRegionNameFromLabel(j, df_regions)
            label_entry = f"{name_i} <-> {name_j} ({val:.2f})"
            
            top3_labels_list.append(label_entry)

        top3_str = "; ".join(top3_labels_list)
        
        summary_rows.append({
            'Method': name,
            'Type': 'Distanza' if is_dist else 'Correlazione',
            'Mean': round(np.mean(vec), 3),
            'Std': round(np.std(vec), 3),
            'Min': round(np.min(vec), 3),
            'Max': round(np.max(vec), 3),
            'Global Agreement vs Pearson (Spearman ρ)': round(global_rank_agreement, 3),
            'Top-3 Functional Pairs': top3_str
        })
        
    summary_df = pd.DataFrame(summary_rows)

    # Jaccard Index is also used in segmentation to check the quality of a mask, based on common area / total area
    pearson_top_set = top_pairs_dict['corr']
    overlap_list = []
    for name in matrices_dict.keys():
        common = len(pearson_top_set.intersection(top_pairs_dict[name]))
        jaccard = common / len(pearson_top_set.union(top_pairs_dict[name]))
        overlap_list.append(f"{common}/{top_k} ({jaccard * 100:.1f}%)")
        
    summary_df[f'Overlap Top-{top_k} w/ Pearson (Jaccard)'] = overlap_list
    
    return summary_df

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
def pcaAnalysis(mat: np.ndarray, n_comp: int, regions_csv: pd.DataFrame, plot: bool = True) -> pd.DataFrame:

    pca = PCA(n_components=n_comp)
    regions_projected = pca.fit_transform(mat)

    if plot:
        plt.figure(figsize=(10, 4))
        plt.bar( range(1, n_comp + 1), pca.explained_variance_ratio_, color='royalblue')
        plt.step(range(1, n_comp + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', color='red', linewidth=2, label='Cumulative')
        plt.xlabel('Principal Component (PC)')
        plt.ylabel('Variance description (%)')
        plt.title('PCA Plot - Similarity Matrix')
        plt.xticks(range(1, n_comp + 1))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend()
        plt.show()

    print(f"Total variance description (first 4 PC): {np.sum(pca.explained_variance_ratio_[:4]) * 100:.2f}%")
        
    # Identification of the best regions associated to PC1
    loadings_pc1 = pca.components_[0]
    top_roi_indices = np.argsort(np.abs(loadings_pc1))[::-1][:10] # Consider the absolute value, ascendent order, reverse to put first the one with higher value, then take the best 10
    top_pc1_df = pd.DataFrame({
        'Parcel_ID': top_roi_indices + 1,
        'Region Name': [extractRegionNameFromLabel(idx + 1, regions_csv) for idx in top_roi_indices],
        'Loading_PC1': np.round(loadings_pc1[top_roi_indices], 4)
    })

    print("\n--- Top 10 Regions associated to PC1) ---")

    return top_pc1_df

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #
def icaAnalysis(mat: np.ndarray, n_comp: int, regions_csv: pd.DataFrame, plot: bool = True) -> pd.DataFrame:

    ica = FastICA(n_components=n_comp, random_state=42, max_iter=500)
    sources = ica.fit_transform(mat)  # Shape: (200, n_comp)

    if plot:
        plt.figure(figsize=(12, 4))
        for comp_idx in range(min(4, n_comp)):
            plt.plot(sources[:, comp_idx], label=f'IC {comp_idx + 1}', alpha=0.8)
        plt.xlabel('Parcel Index (ROI)')
        plt.ylabel('Source Amplitude (a.u.)')
        plt.title('Independent Components (IC) Across Parcels')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

    # extraction of the first source
    ic1_weights = sources[:, 0]
    top_roi_indices = np.argsort(np.abs(ic1_weights))[::-1][:10]

    top_ic1_df = pd.DataFrame({
        'Parcel_ID': top_roi_indices + 1,
        'Region Name': [extractRegionNameFromLabel(idx + 1, regions_csv) for idx in top_roi_indices],
        'Weight_IC1': np.round(ic1_weights[top_roi_indices], 4)
    })

    print(f"\n--- Top 10 Regions associated to Independent Component 1 (IC1) ---")
    return top_ic1_df




















