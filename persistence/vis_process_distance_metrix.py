import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.mixture import GaussianMixture
import seaborn as sns

def cluster_and_visualize_distances(
    distance_matrix: pd.DataFrame,
    metadata_df: pd.DataFrame,
    embedding_method: str = 'MDS',
    plot_mode: str = 'save',
    output_dir: str = None,
    k_number: int = None
) -> pd.DataFrame:
    """
    1) Embed distances into 2D (MDS or t-SNE)
    2) Compute embedding quality (R² & stress for MDS; KL-divergence for t-SNE)
    3) Choose top cluster counts via BIC and plot clusters
    4) 
    5) Save all plots and return enriched labels_df

    Args:
        distance_matrix : square pd.DataFrame (index=subject_id)
        metadata_df     : DataFrame with 'subject_id' + any grouping columns
        embedding_method: 'MDS' or 'TSNE'
        plot_mode       : 'save', 'show', or 'both'
        output_dir      : directory to write outputs
        k_number        : k number to cluster into, comuted automatically if none provided

    Returns:
        labels_df       : input labels_df (unchanged)
    """
    # ensure output directory
    output_dir = output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # pull subjects and raw matrix
    subject_ids = distance_matrix.index.astype(str).tolist()
    D = distance_matrix.values
    n = D.shape[0]

    grouping_cols = [c for c in metadata_df.columns if c != 'subject_id']
    meta = metadata_df.set_index('subject_id').loc[subject_ids, grouping_cols]

    labels_df = pd.DataFrame({'subject_id': subject_ids})

    # choose embedding
    if embedding_method.upper() == 'TSNE':
        
        #perps = [5, 15, 30, 50]
        perp = 50

        embeddings_map = {}
        #for perp in perps:
        tsne = TSNE(
            perplexity=perp,
            metric='precomputed',
            n_components=2,
            learning_rate='auto',
            init='random',
            random_state=42
        )
        emb = tsne.fit_transform(D)
        quality_text = f"KL={tsne.kl_divergence_:.3f}"

        embeddings_map[f'tsne_pp{perp}'] = emb

    else:
        mds = MDS(
            n_components=2,
            dissimilarity='precomputed',
            normalized_stress='auto',
            random_state=42
        )
        emb = mds.fit_transform(D)
        embeddings_map = {'mds': emb}
        i, j = np.triu_indices(n, k=1)
        d_orig = D[i, j]
        D_embed = np.linalg.norm(emb[:,None,:] - emb[None,:,:], axis=2)
        d_hat = D_embed[i, j]
        RSS, TSS = np.sum((d_orig - d_hat)**2), np.sum(d_orig**2)
        R2 = 1 - RSS/TSS
        stress = np.sqrt(RSS/TSS)
        quality_text = f"R²={R2:.3f}, Stress={stress:.3f}"

    # for each embedding method/key, cluster + plot
    for key, emb in embeddings_map.items():
        
        if k_number != None:
            if type(k_number) is int:
                best_k = k_number
            else:
                raise TypeError("Only integers are allowed as k numbers")
        # pick best k via BIC, if none provided
        else:
            ks = np.arange(2, min(10, n-1) + 1)
            bic_scores = []
            for k in ks:
                gm = GaussianMixture(n_components=k, random_state=42).fit(emb)
                bic_scores.append(gm.bic(emb))
            best_k = ks[np.argmin(bic_scores)]

        # cluster
        gm = GaussianMixture(n_components=best_k, random_state=42).fit(emb)
        labels = gm.predict(emb)
        # add to labels_df
        cluster_col = f'{key}_cluster_{best_k}'
        labels_df[cluster_col] = labels

        # plot clusters
        fig, ax = plt.subplots(figsize=(7,6))
        sc = ax.scatter(
            emb[:,0], emb[:,1],
            c=labels, cmap='tab10', s=60,
            edgecolor='k', alpha=0.8
        )
        ax.set_title(f"{key.upper()} + GMM (k={best_k})\n{quality_text}")
        ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.grid(alpha=0.3)
        # legend for clusters
        handles, _ = sc.legend_elements()
        ax.legend(handles, [str(i) for i in range(best_k)],
                  title='cluster')
        if plot_mode in ('save', 'both'):
            fn = os.path.join(output_dir, f"{key}_k{best_k}_clusters.png")
            fig.savefig(fn, dpi=300, bbox_inches='tight')
        if plot_mode in ('show','both'): plt.show()
        plt.close(fig)

        for col in grouping_cols:
            values = meta[col]
            uni = values.unique()
            fig, ax = plt.subplots(figsize=(7,6))
            for val in uni:
                mask = (values == val).values
                ax.scatter(emb[mask,0], emb[mask,1],
                        label=str(val), alpha=0.8, s=60)
            ax.set_title(f"{embedding_method.upper()} embedding colored by {col}")
            ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.grid(alpha=0.3)
            ax.legend(title=col, bbox_to_anchor=(1,1))
            if plot_mode in ('save', 'both'):
                fn = os.path.join(output_dir, f"{embedding_method.lower()}_by_{col}.png")
                fig.savefig(fn, dpi=300, bbox_inches='tight')
            if plot_mode in ('show','both'): plt.show()
            plt.close(fig)
            print(f"Saved group‐colored plot → {fn}")

    return labels_df


distance_matrix = pd.read_csv('/Users/elijah/Desktop/thesis/struct_conn_output/distance_matrix_803_subjects.csv', index_col=0, header=0)
out_dir = '/Users/elijah/Desktop/thesis/tests_3_mdsk8'
labels_df = pd.read_csv('/Users/elijah/Desktop/thesis/tests_2/all_labels.csv')
labels_df['subject_id'] = 'sub-' + labels_df['subject_id']

cluster_and_visualize_distances(distance_matrix=distance_matrix, embedding_method='mds', 
                                output_dir=out_dir, metadata_df=labels_df, k_number=8)