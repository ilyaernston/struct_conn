import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.mixture import GaussianMixture


def cluster_and_visualize_distances(
    distance_matrix: pd.DataFrame,
    metadata_df: pd.DataFrame,
    embedding_method: str = 'MDS',
    plot_mode: str = 'save',
    output_dir: str = None,
    k_number: int = None,
    export_params: bool = False,
    metadata_plots: bool = False
) -> pd.DataFrame:
    """
    1) Embed distances into 2D (MDS or t-SNE)
    2) Compute embedding quality (R² & stress for MDS; KL-divergence for t-SNE)
    3) Choose top cluster counts via BIC and plot clusters
    4) Optionally export embedding coordinates and cluster labels for all subjects

    Args:
        distance_matrix : square pd.DataFrame (index=subject_id)
        metadata_df     : DataFrame with 'subject_id' + any grouping columns
        embedding_method: 'MDS' or 'TSNE'
        plot_mode       : 'save', 'show', or 'both' for embedding in 2d with clusters as colour-coding
        output_dir      : directory to write outputs
        k_number        : k number to cluster into, computed automatically if None
        export_params   : if True, save a CSV of embedding coords and cluster labels
        metadata_plots  : if True, save plots of embedding with metadata as colour-coding

    Returns:
        labels_df       : DataFrame with subject_id, embedding dims, and cluster labels
        meta_df         : DataFrame with metadata on analysis: embedding method, embedding quality metric, k number, BIC score. Tracked to labels_df via 'run_id'
    """
    # ensure output directory
    output_dir = output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # prepare subjects and matrix
    subject_ids = distance_matrix.index.astype(str).tolist()
    D = distance_matrix.values
    n = D.shape[0]

    grouping_cols = [c for c in metadata_df.columns if c != 'subject_id']
    meta = metadata_df.set_index('subject_id').loc[subject_ids, grouping_cols]

    # initialize labels_df with subject IDs
    labels_df = pd.DataFrame({'subject_id': subject_ids})

    # choose embedding
    embeddings_map = {}
    if embedding_method.upper() == 'TSNE':
        perp = 50
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
        i, j = np.triu_indices(n, k=1)
        d_orig = D[i, j]
        D_embed = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
        d_hat = D_embed[i, j]
        RSS, TSS = np.sum((d_orig - d_hat) ** 2), np.sum(d_orig ** 2)
        R2 = 1 - RSS / TSS
        stress = np.sqrt(RSS / TSS)
        quality_text = f"R²={R2:.3f}, Stress={stress:.3f}"
        embeddings_map['mds'] = emb

    # loop embeddings: add coords, cluster, plot
    for key, emb in embeddings_map.items():
        # add embedding coordinates to labels_df
        labels_df['dim1'] = emb[:, 0]
        labels_df['dim2'] = emb[:, 1]

        # determine k
        if k_number is not None:
            if isinstance(k_number, int):
                best_k = k_number
                best_bic = 'number of k set manually'
            else:
                raise TypeError("Only integers are allowed as k_number")
        else:
            ks = np.arange(2, min(10, n - 1) + 1)
            bic_scores = [GaussianMixture(n_components=k, random_state=42).fit(emb).bic(emb) for k in ks]
            best_k = ks[np.argmin(bic_scores)]
            best_bic = np.argmin(bic_scores)

        # fit clusters
        gm = GaussianMixture(n_components=best_k, random_state=42).fit(emb)
        labels = gm.predict(emb)
        labels_df['cluster'] = labels

        run_id = f"{key}_k{best_k}" # e.g. "MDS_k5"
        labels_df['run_id'] = run_id

        meta_df = pd.DataFrame([{
            'run_id'          : run_id,
            'embedding_method': key,
            'quality'         : quality_text,
            'n_clusters'      : best_k,
            'bic_score'       : best_bic
        }])

        # plotting code (unchanged)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=labels, cmap='tab10', s=60,
            edgecolor='k', alpha=0.8
        )
        if type(best_bic) == str:
            ax.set_title(f"{key.upper()} + GMM (k={best_k})\n{quality_text}")
        else:
            ax.set_title(f"{key.upper()} + GMM (k={best_k})\n{quality_text}\nBIC={best_bic}")
        ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.grid(alpha=0.3)
        handles, _ = sc.legend_elements()
        ax.legend(handles, [str(i) for i in range(best_k)], title='cluster')
        if plot_mode in ('save', 'both'):
            fn = os.path.join(output_dir, f"{key}_k{best_k}_clusters.png")
            fig.savefig(fn, dpi=300, bbox_inches='tight')
        if plot_mode in ('show', 'both'): plt.show()
        plt.close(fig)

        # metadata-colored plots if requested
        if metadata_plots:
            for col in grouping_cols:
                values = meta[col]
                fig, ax = plt.subplots(figsize=(7, 6))
                for val in values.unique():
                    mask = (values == val).values
                    ax.scatter(emb[mask, 0], emb[mask, 1], label=str(val), alpha=0.8, s=60)
                ax.set_title(f"{embedding_method.upper()} embedding colored by {col}")
                ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.grid(alpha=0.3)
                ax.legend(title=col, bbox_to_anchor=(1, 1))
                if plot_mode in ('save', 'both'):
                    fn2 = os.path.join(output_dir, f"{key.lower()}_by_{col}.png")
                    fig.savefig(fn2, dpi=300, bbox_inches='tight')
                    print(f"Saved group-colored plot → {fn2}")
                if plot_mode in ('show', 'both'): plt.show()
                plt.close(fig)

    # export parameters if requested
    if export_params:
        export_path = os.path.join(output_dir, 'embedding_and_clustering.csv')
        labels_df.to_csv(export_path, index=False)
        print(f"Exported embedding parameters and clusters → {export_path}")

        export_path = os.path.join(output_dir, 'embedding_and_clustering_meta.csv')
        meta_df.to_csv(export_path, index=False)
        print(f"Exported embedding parameters and clusters → {export_path}")

    return labels_df, meta_df


distance_matrix = pd.read_csv('/Users/elijah/Desktop/thesis/struct_conn_output/distance_matrix_803_subjects.csv', index_col=0, header=0)
out_dir = '/Users/elijah/Desktop/thesis/tests_4.1'
# load and prepare labels
labels_df = pd.read_csv('/Users/elijah/Desktop/thesis/AllSubjectsMeta - Sheet1.csv')
labels_df['Subject Code'] = 'sub-' + labels_df['Subject Code']
labels_df = labels_df.rename(columns={'Subject Code':'subject_id'})

cluster_and_visualize_distances(distance_matrix=distance_matrix, embedding_method='mds', 
                                output_dir=out_dir, metadata_df=labels_df, export_params=True)