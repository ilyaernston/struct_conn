
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import squareform

'''
def cluster_and_visualize_distances(
    distance_matrix: np.ndarray,
    embedding_method: str = 'MDS',
    plot_mode: str = 'save',
    save_dir: str = None
):
    """
    1) Embed distances into 2D (MDS or t-SNE)
    2) Compute embedding quality (R² & stress for MDS; KL-divergence for t-SNE)
    3) Choose top-3 cluster counts via BIC
    4) For each of those 3 k’s, plot the embedding colored by cluster and save/show
    5) Return an (N,3) array of labelings for each top-k
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS, TSNE
    from sklearn.mixture import GaussianMixture
    from scipy.spatial.distance import squareform

    n = distance_matrix.shape[0]
    save_dir = save_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    # perform embedding
    if embedding_method.upper() == 'TSNE':
        """tsne = TSNE(
            n_components=2,
            metric='precomputed',
            learning_rate='auto',
            init='random',
            random_state=42
        )
        embeddings = tsne.fit_transform(distance_matrix)
        quality_text = f"KL={tsne.kl_divergence_:.3f}""""

        perps_to_test = [5, 15, 30, 50]
        n_perps = len(perps_to_test)

        # prepare output array
        cluster_labels = np.zeros((n, n_perps), dtype=int)
        
        for perp, col in zip(perps_to_test, range(n_perps)):
            tsne = TSNE(perplexity=perp, n_components=2, random_state=42)
            embeddings = tsne.fit_transform(distance_matrix)
            quality_text = f"KL={tsne.kl_divergence_:.3f}"

            plt.scatter(embeddings[:,0], embeddings[:,1], s=5)
            plt.title(f"t-SNE (PP={perp})")
            fname = os.path.join(
                save_dir,
                f"embedding_{embedding_method.lower()}_pp{perp}.png"
            )
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved plot for pp={perp} → {fname}")
            
            # find top k via BIC
            max_k = min(10, n - 1)
            ks = np.arange(2, max_k + 1)
            bic = []
            for k in ks:
                gmm = GaussianMixture(n_components=k, random_state=42)
                gmm.fit(embeddings)
                bic.append(gmm.bic(embeddings))
    
            optimal_k = ks[np.argmin(bic)]

            # plot for each perplexity
            gmm = GaussianMixture(n_components=optimal_k, random_state=42)
            labels = gmm.fit_predict(embeddings)
            cluster_labels[:, col] = labels

            fig, ax = plt.subplots(figsize=(7, 6))
            sc = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=labels,
                cmap='tab10',
                s=60,
                edgecolor='k',
                alpha=0.8
            )
            ax.set_title(f"{embedding_method.upper()} (pp={perp}) + GMM (k={optimal_k})\n{quality_text}")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.grid(alpha=0.3)

            fname = os.path.join(
                save_dir,
                f"embedding_{embedding_method.lower()}_pp{perp}_k{optimal_k}_clusters.png"
            )
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved plot for pp={perp} k={optimal_k} → {fname}")

            if plot_mode in ('show', 'both'):
                plt.show()
            plt.close(fig)
            
            # save clusterings as .csv
            csv_name = os.path.join(save_dir, f'PDs_labels_embedding_{embedding_method.lower()}_pp{perp}_k{optimal_k}_clusters.csv')
            np.savetxt(csv_name, cluster_labels, delimiter=',')
            print(f'→ Saved cluterings to {csv_name}')
    else:
        mds = MDS(
            n_components=2,
            dissimilarity='precomputed',
            normalized_stress='auto',
            random_state=42
        )
        embeddings = mds.fit_transform(distance_matrix)

        # compute R² & stress
        i, j = np.triu_indices(n, k=1)
        d_orig = distance_matrix[i, j]
        diffs = embeddings[:, None, :] - embeddings[None, :, :]
        D_embed = np.linalg.norm(diffs, axis=2)
        d_hat = D_embed[i, j]
        RSS = np.sum((d_orig - d_hat) ** 2)
        TSS = np.sum(d_orig ** 2)
        R2 = 1 - RSS / TSS
        stress = np.sqrt(RSS / TSS)
        quality_text = f"R²={R2:.3f}, Stress={stress:.3f}"
    
        # find top-3 ks via BIC
        max_k = min(10, n - 1)
        ks = np.arange(2, max_k + 1)
        bic = []
        for k in ks:
            g = GaussianMixture(n_components=k, random_state=42).fit(embeddings)
            bic.append(g.bic(embeddings))
        bic = np.array(bic)
        top3_idx = np.argsort(bic)[:3]
        top3_ks = ks[top3_idx]

        # prepare output array
        cluster_labels = np.zeros((n, 3), dtype=int)

        # plot for each k
        for col, k in enumerate(top3_ks):
            gmm = GaussianMixture(n_components=k, random_state=42)
            labels = gmm.fit_predict(embeddings)
            cluster_labels[:, col] = labels

            fig, ax = plt.subplots(figsize=(7, 6))
            sc = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=labels,
                cmap='tab10',
                s=60,
                edgecolor='k',
                alpha=0.8
            )
            ax.set_title(f"{embedding_method.upper()} + GMM (k={k})\n{quality_text}")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.grid(alpha=0.3)

            fname = os.path.join(
                save_dir,
                f"embedding_{embedding_method.lower()}_k{k}_clusters.png"
            )
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved plot for k={k} → {fname}")

            if plot_mode in ('show', 'both'):
                plt.show()
            plt.close(fig)
        
        # save clusterings as .csv
        csv_name = os.path.join(save_dir, f'PDs_labels_embedding_{embedding_method.lower()}_k{k}_clusters.csv')
        np.savetxt(csv_name, cluster_labels, delimiter=',')
        print(f'→ Saved cluterings to {csv_name}')
'''

def cluster_and_visualize_distances(
    distance_matrix: pd.DataFrame,
    embedding_method: str = 'MDS',
    plot_mode: str = 'save',
    output_dir: str = None
):
    """
    1) Embed distances into 2D (MDS or t-SNE)
    2) Compute embedding quality (R² & stress for MDS; KL-divergence for t-SNE)
    3) Choose top cluster counts via BIC
    4) For each embedding / parameter set, plot & save
    5) Return a DataFrame of cluster labels with subject_id first column.
    
    Args:
    ----
        distance_matrix : square pd.DataFrame (index=subject_id)
        embedding_method: 'MDS' or 'TSNE'
        plot_mode       : 'save', 'show', or 'both'
        save_dir        : directory to write outputs
    """
    save_dir = output_dir or os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    # extract IDs & raw matrix
    subject_ids = distance_matrix.index.astype(str).tolist()
    D = distance_matrix.values

    n = D.shape[0]
    labels_df = pd.DataFrame({'subject_id': subject_ids})

    if embedding_method.upper() == 'TSNE':
        # TSNE branch
        perps_to_test = [5, 15, 30, 50]
        for perp in perps_to_test:
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

            # scatter for visual inspection
            plt.scatter(emb[:,0], emb[:,1], s=5)
            plt.title(f"t-SNE (perplexity={perp})")
            png = os.path.join(save_dir, f"tsne_pp{perp}.png")
            plt.savefig(png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved t-SNE embedding (pp={perp}) → {png}")

            # pick best k via BIC
            ks = np.arange(2, min(10, n-1) + 1)
            bics = []
            for k in ks:
                gm = GaussianMixture(n_components=k, random_state=42).fit(emb)
                bics.append(gm.bic(emb))
            best_bic = np.argmin(bics)
            best_k = ks[best_bic]

            # cluster & plot
            gm = GaussianMixture(n_components=best_k, random_state=42)
            labs = gm.fit_predict(emb)
            col = f'PD_cluster(pp={perp};k={best_k};bic={best_bic})' 
            labels_df[col] = labs

            fig, ax = plt.subplots(figsize=(7,6))
            ax.scatter(emb[:,0], emb[:,1], c=labs, cmap='tab10',
                       s=60, edgecolor='k', alpha=0.8)
            ax.set_title(f"t-SNE (pp={perp}) + GMM (k={best_k})\n{quality_text}")
            ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.grid(alpha=0.3)
            out_png = os.path.join(save_dir, f"tsne_pp{perp}_k{best_k}.png")
            fig.savefig(out_png, dpi=300, bbox_inches='tight')
            if plot_mode in ('show','both'): plt.show()
            plt.close(fig)
            print(f"Saved clustering (pp={perp}, k={best_k}) → {out_png}")

        # save TSNE labels (for all pp values)
        full_csv = os.path.join(save_dir, "PDs_labels_tsne.csv")
        labels_df.to_csv(full_csv, index=False)
        print(f"→ Saved full TSNE labels to {full_csv}")

    else:
        # MDS branch
        mds = MDS(
            n_components=2,
            dissimilarity='precomputed',
            normalized_stress='auto',
            random_state=42
        )
        emb = mds.fit_transform(D)
        i, j = np.triu_indices(n, k=1)
        d_orig = D[i, j]
        D_embed = np.linalg.norm(emb[:,None,:] - emb[None,:,:], axis=2)
        d_hat = D_embed[i, j]
        RSS, TSS = np.sum((d_orig - d_hat)**2), np.sum(d_orig**2)
        R2 = 1 - RSS/TSS
        stress = np.sqrt(RSS/TSS)
        quality_text = f"R²={R2:.3f}, Stress={stress:.3f}"

        ks = np.arange(2, min(10, n-1) + 1)
        bics = [GaussianMixture(n_components=k, random_state=42)
                   .fit(emb).bic(emb) for k in ks]
        top3 = ks[np.argsort(bics)[:3]]

        for k, bic in zip(top3, bics):
            gm = GaussianMixture(n_components=k, random_state=42)
            labs = gm.fit_predict(emb)
            col = f'PD_cluster(k={k};bic={bic:.3f})'
            labels_df[col] = labs

            fig, ax = plt.subplots(figsize=(7,6))
            ax.scatter(emb[:,0], emb[:,1], c=labs, cmap='tab10',
                       s=60, edgecolor='k', alpha=0.8)
            ax.set_title(f"MDS + GMM (k={k})\n{quality_text}")
            ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.grid(alpha=0.3)
            out_png = os.path.join(save_dir, f"mds_k{k}.png")
            fig.savefig(out_png, dpi=300, bbox_inches='tight')
            if plot_mode in ('show','both'): plt.show()
            plt.close(fig)
            print(f"Saved MDS clustering (k={k}) → {out_png}")

        # save MDS labels
        out_csv = os.path.join(save_dir, "PDs_labels_mds.csv")
        labels_df.to_csv(out_csv, index=False)
        print(f"→ Saved MDS labels to {out_csv}")

distance_matrix = pd.read_csv('/Users/elijah/Desktop/thesis/struct_conn_output/distance_matrix_803_subjects.csv', index_col=0, header=0)
out_dir = '/Users/elijah/Desktop/thesis/tests_2'

cluster_and_visualize_distances(distance_matrix=distance_matrix, embedding_method='mds', output_dir=out_dir)