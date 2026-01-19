import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import scanpy as sc
import sys
sys.path.append('./TransPath')
sys.path.append('./src')

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR for a 1D array."""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

def cluster_vs_rest_pathways(
    predictions: np.ndarray,
    cluster_labels,
    pathway_names,
    *,
    fdr_thresh: float = 0.05,
    min_mean_diff: float = 0.0,
    alternative: str = "greater",   # "greater" = enriched in cluster; "two-sided" also possible
    top_k_fallback: int = 30,
):
    """
    Cluster vs rest DE-like pathway testing using Mann–Whitney U per pathway.

    Returns:
      results_per_cluster: dict cluster -> DataFrame (sorted by qval then effect)
      selected_pathways: list of pathway names (union of significant, or fallback top_k)
    """
    X = np.asarray(predictions)

    n_cells, n_path = X.shape

    labels = np.asarray(cluster_labels)

    pathway_names = np.asarray(pathway_names)

    unique_clusters = pd.unique(labels)

    results_per_cluster = {}
    selected = set()

    for c in unique_clusters:
        in_mask = (labels == c)
        out_mask = ~in_mask

        Xin = X[in_mask]
        Xout = X[out_mask]

        mean_in = Xin.mean(axis=0)
        mean_out = Xout.mean(axis=0)
        mean_diff = mean_in - mean_out

        pvals = np.ones(n_path, dtype=float)
        for j in range(n_path):
            a = Xin[:, j]
            b = Xout[:, j]

            if np.all(a == a[0]) and np.all(b == b[0]) and a[0] == b[0]:
                pvals[j] = 1.0
                continue

            pvals[j] = mannwhitneyu(a, b, alternative=alternative, method="auto").pvalue

        qvals = bh_fdr(pvals)

        df = pd.DataFrame({
            "pathway": pathway_names,
            "mean_in": mean_in,
            "mean_out": mean_out,
            "mean_diff": mean_diff,
            "pval": pvals,
            "qval": qvals,
        })

        # Filter + sort
        df_sig = df[(df["qval"] <= fdr_thresh) & (df["mean_diff"] >= min_mean_diff)].copy()
        df_sig.sort_values(["qval", "mean_diff"], ascending=[True, False], inplace=True)

        if df_sig.empty:
            df_fallback = df.sort_values("mean_diff", ascending=False).head(top_k_fallback)
            chosen = df_fallback["pathway"].tolist()
        else:
            chosen = df_sig["pathway"].tolist()

        selected.update(chosen)
        results_per_cluster[c] = (df_sig if not df_sig.empty else df.sort_values("mean_diff", ascending=False))

    selected_pathways = list(selected)
    return results_per_cluster, selected_pathways


def top_pathways_for_pc(pca_obj, pc_idx, names, top_k=10):
    loadings = pca_obj.components_[pc_idx]
    order = np.argsort(np.abs(loadings))[::-1][:top_k]
    return names[order], loadings[order]

def plot_top_patways(top_names, top_loads, ax_bar, pc):
  pos = top_loads > 0
  top_names = top_names[pos]
  top_loads = top_loads[pos]

  order = np.argsort(top_loads)
  top_names = top_names[order]
  top_loads = top_loads[order]

  x = np.arange(len(top_names))

  norm = mcolors.Normalize(vmin=top_loads.min(), vmax=top_loads.max())
  colors = cm.viridis(norm(top_loads))

  ax_bar.scatter(
      x,
      top_loads,
      s=140,
      c=colors,
      edgecolor="black",
      linewidth=0.6,
      zorder=3,
  )

  ax_bar.set_xticks(x)
  ax_bar.set_xticklabels(
      [n.split(",")[0] for n in top_names],
      rotation=45, ha="right", fontsize=10
  )
  ax_bar.set_ylabel("PC loading")
  ax_bar.set_title(f"Top pathways defining PC{pc}", fontsize=13)
  ax_bar.spines["top"].set_visible(False)
  ax_bar.spines["right"].set_visible(False)

  plt.tight_layout()
  plt.show()


def pca_anaylsis(prediction_df, obj, n_pcs=3):

  pathway_names = np.asarray(prediction_df.columns)

  pca = PCA()
  X_pca = pca.fit_transform(prediction_df)

  scaler = StandardScaler()
  X_pca_z = scaler.fit_transform(X_pca[:, :n_pcs])

  for i in range(n_pcs):
      obj.obs[f"PC{i+1}"] = X_pca_z[:, i]

  for pc in range(1, n_pcs + 1):

      fig, (ax_spatial, ax_bar) = plt.subplots(
          1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.4, 1]}
      )

      sc.pl.spatial(
          obj,
          color=f"PC{pc}",
          spot_size=100,
          cmap="coolwarm",
          vmin=-1, vmax=1,
          show=False,
          ax=ax_spatial,
      )
      ax_spatial.set_title(f"PC{pc} score (z-scored)", fontsize=13)
      ax_spatial.set_xlabel("")
      ax_spatial.set_ylabel("")

      top_names, top_loads = top_pathways_for_pc(
          pca, pc_idx=pc - 1, names=pathway_names, top_k=10
      )
      plot_top_patways(top_names, top_loads, ax_bar, pc)
