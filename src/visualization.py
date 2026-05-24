"""
Visualization module for the "To bee or not to bee" project.

Includes:
- Class distribution plots (bug type and species)
- PCA projection in 2D
- Non-linear projections: t-SNE, UMAP (if available), Isomap as a fallback
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler

try:
    import umap  # umap-learn
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# -----------------------------------------------------------------------------
# Class distribution
# -----------------------------------------------------------------------------

def plot_class_distribution(df, bug_type_col='bug type', species_col='species',
                            save_path=None):
    """Two-panel figure showing the distribution of bug_type and species."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts_bt = df[bug_type_col].value_counts()
    sns.barplot(x=counts_bt.index, y=counts_bt.values, ax=axes[0])
    axes[0].set_title(f'Distribution of {bug_type_col}')
    axes[0].set_ylabel('Count')
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(30)

    counts_sp = df[species_col].value_counts()
    sns.barplot(x=counts_sp.index, y=counts_sp.values, ax=axes[1])
    axes[1].set_title(f'Distribution of {species_col}')
    axes[1].set_ylabel('Count')
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment('right')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# -----------------------------------------------------------------------------
# Projections
# -----------------------------------------------------------------------------

def _prepare_features(X):
    """Standardize features before projection."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def project_pca(X, n_components=2):
    """Linear PCA projection."""
    Xs = _prepare_features(X)
    pca = PCA(n_components=n_components, random_state=0)
    Z = pca.fit_transform(Xs)
    return Z, pca


def project_tsne(X, perplexity=30, random_state=0):
    """Non-linear t-SNE projection."""
    Xs = _prepare_features(X)
    perp = min(perplexity, max(5, len(Xs) // 4))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state,
                init='pca', learning_rate='auto')
    return tsne.fit_transform(Xs)


def project_isomap(X, n_neighbors=10):
    """Non-linear Isomap projection."""
    Xs = _prepare_features(X)
    k = min(n_neighbors, len(Xs) - 1)
    iso = Isomap(n_components=2, n_neighbors=k)
    return iso.fit_transform(Xs)


def project_umap(X, n_neighbors=15, min_dist=0.1, random_state=0):
    """Non-linear UMAP projection (requires umap-learn)."""
    if not UMAP_AVAILABLE:
        raise ImportError('umap-learn is not installed. Run: pip install umap-learn')
    Xs = _prepare_features(X)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(Xs)


# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------

def plot_projection(Z, labels, title, save_path=None, ax=None):
    """Scatter plot of a 2D projection colored by class labels."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    unique_labels = sorted(pd.Series(labels).dropna().unique().tolist())
    palette = sns.color_palette('tab10', n_colors=max(len(unique_labels), 3))
    for i, lab in enumerate(unique_labels):
        idx = np.array(labels) == lab
        ax.scatter(Z[idx, 0], Z[idx, 1], s=35, alpha=0.75,
                   color=palette[i % len(palette)], label=str(lab),
                   edgecolor='white', linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.legend(loc='best', frameon=True)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
