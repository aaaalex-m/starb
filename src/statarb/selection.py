"""Choosing candidate pairs by clustering returns.

The distance method (Module 3) pairs stocks whose *prices* stayed close. Clustering pairs
stocks whose *returns* move together, which includes stocks whose prices have drifted apart.
Fit it on a formation window only: nothing here looks at the trading period.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def standardise_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column: subtract its mean and divide by its sample standard deviation.

    A column with fewer than two observations or with zero variance is dropped. It has no
    direction in return space, so it cannot be clustered.
    """
    std = returns.std(ddof=1)
    keep = returns.columns[(returns.notna().sum() >= 2) & (std > 0)]
    kept = returns[keep]
    return (kept - kept.mean()) / kept.std(ddof=1)


@dataclass
class PCAFeatures:
    """Output of :func:`pca_features`.

    The sign of each component is arbitrary (flipping a column of ``embeddings`` and the
    matching column of ``scores`` changes nothing).
    """

    embeddings: pd.DataFrame
    """Each stock's loadings on the principal components (tickers x PC1, PC2, ...)."""
    explained_variance_ratio: pd.Series
    """Share of variance explained by each component."""
    scores: pd.DataFrame
    """Component time series (dates x components): the common factors, up to scale."""


def pca_features(returns: pd.DataFrame, n_components: int | None = None) -> PCAFeatures:
    """Principal components of column-standardised returns.

    Dates are the observations and stocks are the variables, so the components are common
    factors and a stock's embedding is its loadings on those factors. Any stock with a
    missing return in the window is dropped (same rule as the distance method: a full
    history, or nothing). ``n_components=None`` keeps every component.
    """
    panel = returns.dropna(axis=1, how="any")
    z = standardise_returns(panel)
    n_obs, n_names = z.shape
    if n_names < 2 or n_obs < 2:
        raise ValueError("need at least two complete return series to run PCA")
    limit = min(n_obs, n_names)
    k = limit if n_components is None else n_components
    if not 1 <= k <= limit:
        raise ValueError(f"n_components must be between 1 and {limit}, got {k}")
    pca = PCA(n_components=k)
    scores = pca.fit_transform(z.to_numpy())
    names = [f"PC{i + 1}" for i in range(k)]
    return PCAFeatures(
        embeddings=pd.DataFrame(pca.components_.T, index=z.columns, columns=names),
        explained_variance_ratio=pd.Series(
            pca.explained_variance_ratio_, index=names, name="explained_variance_ratio"
        ),
        scores=pd.DataFrame(scores, index=z.index, columns=names),
    )


def kmeans_clusters(features: pd.DataFrame, k: int, seed: int = 0) -> pd.Series:
    """Cluster the rows of ``features`` with k-means.

    Rows with a missing value are dropped. Returns integer labels starting at 0, indexed
    like the remaining rows. ``seed`` fixes the initialisation (k-means is not unique).
    """
    clean = features.dropna()
    if not 1 <= k <= len(clean):
        raise ValueError(f"k must be between 1 and {len(clean)} (the number of rows), got {k}")
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = model.fit_predict(clean.to_numpy())
    return pd.Series(labels, index=clean.index, name="cluster")


@dataclass
class HierarchicalClusters:
    """Output of :func:`hierarchical_clusters`."""

    labels: pd.Series
    """Integer cluster labels starting at 0, indexed by ticker."""
    linkage: np.ndarray
    """SciPy linkage matrix on correlation distance, in the same ticker order as ``labels``."""


def hierarchical_clusters(returns: pd.DataFrame, n_clusters: int) -> HierarchicalClusters:
    """Average-linkage clustering on correlation distance, cut into ``n_clusters`` groups.

    The distance between two stocks is :math:`1 - \\mathrm{corr}` of their returns (Mantegna
    1999): perfectly correlated stocks sit on top of each other, uncorrelated stocks are one
    unit apart. Stocks with a missing return or zero variance are dropped. The linkage matrix
    is in the same column order as ``labels.index``, so a dendrogram can use that index as
    its leaf labels.
    """
    z = standardise_returns(returns.dropna(axis=1, how="any"))
    n_names = z.shape[1]
    if not 1 <= n_clusters <= n_names:
        raise ValueError(f"n_clusters must be between 1 and {n_names}, got {n_clusters}")
    corr = z.corr().to_numpy()
    distance = np.clip(1.0 - corr, 0.0, None)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    linked = linkage(squareform(distance, checks=False), method="average")
    # fcluster's labels start at 1; the rest of this module starts at 0.
    labels = fcluster(linked, t=n_clusters, criterion="maxclust") - 1
    return HierarchicalClusters(
        labels=pd.Series(labels, index=z.columns, name="cluster"),
        linkage=linked,
    )


def candidate_pairs(
    labels: pd.Series, returns: pd.DataFrame, n_per_cluster: int = 1, min_obs: int = 60
) -> pd.DataFrame:
    """The ``n_per_cluster`` most correlated pairs inside each cluster.

    ``labels`` maps a ticker to a cluster id. Pairs are ordered with ``a`` before ``b``
    alphabetically and sorted by correlation, highest first. A cluster with fewer than two
    tickers contributes nothing, and a pair needs at least ``min_obs`` overlapping returns.
    """
    if n_per_cluster < 1:
        raise ValueError(f"n_per_cluster must be at least 1, got {n_per_cluster}")
    if not labels.index.is_unique:
        raise ValueError("labels index has duplicate tickers")
    rows: list[tuple] = []
    for cluster, members in labels.groupby(labels):
        tickers = [t for t in members.index if t in returns.columns]
        if len(tickers) < 2:
            continue
        corr = returns[tickers].corr(min_periods=min_obs)
        pairs = []
        for i, a in enumerate(tickers):
            for b in tickers[i + 1 :]:
                value = corr.loc[a, b]
                if pd.notna(value):
                    left, right = sorted((str(a), str(b)))
                    pairs.append((left, right, cluster, float(value)))
        pairs.sort(key=lambda row: (-row[3], row[0], row[1]))
        rows.extend(pairs[:n_per_cluster])
    out = pd.DataFrame(rows, columns=["a", "b", "cluster", "correlation"])
    return out.sort_values(["correlation", "a", "b"], ascending=[False, True, True]).reset_index(
        drop=True
    )
