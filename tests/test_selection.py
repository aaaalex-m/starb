import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from statarb.pairs import distance
from statarb.selection import (
    candidate_pairs,
    hierarchical_clusters,
    kmeans_clusters,
    pca_features,
    standardise_returns,
)


def simulate_groups(
    n: int = 400,
    n_groups: int = 3,
    per_group: int = 4,
    seed: int = 0,
    drift: float = 0.0015,
    idio: float = 0.003,
):
    """Each group shares one factor. The second half of each group has a daily drift, so its
    prices diverge from the first half while the returns stay highly correlated."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(0.0, 0.012, size=(n, n_groups))
    columns, groups = {}, {}
    for g in range(n_groups):
        for i in range(per_group):
            name = f"g{g}s{i}"
            mu = drift if i >= per_group // 2 else 0.0
            columns[name] = mu + factors[:, g] + rng.normal(0.0, idio, n)
            groups[name] = g
    index = pd.bdate_range("2010-01-01", periods=n)
    returns = pd.DataFrame(columns, index=index)
    prices = 100 * (1 + returns).cumprod()
    return returns, prices, pd.Series(groups, name="group")


def test_standardise_returns_centers_scales_and_drops_dead_columns():
    returns = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "flat": [1.0, 1.0, 1.0], "short": [1.0, np.nan, np.nan]}
    )
    out = standardise_returns(returns)
    assert list(out.columns) == ["a"]
    assert out["a"].mean() == pytest.approx(0.0)
    assert out["a"].std(ddof=1) == pytest.approx(1.0)


def test_pca_embeddings_and_variance_share():
    returns, _, _ = simulate_groups()
    features = pca_features(returns, n_components=3)
    assert list(features.embeddings.columns) == ["PC1", "PC2", "PC3"]
    assert features.embeddings.shape == (12, 3)
    assert features.scores.shape[1] == 3
    assert 0.7 < features.explained_variance_ratio.sum() < 1.0
    assert features.explained_variance_ratio.is_monotonic_decreasing
    with pytest.raises(ValueError, match="n_components"):
        pca_features(returns, n_components=50)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clustering_recovers_the_hidden_groups(seed):
    returns, _, groups = simulate_groups(seed=seed)
    features = pca_features(returns, n_components=3)
    km = kmeans_clusters(features.embeddings, k=3, seed=0)
    hier = hierarchical_clusters(returns, n_clusters=3)
    assert adjusted_rand_score(groups.loc[km.index], km) == 1
    assert adjusted_rand_score(groups.loc[hier.labels.index], hier.labels) == 1
    assert hier.linkage.shape == (len(hier.labels) - 1, 4)


def test_kmeans_is_deterministic_given_the_seed_and_rejects_a_bad_k():
    returns, _, _ = simulate_groups()
    features = pca_features(returns, n_components=3)
    pd.testing.assert_series_equal(
        kmeans_clusters(features.embeddings, k=3, seed=0),
        kmeans_clusters(features.embeddings, k=3, seed=0),
    )
    with pytest.raises(ValueError, match="k must"):
        kmeans_clusters(features.embeddings, k=20)


def test_candidate_pairs_stay_inside_clusters():
    returns, _, groups = simulate_groups()
    labels = hierarchical_clusters(returns, n_clusters=3).labels
    pairs = candidate_pairs(labels, returns, n_per_cluster=2, min_obs=10)
    assert len(pairs) == 6
    assert pairs["correlation"].is_monotonic_decreasing
    for row in pairs.itertuples():
        assert row.a < row.b
        assert labels[row.a] == labels[row.b] == row.cluster
        assert groups[row.a] == groups[row.b]
    with pytest.raises(ValueError, match="n_per_cluster"):
        candidate_pairs(labels, returns, n_per_cluster=0)


def test_distance_misses_the_drifted_pair_that_clustering_keeps():
    """Same factor, different drift: high return correlation, large price distance."""
    returns, prices, _ = simulate_groups()
    same = returns["g0s0"].corr(returns["g0s1"])
    drifted = returns["g0s0"].corr(returns["g0s2"])
    assert same > 0.8 and drifted > 0.8

    ssd = distance.ssd_matrix(distance.normalise(prices))
    assert ssd.loc["g0s0", "g0s2"] > 3 * ssd.loc["g0s0", "g0s1"]

    top = distance.select_pairs(prices, n=6)
    closest = set(zip(top["a"], top["b"], strict=True))
    assert ("g0s0", "g0s1") in closest
    assert ("g0s0", "g0s2") not in closest

    labels = hierarchical_clusters(returns, n_clusters=3).labels
    kept = candidate_pairs(labels, returns, n_per_cluster=20, min_obs=10)
    hit = kept[(kept["a"] == "g0s0") & (kept["b"] == "g0s2")]
    assert len(hit) == 1
    assert hit.iloc[0]["cluster"] == labels["g0s0"]
