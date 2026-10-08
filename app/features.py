"""Generic feature step: applies whatever physics features the active profile defines, per well."""
import pandas as pd


def add_profile_features(df, profile):
    parts = []
    for _, g in df.sort_values(["well_id", "timestamp"]).groupby("well_id", sort=False):
        g = g.copy()
        for f in profile.physics_features:
            g[f.name] = f.fn(g)
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df.copy()


def add_context(df, columns):
    """Past-only rolling context per well: 15-min mean/std, deviation from a 180-min trailing
    mean, 5-min diff. Assumes 1-minute rows sorted by time within each well."""
    parts = []
    for _, g in df.groupby("well_id", sort=False):
        cols = {}
        for c in columns:
            s = g[c]
            cols[f"{c}__mean15"] = s.rolling(15, min_periods=5).mean()
            cols[f"{c}__std15"] = s.rolling(15, min_periods=5).std()
            cols[f"{c}__dev180"] = s - s.rolling(180, min_periods=30).mean()
            cols[f"{c}__d5"] = s.diff(5)
        parts.append(pd.DataFrame(cols, index=g.index))
    return pd.concat([df, pd.concat(parts)], axis=1)
