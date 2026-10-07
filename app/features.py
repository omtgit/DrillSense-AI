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
