"""Accessors that return stored results as labelled pandas objects."""

from __future__ import annotations

import numpy as np

from .pp import residual_operator as operator  # noqa: F401  (re-exported)

__all__ = ["activities", "loadings", "operator", "pcs", "stability_table"]


def _frame(values, index, prefix):
    import pandas as pd

    values = np.asarray(values)
    columns = [f"{prefix}{k}" for k in range(values.shape[1])]
    return pd.DataFrame(values, index=index, columns=columns)


def activities(adata, key="pf_gica"):
    """Cells x components activities of ``tl.gica`` as a DataFrame."""
    return _frame(adata.obsm[f"X_{key}"], adata.obs_names, "c")


def loadings(adata, key="pf_gica"):
    """Features x components loadings of ``tl.gica`` as a DataFrame."""
    return _frame(adata.varm[f"{key}_loadings"], adata.var_names, "c")


def pcs(adata, key="pf_pca"):
    """Cells x components residual PCA scores as a DataFrame."""
    return _frame(adata.obsm[f"X_{key}"], adata.obs_names, "PC")


def stability_table(adata, key="pf_gica"):
    """One row per component with every reliability statistic that was computed."""
    import pandas as pd

    spec = adata.uns[key]
    if "stability" not in spec:
        raise KeyError(f"no stability results under {key!r}; run tl.gica_stability first")
    stab = spec["stability"]
    r = len(stab["effective_support"])
    table = pd.DataFrame(index=[f"c{k}" for k in range(r)])
    table["excess_kurtosis"] = np.asarray(spec["excess_kurtosis"])
    table["effective_support"] = np.asarray(stab["effective_support"])
    if "schedule" in stab:
        table["schedule_worst_abs_r"] = np.asarray(stab["schedule"]["worst_abs_r"])
    if "feature_bootstrap" in stab:
        table["bootstrap_worst_abs_r"] = np.asarray(stab["feature_bootstrap"]["worst_abs_r"])
    if "split_half" in stab:
        table["split_half_abs_cos_min"] = np.asarray(stab["split_half"]["abs_cos_min"])
    table["stable"] = np.asarray(stab["stable"], bool)
    if "dependence" in spec:
        dep = spec["dependence"]
        z = np.asarray(dep["zE"] if dep["statistic"] == "energy" else dep["zD"])
        top = np.asarray(dep["partners"])[:, 0]
        table["top_partner"] = [f"c{t}" if t >= 0 else "" for t in top]
        table["top_partner_z"] = [float(z[k, t]) if t >= 0 else np.nan for k, t in enumerate(top)]
    if "groups" in stab:
        grp = stab["groups"]
        table["group"] = ["" if g is None else "+".join(f"c{m}" for m in g) for g in grp["group_of"]]
        table["group_worst_min_cancorr"] = np.asarray(grp["group_worst"], float)
        table["group_refits_below_gate"] = np.asarray(grp["group_below_gate"], int)
        table["verdict"] = list(grp["verdict"])
    return table
