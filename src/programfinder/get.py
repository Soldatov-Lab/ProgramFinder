"""Accessors that return stored results as labelled pandas objects."""

from __future__ import annotations

import numpy as np

from .pp import residual_operator as operator  # noqa: F401  (re-exported)

__all__ = ["activities", "copy_number_table", "loadings", "operator", "pcs", "stability_table"]


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
    if "copy_number" in spec:
        table = table.join(copy_number_table(adata, key))
    return table


def copy_number_table(adata, key="pf_gica"):
    """One row per component: the four copy-number statistics, where each peaks, routes and call."""
    import pandas as pd

    spec = adata.uns[key]
    if "copy_number" not in spec:
        raise KeyError(f"no copy-number routes under {key!r}; run tl.copy_number_routes first")
    cn = spec["copy_number"]
    columns = ["chrom_share", "chrom_share_chrom", "chrom_share_pole", "run_z", "run_chrom",
               "run_start", "run_end", "run_sign", "opposite_arms_z", "opposite_arms_chrom",
               "top_arm", "top_arm_z", "second_chrom_arm", "second_chrom_arm_z"]
    table = pd.DataFrame({c: np.asarray(cn[c]) for c in columns})
    table.index = [f"c{k}" for k in range(len(table))]
    routes = cn["routes"]
    table["cn_routes"] = ["+".join(r for r in routes if routes[r][k]) for k in range(len(table))]
    table["copy_number"] = np.asarray(cn["copy_number"], bool)
    return table
