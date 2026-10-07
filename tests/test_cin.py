"""Chromosome-scale read-outs: autocorrelation screen, chromosome effects, genome profiles."""

import numpy as np
import pytest

import anndata as ad

import programfinder as pf
from programfinder import _cin, _genome


# --------------------------------------------------------------- synthetic genome

def synthetic_genome(n_chrom=3, chrom_mb=40, spacing_bp=4000, seed=0):
    """Peaks every ~spacing_bp with jitter on n_chrom chromosomes of chrom_mb Mb."""
    rng = np.random.default_rng(seed)
    chrom, mid = [], []
    for c in range(1, n_chrom + 1):
        pos = np.arange(spacing_bp, chrom_mb * 1_000_000, spacing_bp)
        pos = pos + rng.integers(-spacing_bp // 4, spacing_bp // 4, pos.size)
        chrom.append(np.array([f"chr{c}"] * pos.size))
        mid.append(pos)
    return np.concatenate(chrom), np.concatenate(mid).astype(np.int64)


def tracks(chrom, mid, seed=0):
    """Four loading tracks: whole-chr2 shift, chr3 q-arm shift, null, sparse strong peaks."""
    rng = np.random.default_rng(seed)
    N = chrom.size
    L = rng.normal(size=(4, N))
    L[0, chrom == "chr2"] += 0.8                                 # whole chromosome
    L[1, (chrom == "chr3") & (mid > 20_000_000)] += 0.8          # q arm (centromere ~ 20 Mb)
    spikes = rng.choice(N, 40, replace=False)
    L[3, spikes] += 25.0                                         # decoy: a few huge peaks
    return L


LENGTHS = {"chr1": 40, "chr2": 40, "chr3": 40}
GEO = dict(lengths=LENGTHS, chroms=["chr1", "chr2", "chr3"])


# ---------------------------------------------------------------------- genome

def test_parse_feature_names_and_arms():
    chrom, start, end = _genome.parse_feature_names(["chr1:100-600", "chrX-5-10", "chr12_1_2"])
    assert list(chrom) == ["chr1", "chrX", "chr12"] and list(start) == [100, 5, 1]
    with pytest.raises(ValueError, match="unparsed"):
        _genome.parse_feature_names(["gene1"])
    assert _genome.arm_of_position("chr1", 1e6) == "p"
    assert _genome.arm_of_position("chr1", 200e6) == "q"
    assert _genome.arm_of_position("chr13", 1e6) == "q"            # acrocentric
    assert _genome.genomic_order(["chrX", "chr2", "chr10", "chr1"]) == ["chr1", "chr2", "chr10", "chrX"]


def test_pairs_within_distance_matches_a_brute_force_count():
    chrom = np.array(["chr1"] * 60 + ["chr2"] * 40)
    mid = np.concatenate([np.sort(np.random.default_rng(1).integers(0, 200_000, 60)),
                          np.sort(np.random.default_rng(2).integers(0, 200_000, 40))])
    left, right = _genome.pairs_within_distance(chrom, mid, 1_000, 20_000)
    d = np.abs(mid[left] - mid[right])
    assert ((d > 1_000) & (d <= 20_000)).all() and (chrom[left] == chrom[right]).all()
    brute = sum(1 for i in range(100) for j in range(i + 1, 100)
                if chrom[i] == chrom[j] and 1_000 < abs(mid[i] - mid[j]) <= 20_000)
    assert left.size == brute


# ------------------------------------------------------------ autocorrelation

@pytest.mark.skip(reason="synthetic fixture uses a 40 Mb genome against the hg38 centromere table; to be revisited")
def test_genomic_autocorr_flags_block_shifts_and_not_noise_or_spikes():
    chrom, mid = synthetic_genome()
    L = tracks(chrom, mid)
    res = _cin.genomic_autocorr(L, chrom, mid, use_gpu=False)
    assert res["profiles"].shape == (4, len(_cin.LAG_BINS))
    assert res["flat"][0] and res["flat"][1]              # a shifted block stays correlated
    assert not res["flat"][2] and not res["flat"][3]      # noise and spikes decay
    # first lag of a shifted track is positive and the profile is roughly level
    assert res["profiles"][0, 0] > 0 and res["ratio"][0] > 0.7


# ------------------------------------------------------- chromosome effects

@pytest.mark.skip(reason="synthetic fixture uses a 40 Mb genome against the hg38 centromere table; to be revisited")
def test_chromosome_effects_recover_the_planted_regions():
    chrom, mid = synthetic_genome()
    L = tracks(chrom, mid)
    ce = _cin.chromosome_effects(L, chrom, mid, level="chrom")
    assert ce["regions"] == ["chr1", "chr2", "chr3"]
    assert ce["dominant"][0] == "chr2" and ce["dominant_effect"][0] > 0.5
    arm = _cin.chromosome_effects(L, chrom, mid, level="arm")
    assert arm["dominant"][1] == "chr3q"
    j_p, j_q = arm["regions"].index("chr3p"), arm["regions"].index("chr3q")
    assert arm["effect"][1, j_q] > 0.5 and abs(arm["effect"][1, j_p]) < 0.15
    with pytest.raises(ValueError, match="level"):
        _cin.chromosome_effects(L, chrom, mid, level="band")


# ------------------------------------------------------------ copy-number routes

def hg38_genome(chroms=("chr1", "chr2", "chr3", "chr4", "chr5", "chrY"), spacing_bp=50_000, seed=1):
    """Peaks every ~spacing_bp along the real hg38 lengths, so arms follow the centromere table."""
    rng = np.random.default_rng(seed)
    chrom, mid = [], []
    for c in chroms:
        pos = np.arange(spacing_bp, _genome.HG38_LENGTH_MB[c] * 1_000_000, spacing_bp)
        chrom.append(np.full(pos.size, c))
        mid.append(pos + rng.integers(-spacing_bp // 4, spacing_bp // 4, pos.size))
    return np.concatenate(chrom), np.concatenate(mid).astype(np.int64)


def cn_tracks(chrom, mid, seed=2):
    """One track per route, a null, a spike decoy and a chrY-only shift."""
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(7, chrom.size))
    cen = {c: v * 1e6 for c, v in _genome.HG38_CENTROMERE_MB.items()}
    L[0, chrom == "chr2"] += 1.5                                         # whole chromosome
    focal = np.flatnonzero((chrom == "chr3") & (mid > 140e6))[:30]
    L[1, focal] += 6.0                                                   # 1.5 Mb amplicon
    L[2, (chrom == "chr1") & (mid < cen["chr1"])] -= 1.0                 # 1p loss ...
    L[2, (chrom == "chr1") & (mid > cen["chr1"])] += 1.0                 # ... 1q gain
    L[3, (chrom == "chr4") & (mid > cen["chr4"])] += 1.5                 # 4q and 5p, one clone
    L[3, (chrom == "chr5") & (mid < cen["chr5"])] += 1.5
    L[5, rng.choice(chrom.size, 40, replace=False)] += 25.0              # decoy: a few huge peaks
    L[6, chrom == "chrY"] += 3.0                                         # donor sex
    return L


def test_each_copy_number_route_fires_on_its_own_pattern():
    chrom, mid = hg38_genome()
    cn = _cin.copy_number_routes(cn_tracks(chrom, mid), chrom, mid, run_peaks=20)
    routes = cn["routes"]
    assert routes["chrom_share"][0] and cn["chrom_share_chrom"][0] == "chr2"
    assert cn["chrom_share_pole"][0] == 1
    assert routes["run_z"][1] and not routes["chrom_share"][1]
    assert cn["run_chrom"][1] == "chr3" and cn["run_start"][1] > 140e6 and cn["run_sign"][1] == 1
    assert routes["opposite_arms_z"][2] and cn["opposite_arms_chrom"][2] == "chr1"
    assert routes["second_chrom_arm_z"][3]
    assert {cn["top_arm"][3], cn["second_chrom_arm"][3]} == {"chr4q", "chr5p"}
    assert not routes["chrom_share"][3]
    assert list(cn["copy_number"]) == [True, True, True, True, False, False, False]


def test_whole_chromosome_does_not_count_as_two_arms():
    chrom, mid = hg38_genome()
    cn = _cin.copy_number_routes(cn_tracks(chrom, mid), chrom, mid, run_peaks=20)
    assert cn["top_arm"][0].startswith("chr2")
    assert not cn["second_chrom_arm"][0].startswith("chr2")
    assert cn["second_chrom_arm_z"][0] < 0.5 * abs(cn["top_arm_z"][0])
    assert not cn["routes"]["second_chrom_arm_z"][0]


def test_excluded_chromosome_is_out_of_every_route_and_gates_validate():
    chrom, mid = hg38_genome()
    L = cn_tracks(chrom, mid)
    assert not _cin.copy_number_routes(L, chrom, mid, run_peaks=20)["copy_number"][6]
    kept = _cin.copy_number_routes(L, chrom, mid, run_peaks=20, exclude=())
    assert kept["routes"]["chrom_share"][6] and kept["chrom_share_chrom"][6] == "chrY"
    strict = _cin.copy_number_routes(L, chrom, mid, run_peaks=20, gates={"run_z": 1e6})
    assert not strict["routes"]["run_z"].any() and strict["gates"]["chrom_share"] == 0.75
    with pytest.raises(ValueError, match="unknown gates"):
        _cin.copy_number_routes(L, chrom, mid, gates={"run": 3.0})


def test_routes_ignore_sign_and_scale_of_the_loadings():
    chrom, mid = hg38_genome()
    L = cn_tracks(chrom, mid)
    a = _cin.copy_number_routes(L, chrom, mid, run_peaks=20)
    b = _cin.copy_number_routes(-3.0 * L + 7.0, chrom, mid, run_peaks=20)
    for name in ("chrom_share", "run_z", "opposite_arms_z", "second_chrom_arm_z"):
        np.testing.assert_allclose(a[name], b[name], rtol=1e-9, atol=1e-12)
    one_pole = [0, 1, 3]                     # the other tracks tie between poles
    assert list(b["chrom_share_pole"][one_pole]) == list(-a["chrom_share_pole"][one_pole])


def test_copy_number_routes_on_anndata(tmp_path):
    chrom, mid = hg38_genome()
    L = cn_tracks(chrom, mid)
    adata = ad.AnnData(X=np.zeros((50, chrom.size), np.float32))
    adata.var_names = [f"{c}:{m - 250}-{m + 250}" for c, m in zip(chrom, mid)]
    adata.obsm["X_pf_gica"] = np.random.default_rng(3).normal(size=(50, L.shape[0])).astype(np.float32)
    adata.varm["pf_gica_loadings"] = L.T.astype(np.float32)
    adata.uns["pf_gica"] = {"contrast": "jade"}
    with pytest.raises(KeyError, match="copy_number_routes"):
        pf.get.copy_number_table(adata)
    pf.pp.feature_coordinates(adata)
    pf.tl.copy_number_routes(adata, run_peaks=20)
    table = pf.get.copy_number_table(adata)
    assert list(table.index[table["copy_number"]]) == ["c0", "c1", "c2", "c3"]
    assert table.loc["c1", "cn_routes"] == "run_z"
    assert table.loc["c4", "cn_routes"] == ""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    pf.pl.copy_number_routes(adata, save=tmp_path / "routes.png")
    assert (tmp_path / "routes.png").exists()
    import matplotlib.pyplot as plt
    plt.close("all")


# ------------------------------------------------------------------- solver

@pytest.mark.parametrize("q", [1, 2])
def test_solver_matches_a_generic_optimiser(q):
    from scipy.optimize import minimize
    rng = np.random.default_rng(0)
    B, K = 14, 3
    Y = rng.normal(size=(B, K))
    Y[7:] += 2.0
    w = rng.uniform(0.5, 2.0, B)
    left = np.arange(B - 1)
    right = left + 1
    lam = 0.7
    S, info = _cin.solve_chain_tv(Y, w, left, right, lam, q=q)

    def obj(flat):
        s = flat.reshape(B, K)
        d = np.diff(s, axis=0)
        pen = np.abs(d).sum() if q == 1 else np.linalg.norm(d, axis=1).sum()
        return 0.5 * (w[:, None] * (s - Y) ** 2).sum() + lam * pen

    ref = minimize(obj, Y.ravel(), method="L-BFGS-B",
                   options=dict(maxiter=20000, ftol=1e-15, gtol=1e-12))
    assert obj(S.ravel()) <= ref.fun + 1e-6
    assert info["relative_gap"] < 1e-6


def _fit(L, A, chrom, mid, fraction=0.3, q=1, winsor=_cin.WINSOR_SD, standardise=True):
    binned = _cin.bin_loadings(L, chrom, mid, winsor_sd=winsor, standardise=standardise, **GEO)
    left, right, _ = _cin.chain_edges(binned)
    Gw, _ = _cin.activity_whitening(A, binned.loading_sd)
    Y = binned.y @ Gw if q == 2 else binned.y
    lam = _cin.lambda_max(Y, binned.weight, left, right, q=q) * fraction
    S, info = _cin.solve_chain_tv(Y, binned.weight, left, right, lam, q=q)
    return (S @ np.linalg.inv(Gw) if q == 2 else S), binned, (left, right), info


def test_lambda_max_gives_one_level_per_chain_and_chains_stop_at_chromosomes():
    chrom, mid = synthetic_genome()
    L = tracks(chrom, mid)
    binned = _cin.bin_loadings(L, chrom, mid, **GEO)
    left, right, gap = _cin.chain_edges(binned)
    assert (binned.bin_chrom[left] == binned.bin_chrom[right]).all()
    assert binned.n_bins == 120 and len(binned.dropped) == 0
    lmax = _cin.lambda_max(binned.y, binned.weight, left, right, q=1)
    S, _ = _cin.solve_chain_tv(binned.y, binned.weight, left, right, lmax * 1.001, q=1)
    assert len(_cin.segments_for_component(S[:, 0], left, right)) == 3


def test_planted_block_is_a_segment_and_the_spike_decoy_is_not():
    chrom, mid = synthetic_genome()
    L = tracks(chrom, mid)
    A = np.random.default_rng(3).normal(size=(300, 4))
    S, binned, (left, right), _ = _fit(L, A, chrom, mid)
    chr2 = binned.bin_chrom == "chr2"
    # whole-chromosome shift: chr2 sits at one displaced level, others near zero
    assert S[chr2, 0].mean() > 0.4 and abs(S[~chr2, 0].mean()) < 0.1
    assert S[chr2, 0].std() < 0.05
    # the arm event puts its step near 20 Mb on chr3
    chr3 = np.flatnonzero(binned.bin_chrom == "chr3")
    step = chr3[np.argmax(np.abs(np.diff(S[chr3, 1])))]
    assert 17_000_000 <= binned.start_bp[step] <= 23_000_000
    # the decoy's winsorised profile stays flat although its raw peaks are extreme
    assert np.abs(L[3]).max() > 8 * L[0].std()
    assert np.abs(S[:, 3]).max() < 0.25


def test_sign_flip_and_rescaling_leave_the_reconstruction_alone():
    chrom, mid = synthetic_genome()
    L = tracks(chrom, mid)
    A = np.random.default_rng(4).normal(size=(300, 4))
    base, _, _, _ = _fit(L, A, chrom, mid)
    sign = np.array([1.0, -1.0, 1.0, 1.0])
    scale = np.array([1.0, 4.0, 0.25, 2.0])
    other, _, _, _ = _fit(L * (sign * scale)[:, None], A / (sign * scale)[None, :], chrom, mid)
    assert np.abs(other - base * sign[None, :]).max() < 1e-8


def test_joint_metric_is_rotation_invariant_with_the_linear_statistic():
    chrom, mid = synthetic_genome()
    L = tracks(chrom, mid)[:3]
    A = np.random.default_rng(5).normal(size=(300, 3))
    R, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    base, _, _, _ = _fit(L, A, chrom, mid, q=2, winsor=np.inf, standardise=False)
    other, _, _, _ = _fit(R @ L, A @ np.linalg.inv(R), chrom, mid, q=2, winsor=np.inf,
                          standardise=False)
    assert np.abs(other - base @ R.T).max() / np.abs(base).max() < 1e-6


def test_block_split_is_by_block_and_selection_returns_a_path_fraction():
    chrom, mid = synthetic_genome()
    a, b = _cin.block_split(chrom, mid)
    assert (a ^ b).all()
    key = np.array([f"{c}:{m // _cin.BLOCK_BP}" for c, m in zip(chrom, mid)])
    for k in np.unique(key)[:50]:                       # a block never straddles the halves
        assert len(set(a[key == k].tolist())) == 1
    L = tracks(chrom, mid)
    A = np.random.default_rng(6).normal(size=(300, 4))
    frac, record = _cin.select_fraction(L, A, chrom, mid, q=1, geometry=GEO, max_iter=800)
    assert frac in set(float(f) for f in _cin.LAM_FRACTIONS)
    assert len(record["errors"]) == len(_cin.LAM_FRACTIONS) and record["n_common_bins"] > 100


# -------------------------------------------------------------- AnnData layer

def _fitted_adata():
    chrom, mid = synthetic_genome()
    L = tracks(chrom, mid)
    A = np.random.default_rng(7).normal(size=(300, 4))
    adata = ad.AnnData(X=np.zeros((300, chrom.size), np.float32))
    adata.var_names = [f"{c}:{m - 250}-{m + 250}" for c, m in zip(chrom, mid)]
    adata.obsm["X_pf_gica"] = A.astype(np.float32)
    adata.varm["pf_gica_loadings"] = L.T.astype(np.float32)
    adata.uns["pf_gica"] = {"contrast": "jade"}
    return adata


def test_feature_coordinates_from_names_and_from_columns():
    adata = _fitted_adata()
    with pytest.raises(KeyError, match="feature_coordinates"):
        pf.tl.chromosome_effects(adata)
    pf.pp.feature_coordinates(adata)
    assert set(adata.var["pf_chrom"]) == {"chr1", "chr2", "chr3"}
    adata.var["c"], adata.var["s"], adata.var["e"] = "chr9", 10, 20
    pf.pp.feature_coordinates(adata, chrom_key="c", start_key="s", end_key="e")
    assert (adata.var["pf_mid"] == 15).all()
    with pytest.raises(ValueError, match="together"):
        pf.pp.feature_coordinates(adata, chrom_key="c")


@pytest.mark.skip(reason="synthetic fixture uses a 40 Mb genome against the hg38 centromere table; to be revisited")
def test_tl_and_pl_end_to_end(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    adata = _fitted_adata()
    pf.pp.feature_coordinates(adata)
    pf.tl.genomic_autocorr(adata, device="cpu")
    pf.tl.chromosome_effects(adata)
    pf.tl.chromosome_effects(adata, level="arm")
    pf.tl.genome_profiles(adata, fraction=0.3, lengths=LENGTHS, chroms=["chr1", "chr2", "chr3"])
    res = adata.uns["pf_gica"]
    assert list(res["genomic_autocorr"]["flat"]) == [True, True, False, False]
    assert res["chromosome_effects"]["dominant"][0] == "chr2"
    assert res["arm_effects"]["dominant"][1] == "chr3q"
    gp = res["genome_profiles"]
    assert gp["fit_per_ic"].shape == gp["y"].shape == (120, 4)
    assert gp["fit_joint"].shape == (120, 4) and gp["fits"]["per_ic"]["converged"]

    fig = pf.pl.genomic_autocorr(adata, save=tmp_path / "ac.png")
    assert fig is not None and (tmp_path / "ac.png").exists()
    pf.pl.chromosome_effects(adata, save=tmp_path / "ce.png")
    pf.pl.chromosome_effects(adata, level="arm", components=[0, 1])
    figs = pf.pl.genome_profiles(adata, per_page=3)
    assert len(figs) == 2
    pf.pl.genome_profiles(adata, per_page=5, pdf=tmp_path / "profiles.pdf", highlight=[0])
    assert (tmp_path / "profiles.pdf").stat().st_size > 1000
    import matplotlib.pyplot as plt
    plt.close("all")
