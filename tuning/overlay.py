#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import uproot
from ROOT import TLorentzVector

MUON_MASS_GEV = 0.105658

# =========================================================
#   MODE SELECTION: "jpsi", "psip", "dy", or "comb"
#
#   jpsi : ml_p_jpsi > 0.9
#   psip : ml_p_jpsi < 0.9  AND  ml_p_psip > 0.9
#   dy   : ml_p_dy > 0.9
#            AND ml_p_jpsi < 0.9  AND ml_p_psip < 0.9
#   comb : ml_p_dy < 0.9
#            AND ml_p_jpsi < 0.9  AND ml_p_psip < 0.9
# =========================================================
MODE = "comb"   # <-- change to "psip", "dy", or "comb" as needed

assert MODE in ("jpsi", "psip", "dy", "comb"), \
    f"MODE must be 'jpsi', 'psip', 'dy', or 'comb', got '{MODE}'"

ML_JPSI_CUT    = 0.8
ML_PSIP_CUT    = 0.8
ML_DY_COMB_CUT = 0.8

# Kinematic cuts (both EXP and SIM)
_MASS_RANGES = {
    "jpsi": (2.2, 4.2),
    "psip": (2.8, 4.8),
    "dy":   (1.5, 6.0),
    "comb": (1.5, 6.0),
}
MASS_MIN, MASS_MAX = _MASS_RANGES[MODE]
VZ_MIN = -600.0

MOMENTUM_RANGES = {
    'rec_dimu_mu_pos_px': (-3.2, 0.5),
    'rec_dimu_mu_pos_py': (-2.5, 2.5),
    'rec_dimu_mu_pos_pz': (20.0, 70.0),
    'rec_dimu_mu_neg_px': (-0.5, 3.2),
    'rec_dimu_mu_neg_py': (-2.5, 2.5),
    'rec_dimu_mu_neg_pz': (20.0, 70.0)
}

# Preferred histogram ranges for all plotted variables
FIXED_RANGES = {
    "mass":                 (1.5, 6.0),
    "rec_dimu_pt":          (0.0, 3.0),
    "rec_dimu_y":           (2.5, 5.0),
    "rec_dimu_eta":         (3.0, 7.),
    "rec_dimu_E":           (0.0, 120.0),
    "rec_dimu_pz":          (0.0, 120.0),
    "rec_dimu_mT":          (2.0, 6.0),
    "rec_mu_open_angle":    (0.0, 0.15),
    "rec_mu_theta_pos":     (-0.15, 0.05),
    "rec_mu_theta_neg":     (-0.05, 0.15),
    "rec_mu_dpt":           (-4.0, 4.0),
    "rec_mu_deltaR":        (1.0, 4.2),
    "rec_dimu_mu_pos_px":   (-3.2, 0.5),
    "rec_dimu_mu_pos_py":   (-2.5, 2.5),
    "rec_dimu_mu_pos_pz":   (20.0, 70.0),
    "rec_dimu_mu_neg_px":   (-0.5, 3.2),
    "rec_dimu_mu_neg_py":   (-2.5, 2.5),
    "rec_dimu_mu_neg_pz":   (20.0, 70.0),
    "rec_track_pos_x_st1":  (-50.0, 30.0),
    "rec_track_neg_x_st1":  (-30.0, 50.0),
    "rec_track_pos_px_st1": (-0.5, 3.0),
    "rec_track_neg_px_st1": (-3.0, 0.5),
    "rec_track_pos_vz":     (-600.0, 600.0),
    "rec_track_neg_vz":     (-600.0, 600.0),
    "rec_dz_vtx":           (-300.0, 300.0),
}

# EXP cut variables — ML scores and kinematics
CUT_VARS_EXP = [
    "ml_p_jpsi",
    "ml_p_psip",
    "ml_p_dy",
    "mass",
    "rec_track_neg_vz",
    "rec_track_pos_vz",
]

CUT_VARS_SIM = [
    "mass",
    "rec_track_neg_vz",
    "rec_track_pos_vz",
]

MOMENTUM_BRANCHES = [
    "rec_dimu_mu_pos_px", "rec_dimu_mu_pos_py", "rec_dimu_mu_pos_pz",
    "rec_dimu_mu_neg_px", "rec_dimu_mu_neg_py", "rec_dimu_mu_neg_pz",
]

STATION1_BRANCHES = [
    "rec_track_pos_x_st1", "rec_track_neg_x_st1",
    "rec_track_pos_px_st1", "rec_track_neg_px_st1",
]


# =========================================================
#                      Cut functions
# =========================================================
def _exp_cut_description() -> str:
    if MODE == "jpsi":
        return (f"jpsi: ML_JPSI > {ML_JPSI_CUT}, "
                f"mass [{MASS_MIN}, {MASS_MAX}], vz > {VZ_MIN}")
    if MODE == "psip":
        return (f"psip: ML_JPSI < {ML_JPSI_CUT} && ML_PSIP > {ML_PSIP_CUT}, "
                f"mass [{MASS_MIN}, {MASS_MAX}], vz > {VZ_MIN}")
    if MODE == "dy":
        return (f"dy: ML_DY_COMB > {ML_DY_COMB_CUT} "
                f"&& ML_JPSI < {ML_JPSI_CUT} && ML_PSIP < {ML_PSIP_CUT}, "
                f"mass [{MASS_MIN}, {MASS_MAX}], vz > {VZ_MIN}")
    # comb
    return (f"comb: ML_DY_COMB < {ML_DY_COMB_CUT} "
            f"&& ML_JPSI < {ML_JPSI_CUT} && ML_PSIP < {ML_PSIP_CUT}, "
            f"mass [{MASS_MIN}, {MASS_MAX}], vz > {VZ_MIN}")


def apply_basic_cuts_exp(arrays: Dict) -> np.ndarray:

    if MODE == "jpsi":
        ml_mask = (arrays["ml_p_jpsi"] > ML_JPSI_CUT)
    elif MODE == "psip":
        ml_mask = (
            (arrays["ml_p_jpsi"] < ML_JPSI_CUT) &
            (arrays["ml_p_psip"] > ML_PSIP_CUT)
        )
    elif MODE == "dy":
        ml_mask = (
            (arrays["ml_p_dy"]       > ML_DY_COMB_CUT) &
            (arrays["ml_p_jpsi"]  < ML_JPSI_CUT) &
            (arrays["ml_p_psip"]  < ML_PSIP_CUT)
        )
    else:  # comb
        ml_mask = (
            (arrays["ml_p_dy"]       < ML_DY_COMB_CUT) &
            (arrays["ml_p_jpsi"]  < ML_JPSI_CUT) &
            (arrays["ml_p_psip"]  < ML_PSIP_CUT)
        )

    mask = (
        ml_mask &
        (arrays["mass"] > MASS_MIN) &
        (arrays["mass"] < MASS_MAX) &
        (arrays["rec_track_neg_vz"] > VZ_MIN) &
        (arrays["rec_track_pos_vz"] > VZ_MIN) &
        (np.abs(arrays["rec_track_pos_vz"] - arrays["rec_track_neg_vz"]) < 200)
    )

    for var, (min_val, max_val) in MOMENTUM_RANGES.items():
        mask &= (arrays[var] > min_val) & (arrays[var] < max_val)

    return mask


def apply_basic_cuts_sim(arrays: Dict) -> np.ndarray:
    """
    Kinematic-only cuts for SIM data — no ML variables.
    """
    mask = (
        (arrays["mass"] > MASS_MIN) &
        (arrays["mass"] < MASS_MAX) &
        (arrays["rec_track_neg_vz"] > VZ_MIN) &
        (arrays["rec_track_pos_vz"] > VZ_MIN) &
        (np.abs(arrays["rec_track_pos_vz"] - arrays["rec_track_neg_vz"]) < 200)
    )

    for var, (min_val, max_val) in MOMENTUM_RANGES.items():
        mask &= (arrays[var] > min_val) & (arrays[var] < max_val)

    return mask


def apply_momentum_cuts(features: Dict) -> np.ndarray:
    n = len(features.get("mass", np.array([])))
    if n == 0:
        return np.array([], dtype=bool)

    mask = np.ones(n, dtype=bool)
    for var, (lo, hi) in MOMENTUM_RANGES.items():
        if var not in features:
            continue
        vals = features[var]
        cut = (vals >= lo) & (vals <= hi)
        rejected = n - cut.sum()
        if rejected > 0:
            print(f"  → {var:22s} rejects {rejected:6,d} / {n:,} ({rejected/n:.1%})")
        mask &= cut
    return mask


def delta_phi(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def compute_derived_features(arrays: Dict) -> Dict:
    n = len(arrays["rec_dimu_mu_pos_px"])
    features = {}

    mu_pos = TLorentzVector()
    mu_neg = TLorentzVector()

    keys = [
        "rec_dimu_y", "rec_dimu_eta", "rec_dimu_E", "rec_dimu_pz", "rec_dimu_pt",
        "mass", "rec_dimu_mT", "rec_mu_theta_pos", "rec_mu_theta_neg",
        "rec_mu_open_angle", "rec_mu_dpt", "rec_mu_deltaR", "rec_dz_vtx"
    ]
    for k in keys:
        features[k] = np.zeros(n)

    for i in range(n):
        pxp = float(arrays["rec_dimu_mu_pos_px"][i])
        pyp = float(arrays["rec_dimu_mu_pos_py"][i])
        pzp = float(arrays["rec_dimu_mu_pos_pz"][i])
        pxn = float(arrays["rec_dimu_mu_neg_px"][i])
        pyn = float(arrays["rec_dimu_mu_neg_py"][i])
        pzn = float(arrays["rec_dimu_mu_neg_pz"][i])

        mu_pos.SetXYZM(pxp, pyp, pzp, MUON_MASS_GEV)
        mu_neg.SetXYZM(pxn, pyn, pzn, MUON_MASS_GEV)
        dimu = mu_pos + mu_neg

        features["rec_dimu_y"][i]         = dimu.Rapidity()
        features["rec_dimu_eta"][i]       = dimu.Eta()
        features["rec_dimu_E"][i]         = dimu.E()
        features["rec_dimu_pz"][i]        = dimu.Pz()
        features["rec_dimu_pt"][i]        = dimu.Pt()
        features["mass"][i]               = dimu.M()
        features["rec_dimu_mT"][i]        = np.sqrt(dimu.M()**2 + dimu.Pt()**2)

        features["rec_mu_theta_pos"][i]   = mu_pos.Px()/mu_pos.Pz()
        features["rec_mu_theta_neg"][i]   = mu_neg.Px()/mu_neg.Pz()
        features["rec_mu_dpt"][i]         = mu_pos.Pt() - mu_neg.Pt()

        vpos  = mu_pos.Vect()
        vneg  = mu_neg.Vect()
        denom = vpos.Mag() * vneg.Mag()
        cos_open = vpos.Dot(vneg) / denom if denom > 1e-9 else 1.0
        features["rec_mu_open_angle"][i]  = np.arccos(np.clip(cos_open, -1.0, 1.0))

        d_eta = mu_pos.Eta() - mu_neg.Eta()
        d_phi = delta_phi(mu_pos.Phi(), mu_neg.Phi())
        features["rec_mu_deltaR"][i]      = np.sqrt(d_eta**2 + d_phi**2)

        features["rec_dz_vtx"][i] = (
            arrays["rec_track_pos_vz"][i] - arrays["rec_track_neg_vz"][i]
        )

    # Pass-through raw branches
    for br in MOMENTUM_BRANCHES + STATION1_BRANCHES + ["rec_track_pos_vz", "rec_track_neg_vz"]:
        if br in arrays:
            features[br] = arrays[br]

    return features


# =========================================================
#                    Data loading
# =========================================================
def load_and_cut_data(
    file_path: Path,
    tree_name: str,
    is_exp: bool,
    max_events: Optional[int] = None,
) -> Dict:
    label = "EXP" if is_exp else "SIM"
    print(f"\nLoading {label} from: {file_path}")

    with uproot.open(file_path) as f:
        tree = f[tree_name]
        needed = list(set(
            (CUT_VARS_EXP if is_exp else CUT_VARS_SIM) +
            MOMENTUM_BRANCHES + STATION1_BRANCHES +
            ["rec_track_pos_vz", "rec_track_neg_vz"]
        ))
        arrays = tree.arrays(needed, library="np", entry_stop=max_events)

    n_total = len(arrays.get("mass", np.array([])))
    print(f"  Loaded {n_total:,} events")

    mask    = apply_basic_cuts_exp(arrays) if is_exp else apply_basic_cuts_sim(arrays)
    n_basic = mask.sum()
    print(f"  After basic cuts: {n_basic:,} ({n_basic/n_total:.1%})")

    arrays   = {k: v[mask] for k, v in arrays.items()}
    features = compute_derived_features(arrays)

    mom_mask = apply_momentum_cuts(features)
    n_final  = mom_mask.sum()
    print(f"  After momentum cuts: {n_final:,} ({n_final/n_basic:.1%} of basic)")

    return {k: v[mom_mask] for k, v in features.items()}


# =========================================================
#                    Plotting helpers
# =========================================================
def make_overlay_plot(
    exp_data: Dict,
    sim_data: Dict,
    sim2_data: Optional[Dict],
    variable: str,
    ax,
    bins: int = 40,
    xlabel: Optional[str] = None,
    sim_label: str = "SIM (muon gun)",
    sim2_label: str = "SIM (QCD+diffraction)",
):
    missing = variable not in exp_data or variable not in sim_data
    if missing:
        ax.text(0.5, 0.5, f"{variable}\nmissing", ha="center", va="center")
        ax.set_axis_off()
        return

    exp_vals  = exp_data[variable]
    sim_vals  = sim_data[variable]
    sim2_vals = sim2_data[variable] if (sim2_data and variable in sim2_data) else None

    range_tuple = FIXED_RANGES.get(variable)
    if range_tuple is None:
        candidates = [
            np.percentile(exp_vals, 0.5), np.percentile(sim_vals, 0.5),
            np.percentile(exp_vals, 99.5), np.percentile(sim_vals, 99.5),
        ]
        if sim2_vals is not None:
            candidates += [np.percentile(sim2_vals, 0.5), np.percentile(sim2_vals, 99.5)]
        lo, hi = min(candidates[:len(candidates)//2]), max(candidates[len(candidates)//2:])
    else:
        lo, hi = range_tuple

    hist_exp, edges = np.histogram(exp_vals, bins=bins, range=(lo, hi), density=True)
    hist_sim, _     = np.histogram(sim_vals, bins=edges, density=True)
    centers         = (edges[:-1] + edges[1:]) / 2

    binw  = edges[1] - edges[0]
    n_exp = len(exp_vals)
    err   = np.sqrt(hist_exp * n_exp * binw) / (n_exp * binw + 1e-12)

    # EXP: data points with error bars
    ax.errorbar(centers, hist_exp, yerr=err, fmt="o", ms=5, capsize=3,
                color="navy", elinewidth=1.2, alpha=0.9,
                label=f"EXP  (N={n_exp:,})")

    # SIM 1: solid step line
    ax.step(centers, hist_sim, where="mid", lw=2.2, color="darkred",
            label=f"{sim_label}  (N={len(sim_vals):,})")

    # SIM 2: dashed step line (different colour)
    if sim2_vals is not None:
        hist_sim2, _ = np.histogram(sim2_vals, bins=edges, density=True)
        ax.step(centers, hist_sim2, where="mid", lw=2.2, color="darkorange",
                linestyle="--", label=f"{sim2_label}  (N={len(sim2_vals):,})")

    ax.set_xlabel(xlabel or variable.replace("_", " ").title(), fontsize=20)
    ax.set_ylabel("Normalized", fontsize=20)
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.2, ls="--")


def plot_all_features(
    exp_data: Dict,
    sim_data: Dict,
    sim2_data: Optional[Dict],
    output_dir: Path,
    sim_label: str = "SIM (muon gun)",
    sim2_label: str = "SIM (QCD+diffraction)",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {"var": "mass",                 "bins": 50, "xlabel": "Dimuon mass [GeV/c²]"},
        {"var": "rec_dimu_pt",          "bins": 40, "xlabel": "Dimuon p_T [GeV/c]"},
        {"var": "rec_dimu_y",           "bins": 40, "xlabel": "Dimuon rapidity"},
        {"var": "rec_dimu_eta",         "bins": 40, "xlabel": r"Dimuon $\eta$"},
        {"var": "rec_dimu_E",           "bins": 50, "xlabel": "Dimuon energy [GeV]"},
        {"var": "rec_dimu_pz",          "bins": 50, "xlabel": "Dimuon p_z [GeV/c]"},
        {"var": "rec_dimu_mT",          "bins": 40, "xlabel": "Dimuon m_T [GeV/c²]"},
        {"var": "rec_mu_open_angle",    "bins": 40, "xlabel": "Opening angle [rad]"},
        {"var": "rec_mu_theta_pos",     "bins": 40, "xlabel": r"$\mu^+$ $\theta$ [rad]"},
        {"var": "rec_mu_theta_neg",     "bins": 40, "xlabel": r"$\mu^-$ $\theta$ [rad]"},
        {"var": "rec_mu_dpt",           "bins": 40, "xlabel": r"$\Delta p_T$ [GeV/c]"},
        {"var": "rec_mu_deltaR",        "bins": 40, "xlabel": r"$\Delta R(\mu^+,\mu^-)$"},
        {"var": "rec_dimu_mu_pos_px",   "bins": 50, "xlabel": r"$\mu^+$ p_x [GeV/c]"},
        {"var": "rec_dimu_mu_pos_py",   "bins": 50, "xlabel": r"$\mu^+$ p_y [GeV/c]"},
        {"var": "rec_dimu_mu_pos_pz",   "bins": 50, "xlabel": r"$\mu^+$ p_z [GeV/c]"},
        {"var": "rec_dimu_mu_neg_px",   "bins": 50, "xlabel": r"$\mu^-$ p_x [GeV/c]"},
        {"var": "rec_dimu_mu_neg_py",   "bins": 50, "xlabel": r"$\mu^-$ p_y [GeV/c]"},
        {"var": "rec_dimu_mu_neg_pz",   "bins": 50, "xlabel": r"$\mu^-$ p_z [GeV/c]"},
        {"var": "rec_track_pos_x_st1",  "bins": 50, "xlabel": r"$\mu^+$ x @ St1 [cm]"},
        {"var": "rec_track_neg_x_st1",  "bins": 50, "xlabel": r"$\mu^-$ x @ St1 [cm]"},
        {"var": "rec_track_pos_px_st1", "bins": 50, "xlabel": r"$\mu^+$ p_x @ St1 [GeV/c]"},
        {"var": "rec_track_neg_px_st1", "bins": 50, "xlabel": r"$\mu^-$ p_x @ St1 [GeV/c]"},
        {"var": "rec_track_pos_vz",     "bins": 60, "xlabel": r"$\mu^+$ v_z [cm]"},
        {"var": "rec_track_neg_vz",     "bins": 60, "xlabel": r"$\mu^-$ v_z [cm]"},
    ]

    n_cols = 5
    n_rows = int(np.ceil(len(configs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.8 * n_rows),
                             constrained_layout=True)
    fig.suptitle(
        f"EXP vs SIM overlay — {MODE.upper()} selection\n{_exp_cut_description()}",
        fontsize=15, y=1.01
    )
    axes = axes.flat

    for ax, cfg in zip(axes, configs):
        make_overlay_plot(
            exp_data, sim_data, sim2_data,
            cfg["var"], ax,
            bins=cfg["bins"], xlabel=cfg.get("xlabel"),
            sim_label=sim_label, sim2_label=sim2_label,
        )

    for ax in axes[len(configs):]:
        ax.set_axis_off()

    outpath = output_dir / f"overlay_{MODE}_fixed_ranges.png"
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    print(f"Saved combined plot → {outpath}")
    plt.close()


# =========================================================
#                         Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Overlay EXP vs reweighted SIM (jpsi / psip / dy / comb modes)"
    )
    parser.add_argument(
        "--exp-file", type=Path,
        default=Path("/Users/spin/spinquest-combinatoric-bkg/data/raw_input/exp_tagged_tgt_data_II.root"),
    )
    parser.add_argument(
        "--sim-file", type=Path,
        #default=Path(f"data/mc_comb_muon_gun_march19_tuned_I.root"),
        #default=Path(f"data/out_test_dy_momentum_adaptive.root"),
        default=Path(f"data/out_test_comb_momentum_adaptive.root"),
    )
    parser.add_argument(
        "--sim2-file", type=Path,
        default=Path("/Users/spin/CombBkgDNP/comb/data/out_test_comb.root"),
        help="Optional second SIM file (untuned, for comparison). Pass empty string to skip.",
    )
    parser.add_argument("--sim-label",  type=str, default="SIM (tuned)",
                        help="Legend label for the first SIM dataset.")
    parser.add_argument("--sim2-label", type=str, default="SIM (untuned)",
                        help="Legend label for the second SIM dataset.")
    parser.add_argument("--tree-name",   type=str,  default="tree")
    parser.add_argument("--output-dir",  type=Path, default=Path(f"overlay_plots_{MODE}"))
    parser.add_argument("--max-events",  type=int,  default=None, help="cap for quick testing")
    args = parser.parse_args()

    print(f"=== Overlay plot generation — MODE = '{MODE}' ===")
    print(f"  Cuts : {_exp_cut_description()}")
    print(f"  EXP  : {args.exp_file}")
    print(f"  SIM1 : {args.sim_file}  [{args.sim_label}]")
    print(f"  SIM2 : {args.sim2_file}  [{args.sim2_label}]")
    print(f"  Out  : {args.output_dir}\n")

    exp_data  = load_and_cut_data(args.exp_file,  args.tree_name, is_exp=True,  max_events=args.max_events)
    sim_data  = load_and_cut_data(args.sim_file,  args.tree_name, is_exp=False, max_events=args.max_events)

    sim2_data = None
    if args.sim2_file and str(args.sim2_file).strip():
        sim2_data = load_and_cut_data(args.sim2_file, args.tree_name, is_exp=False, max_events=args.max_events)

    plot_all_features(exp_data, sim_data, sim2_data, args.output_dir,
                      sim_label=args.sim_label, sim2_label=args.sim2_label)
    print("\nDone.")


if __name__ == "__main__":
    main()
