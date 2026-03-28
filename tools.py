"""
tools.py — Tool implementations for the combinatoric background tuning agent.
All paths are relative to the tuning/ subdirectory.
"""

import subprocess
import re
import json
import numpy as np
from pathlib import Path

TUNING_DIR   = Path(__file__).parent / "tuning"
TUNING_SCRIPT = TUNING_DIR / "tuning.py"
OVERLAY_SCRIPT = TUNING_DIR / "overlay.py"
OUTPUT_ROOT   = TUNING_DIR / "data" / "out_test_comb_momentum_adaptive.root"
CONDA_ENV     = "root_env"

# Low-mass region of interest for combinatoric background
LOW_MASS_MIN = 1.5
LOW_MASS_MAX = 2.5   # full low-mass window (training window matches this)


# ── parameter registry ────────────────────────────────────────────────────────
# Each entry: (python_type, min, max, description)
PARAM_REGISTRY = {
    "RATIO_CLIP_MAX":    (float, 1.0,  50.0,  "Normalisation constant M in acceptance prob"),
    "BDT_N_ESTIMATORS":  (int,   10,   500,   "Number of gradient-boosting trees for mass reweighting"),
    "BDT_LEARNING_RATE": (float, 0.01, 0.5,   "Learning rate for GBReweighter"),
    "MASS_WEIGHT_CAP":   (float, 1.0,  50.0,  "Hard cap on per-event BDT mass weights"),
    "EPOCHS_EXP":        (int,   200,  3000,  "Training epochs for EXP normalizing flow"),
    "EPOCHS_SIM":        (int,   200,  3000,  "Training epochs for SIM normalizing flow"),
    "ML_JPSI_CUT":       (float, 0.5,  0.99,  "ML cut: reject events with ml_p_jpsi > this"),
    "ML_PSIP_CUT":       (float, 0.5,  0.99,  "ML cut: reject events with ml_p_psip > this"),
    "ML_DY_COMB_CUT":    (float, 0.5,  0.99,  "ML cut: comb region ml_p_dy < this"),
}


def get_params() -> dict:
    """Read current parameter values from tuning.py."""
    text = TUNING_SCRIPT.read_text()
    result = {}
    for name, (typ, lo, hi, desc) in PARAM_REGISTRY.items():
        m = re.search(rf"^{name}\s*=\s*([^\s#\n]+)", text, re.MULTILINE)
        if m:
            raw = m.group(1).rstrip(",")
            try:
                result[name] = typ(raw.replace("_", ""))
            except ValueError:
                result[name] = raw
        else:
            result[name] = "NOT FOUND"
    return result


def set_param(name: str, value) -> dict:
    """Set a single parameter in tuning.py. Returns {'ok': True} or {'error': ...}."""
    if name not in PARAM_REGISTRY:
        return {"error": f"Unknown parameter '{name}'. Valid: {list(PARAM_REGISTRY.keys())}"}

    typ, lo, hi, desc = PARAM_REGISTRY[name]
    try:
        value = typ(value)
    except (ValueError, TypeError) as e:
        return {"error": f"Cannot cast {value!r} to {typ.__name__}: {e}"}

    if not (lo <= value <= hi):
        return {"error": f"{name}={value} is out of allowed range [{lo}, {hi}]"}

    text = TUNING_SCRIPT.read_text()
    new_text, n = re.subn(
        rf"^({name}\s*=\s*)([^\s#\n]+)(.*)",
        lambda m: f"{m.group(1)}{value}{m.group(3)}",
        text,
        flags=re.MULTILINE,
    )
    if n == 0:
        return {"error": f"Could not find '{name}' assignment in tuning.py"}

    TUNING_SCRIPT.write_text(new_text)
    return {"ok": True, "name": name, "new_value": value, "description": desc}


def run_tuning(timeout: int = 2400) -> dict:
    """
    Run tuning.py via conda root_env.
    Returns KS statistic, accepted event count, and tail of stdout.
    """
    cmd = ["conda", "run", "-n", CONDA_ENV, "python", str(TUNING_SCRIPT)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(TUNING_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout + proc.stderr
        tail   = "\n".join(stdout.splitlines()[-40:])

        ks_match = re.search(r"Mass KS\s*:\s*([\d.]+)", stdout)
        ks = float(ks_match.group(1)) if ks_match else None

        acc_match = re.search(r"Accepted\s*:\s*([\d,]+)", stdout)
        n_acc = int(acc_match.group(1).replace(",", "")) if acc_match else None

        rate_match = re.search(r"rate:\s*([\d.]+)%", stdout)
        rate = float(rate_match.group(1)) if rate_match else None

        if proc.returncode != 0:
            return {"error": f"tuning.py exited with code {proc.returncode}", "tail": tail}

        return {
            "ks_stat":        ks,
            "n_accepted":     n_acc,
            "acceptance_rate_pct": rate,
            "output_file":    str(OUTPUT_ROOT),
            "tail":           tail,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"tuning.py timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


def run_overlay(timeout: int = 300) -> dict:
    """
    Run overlay.py to generate comparison plots.
    Returns path to the output plot.
    """
    cmd = ["conda", "run", "-n", CONDA_ENV, "python", str(OVERLAY_SCRIPT)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(TUNING_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = proc.stdout + proc.stderr
        tail   = "\n".join(stdout.splitlines()[-20:])

        if proc.returncode != 0:
            return {"error": f"overlay.py exited with code {proc.returncode}", "tail": tail}

        plot_match = re.search(r"Saved combined plot.*?→\s*(.+\.png)", stdout)
        plot_path  = plot_match.group(1).strip() if plot_match else None

        return {"plot_path": plot_path, "tail": tail}
    except subprocess.TimeoutExpired:
        return {"error": f"overlay.py timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


def compute_low_mass_ks() -> dict:
    """
    Load the output ROOT file and EXP file, apply the same cuts as tuning.py,
    then compute the KS statistic *only* in the low-mass region [1.5, 2.5] GeV.
    This is the primary metric the agent should optimise.
    """
    try:
        import uproot
        from scipy.stats import ks_2samp

        EXP_FILE = "/Users/spin/spinquest-combinatoric-bkg/data/raw_input/exp_tagged_tgt_data_II.root"

        if not OUTPUT_ROOT.exists():
            return {"error": "Output ROOT file not found. Run tuning first."}

        params = get_params()
        ml_dy  = float(params.get("ML_DY_COMB_CUT", 0.8))
        ml_jp  = float(params.get("ML_JPSI_CUT", 0.8))
        ml_ps  = float(params.get("ML_PSIP_CUT", 0.8))

        with uproot.open(EXP_FILE) as f:
            exp_arr = f["tree"].arrays(
                ["mass", "ml_p_jpsi", "ml_p_psip", "ml_p_dy",
                 "rec_track_neg_vz", "rec_track_pos_vz"],
                library="np",
            )
        exp_mask = (
            (exp_arr["ml_p_dy"]   < ml_dy) &
            (exp_arr["ml_p_jpsi"] < ml_jp) &
            (exp_arr["ml_p_psip"] < ml_ps) &
            (exp_arr["mass"] > 1.5) &
            (exp_arr["mass"] < 6.0) &
            (exp_arr["rec_track_neg_vz"] > -600) &
            (exp_arr["rec_track_pos_vz"] > -600) &
            (np.abs(exp_arr["rec_track_pos_vz"] - exp_arr["rec_track_neg_vz"]) < 200)
        )
        exp_mass = exp_arr["mass"][exp_mask]

        with uproot.open(str(OUTPUT_ROOT)) as f:
            sim_arr = f["tree"].arrays(["mass"], library="np")
        sim_mass = sim_arr["mass"]

        ks_full, _ = ks_2samp(exp_mass, sim_mass)

        exp_low = exp_mass[(exp_mass >= LOW_MASS_MIN) & (exp_mass <= LOW_MASS_MAX)]
        sim_low = sim_mass[(sim_mass >= LOW_MASS_MIN) & (sim_mass <= LOW_MASS_MAX)]

        if len(exp_low) < 10 or len(sim_low) < 10:
            return {
                "error": "Too few events in low-mass region",
                "n_exp_low": int(len(exp_low)),
                "n_sim_low": int(len(sim_low)),
            }

        ks_low, _ = ks_2samp(exp_low, sim_low)

        return {
            "ks_full_range":      round(float(ks_full), 6),
            "ks_low_mass":        round(float(ks_low), 6),
            "low_mass_range_GeV": [LOW_MASS_MIN, LOW_MASS_MAX],
            "n_exp_low":          int(len(exp_low)),
            "n_sim_low":          int(len(sim_low)),
            "n_exp_total":        int(len(exp_mass)),
            "n_sim_total":        int(len(sim_mass)),
        }
    except Exception as e:
        return {"error": str(e)}
