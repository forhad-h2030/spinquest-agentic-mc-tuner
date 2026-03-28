#!/usr/bin/env python3
"""
rej.py — 6-D momentum NF reweighting + BDT mass-ratio adaptive rejection sampling
for SpinQuest combinatoric background tuning. Minimises KS(exp_mass, sim_mass).
"""
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import uproot
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from hep_ml import reweight as hep_reweight
from tqdm import tqdm
import ROOT

MODE = "comb"

assert MODE in ("jpsi", "psip", "dy", "comb"), \
    f"MODE must be 'jpsi', 'psip', 'dy', or 'comb', got '{MODE}'"

ACTIVE_VARS = [
    "rec_dimu_mu_pos_px", "rec_dimu_mu_pos_py", "rec_dimu_mu_pos_pz",
    "rec_dimu_mu_neg_px", "rec_dimu_mu_neg_py", "rec_dimu_mu_neg_pz",
]

SIM_FILES = {
    "jpsi": "data/jpsi/raw_mc_jpsi_target_pythia8.root",
    "psip": "data/psip/raw_mc_psip_target_pythia8.root",
    "dy":   "dy/dy.root",
    "comb": "/Users/spin/CombBkgDNP/comb/data/mc_comb_muon_gun_march19.root",
}
SIM_FILE  = SIM_FILES[MODE]
#EXP_FILE  = "/Users/spin/spinquest-combinatoric-bkg/data/raw_input/exp_tagged_tgt_data.root"
EXP_FILE  = "/Users/spin/spinquest-combinatoric-bkg/data/raw_input/exp_tagged_tgt_data_II.root"

INPUT_FILE  = SIM_FILE
OUTPUT_FILE = f"data/out_test_{MODE}_momentum_adaptive.root"
TREE_NAME   = "tree"

ML_JPSI_CUT    = 0.8
ML_PSIP_CUT    = 0.8
ML_DY_COMB_CUT = 0.8

VZ_MIN = -600.0

_MASS_RANGES = {
    "jpsi": (2.2, 4.2),
    "psip": (2.8, 4.8),
    "dy":   (1.5, 6.0),
    "comb": (1.5, 6.0),
}
MASS_MIN, MASS_MAX = _MASS_RANGES[MODE]

MOMENTUM_RANGES = {
    'rec_dimu_mu_pos_px': (-3.2, 0.5),
    'rec_dimu_mu_pos_py': (-2.5, 2.5),
    'rec_dimu_mu_pos_pz': (20.0, 70.0),
    'rec_dimu_mu_neg_px': (-0.5, 3.2),
    'rec_dimu_mu_neg_py': (-2.5, 2.5),
    'rec_dimu_mu_neg_pz': (20.0, 70.0),
}

CLIP_ZSCORE      = 15.0
BATCH_SIZE_TRAIN = 1000
EPOCHS_EXP       = 700
EPOCHS_SIM       = 700
LR               = 1e-4
BATCH_SIZE_RS    = 1_000

RATIO_CLIP_INIT      = 5.0
RATIO_CLIP_MIN       = 3.0
RATIO_CLIP_MAX_CAP   = 15.0
ADAPTIVE_MAX_ITER    = 5
ADAPTIVE_TOL         = 1e-4
BDT_N_ESTIMATORS     = 300
BDT_LEARNING_RATE    = 0.1
MASS_WEIGHT_CAP      = 10.0

ADD_TRAIN_NOISE   = False
TRAIN_NOISE_SIGMA = 0.01

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)



class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        in_dim   = dim // 2
        out_dim  = dim - in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * out_dim),
        )

    def forward(self, x):
        x1, x2  = x[:, : self.dim // 2], x[:, self.dim // 2 :]
        theta    = self.net(x1)
        s, t     = theta[:, : x2.shape[1]], theta[:, x2.shape[1] :]
        s        = torch.clamp(s, min=-7.0, max=7.0)
        y2       = x2 * torch.exp(s) + t
        y        = torch.cat([x1, y2], dim=-1)
        log_det  = s.sum(dim=-1)
        return y, log_det


class Permute(nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.register_buffer("perm", perm)

    def forward(self, x):
        return x[:, self.perm]


class FlowModel(nn.Module):
    def __init__(self, dim, num_layers=6, hidden_dim=64):
        super().__init__()
        modules = []
        for _ in range(num_layers):
            modules.append(AffineCoupling(dim, hidden_dim=hidden_dim))
            modules.append(Permute(torch.randperm(dim)))
        self.transforms = nn.ModuleList(modules)

    def forward_flow(self, x):
        z             = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for t in self.transforms:
            if isinstance(t, AffineCoupling):
                z, ld = t(z)
            else:
                z  = t(z)
                ld = torch.zeros_like(log_det_total)
            log_det_total += ld
        return z, log_det_total

    def log_prob(self, x, base_dist):
        z, log_det = self.forward_flow(x)
        return base_dist.log_prob(z) + log_det


def make_base_dist(dim, device):
    return D.Independent(
        D.Normal(torch.zeros(dim, device=device), torch.ones(dim, device=device)),
        1
    )


def train_flow(model, base_dist, data_tensor, epochs, batch_size, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n  = int(data_tensor.shape[0])
    bs = int(min(batch_size, n))

    for _ in tqdm(range(int(epochs)), desc="Training"):
        idx = torch.randint(0, n, (bs,), device=data_tensor.device)
        xb  = data_tensor[idx]
        if ADD_TRAIN_NOISE and TRAIN_NOISE_SIGMA > 0:
            xb = xb + TRAIN_NOISE_SIGMA * torch.randn_like(xb)
        loss = -model.log_prob(xb, base_dist).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()



class BDTMassWeighter:
    def __init__(self, exp_mass: np.ndarray):
        self._exp_mass  = np.clip(exp_mass, MASS_MIN, MASS_MAX).reshape(-1, 1)
        self._reweighter = None
        self.weight_cap  = MASS_WEIGHT_CAP

    def update(self, sim_mass: np.ndarray):
        """Retrain a GBReweighter on (sim_mass → exp_mass)."""
        if len(sim_mass) < 10:
            print("[BDTMassWeighter] Warning: fewer than 10 SIM mass events — "
                  "keeping previous BDT.")
            return
        sim_clipped = np.clip(sim_mass, MASS_MIN, MASS_MAX).reshape(-1, 1)

        n = min(len(sim_clipped), len(self._exp_mass))
        rng = np.random.default_rng(SEED)
        sim_idx = rng.choice(len(sim_clipped), n, replace=False)
        exp_idx = rng.choice(len(self._exp_mass), n, replace=False)

        self._reweighter = hep_reweight.GBReweighter(
            n_estimators=BDT_N_ESTIMATORS,
            learning_rate=BDT_LEARNING_RATE,
        )
        self._reweighter.fit(
            sim_clipped[sim_idx],
            self._exp_mass[exp_idx],
        )
        print(f"[BDTMassWeighter] BDT retrained on {n:,} events.")

    def weights(self, mass_array: np.ndarray) -> np.ndarray:
        """
        Return per-event BDT mass weights, capped and normalised.
        Returns ones if BDT not yet trained (first iteration).
        """
        if self._reweighter is None:
            return np.ones(len(mass_array), dtype=np.float32)

        m = np.clip(mass_array, MASS_MIN, MASS_MAX).reshape(-1, 1)
        w = self._reweighter.predict_weights(m)
        w = np.clip(w, 0.0, self.weight_cap)
        mean_w = w.mean()
        if mean_w > 1e-8:
            w = w / mean_w
        return w.astype(np.float32)



def apply_cuts_exp(data_dict):
    if MODE == "jpsi":
        ml_mask = (data_dict["ml_p_jpsi"] > ML_JPSI_CUT)
    elif MODE == "psip":
        ml_mask = (
            (data_dict["ml_p_jpsi"] < ML_JPSI_CUT) &
            (data_dict["ml_p_psip"] > ML_PSIP_CUT)
        )
    elif MODE == "dy":
        ml_mask = (
            (data_dict["ml_p_dy"]   > ML_DY_COMB_CUT) &
            (data_dict["ml_p_jpsi"] < ML_JPSI_CUT) &
            (data_dict["ml_p_psip"] < ML_PSIP_CUT)
        )
    else:  # comb
        ml_mask = (
            (data_dict["ml_p_dy"]   < ML_DY_COMB_CUT) &
            (data_dict["ml_p_jpsi"] < ML_JPSI_CUT) &
            (data_dict["ml_p_psip"] < ML_PSIP_CUT)
        )
    mask = (
        ml_mask &
        (data_dict["mass"] > MASS_MIN) &
        (data_dict["mass"] < MASS_MAX) &
        (data_dict["rec_track_neg_vz"] > VZ_MIN) &
        (data_dict["rec_track_pos_vz"] > VZ_MIN) &
        (np.abs(data_dict["rec_track_pos_vz"] - data_dict["rec_track_neg_vz"]) < 200)
    )
    for var, (lo, hi) in MOMENTUM_RANGES.items():
        mask &= (data_dict[var] > lo) & (data_dict[var] < hi)
    return mask


def apply_cuts_sim(data_dict):
    mask = (
        (data_dict["mass"] > MASS_MIN) &
        (data_dict["mass"] < MASS_MAX) &
        (data_dict["rec_track_neg_vz"] > VZ_MIN) &
        (data_dict["rec_track_pos_vz"] > VZ_MIN) &
        (np.abs(data_dict["rec_track_pos_vz"] - data_dict["rec_track_neg_vz"]) < 200)
    )
    for var, (lo, hi) in MOMENTUM_RANGES.items():
        mask &= (data_dict[var] > lo) & (data_dict[var] < hi)
    return mask


CUT_VARS = [
    "ml_p_jpsi", "ml_p_psip", "ml_p_dy",
    "mass", "rec_track_neg_vz", "rec_track_pos_vz",
] + list(MOMENTUM_RANGES.keys())

CUT_VARS_SIM = [
    "mass", "rec_track_neg_vz", "rec_track_pos_vz",
] + list(MOMENTUM_RANGES.keys())

ALL_VARS     = list(set(ACTIVE_VARS + CUT_VARS))
ALL_VARS_SIM = list(set(ACTIVE_VARS + CUT_VARS_SIM))

RS_VARS     = list(set(ALL_VARS     + ["mass"]))
RS_VARS_SIM = list(set(ALL_VARS_SIM + ["mass"]))


def count_entries(file_path, tree_name):
    with uproot.open(file_path) as f:
        return int(f[tree_name].num_entries)


def _extract_active(arrays, mask):
    cols = [arrays[v][mask] for v in ACTIVE_VARS]
    return np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)


def load_first_n_flat(file_path, tree_name, var_list, n_rows, cut_type='none'):
    with uproot.open(file_path) as f:
        tree   = f[tree_name]
        n_rows = int(min(n_rows, tree.num_entries))
        arrays = tree.arrays(var_list, library="np", entry_stop=n_rows)

    if cut_type == 'exp':
        mask          = apply_cuts_exp(arrays)
        sampling_data = _extract_active(arrays, mask)
        print(f"  Loaded {n_rows:,} events, {mask.sum():,} passed EXP cuts "
              f"({100*mask.sum()/n_rows:.1f}%)")
    elif cut_type == 'sim':
        mask          = apply_cuts_sim(arrays)
        sampling_data = _extract_active(arrays, mask)
        print(f"  Loaded {n_rows:,} events, {mask.sum():,} passed SIM cuts "
              f"({100*mask.sum()/n_rows:.1f}%)")
    else:
        sampling_data = _extract_active(arrays, np.ones(n_rows, dtype=bool))
        print(f"  Loaded {n_rows:,} events (no cuts)")
    return sampling_data


def load_random_window(file_path, tree_name, var_list, n_rows, seed=42, cut_type='none'):
    with uproot.open(file_path) as f:
        tree   = f[tree_name]
        total  = int(tree.num_entries)
        n_rows = int(min(n_rows, total))
        start  = 0 if total == n_rows else int(
            np.random.default_rng(seed).integers(0, total - n_rows + 1))
        arrays = tree.arrays(var_list, library="np",
                             entry_start=start, entry_stop=start + n_rows)

    if cut_type == 'exp':
        mask          = apply_cuts_exp(arrays)
        sampling_data = _extract_active(arrays, mask)
        print(f"  Loaded {n_rows:,} events, {mask.sum():,} passed EXP cuts "
              f"({100*mask.sum()/n_rows:.1f}%)")
    elif cut_type == 'sim':
        mask          = apply_cuts_sim(arrays)
        sampling_data = _extract_active(arrays, mask)
        print(f"  Loaded {n_rows:,} events, {mask.sum():,} passed SIM cuts "
              f"({100*mask.sum()/n_rows:.1f}%)")
    else:
        sampling_data = _extract_active(arrays, np.ones(n_rows, dtype=bool))
        print(f"  Loaded {n_rows:,} events (no cuts)")
    return sampling_data


def fit_scaler_on_exp_and_transform(exp_points, sim_points):
    scaler     = StandardScaler()
    exp_scaled = scaler.fit_transform(exp_points)
    sim_scaled = scaler.transform(sim_points)
    exp_scaled = np.clip(exp_scaled, -CLIP_ZSCORE, CLIP_ZSCORE)
    sim_scaled = np.clip(sim_scaled, -CLIP_ZSCORE, CLIP_ZSCORE)
    return exp_scaled, sim_scaled, scaler


def to_device_tensor(np_array):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(np_array, dtype=torch.float32, device=device), device


def _exp_cut_description():
    if MODE == "jpsi":
        return f"jpsi: ML_JPSI > {ML_JPSI_CUT}, mass [{MASS_MIN},{MASS_MAX}], vz > {VZ_MIN}"
    if MODE == "psip":
        return (f"psip: ML_JPSI < {ML_JPSI_CUT} && ML_PSIP > {ML_PSIP_CUT}, "
                f"mass [{MASS_MIN},{MASS_MAX}], vz > {VZ_MIN}")
    if MODE == "dy":
        return (f"dy: ML_DY > {ML_DY_COMB_CUT} && ML_JPSI < {ML_JPSI_CUT} "
                f"&& ML_PSIP < {ML_PSIP_CUT}, mass [{MASS_MIN},{MASS_MAX}], vz > {VZ_MIN}")
    return (f"comb: ML_DY < {ML_DY_COMB_CUT} && ML_JPSI < {ML_JPSI_CUT} "
            f"&& ML_PSIP < {ML_PSIP_CUT}, mass [{MASS_MIN},{MASS_MAX}], vz > {VZ_MIN}")



def compute_mass_ks(exp_mass: np.ndarray, sim_mass: np.ndarray) -> float:
    if len(sim_mass) < 2:
        return 1.0
    ks_stat, _ = ks_2samp(exp_mass, sim_mass)
    return float(ks_stat)



@torch.no_grad()
def rejection_sampling_streaming(
    input_file,
    output_file,
    tree_name,
    exp_model,
    sim_model,
    base_dist,
    scaler,
    ratio_clip_max: float,
    mass_weighter,          # MassWeighter instance or None
    batch_size: int = 10_000,
):
    t0     = time.time()
    device = next(exp_model.parameters()).device

    f_in_root = ROOT.TFile.Open(input_file, "READ")
    t_in      = f_in_root.Get(tree_name)
    f_out     = ROOT.TFile(output_file, "RECREATE")
    t_out     = t_in.CloneTree(0)
    t_out.SetAutoFlush(10000)

    f_up     = uproot.open(input_file)
    tree_up  = f_up[tree_name]
    n_events = int(tree_up.num_entries)

    has_ml_vars = all(v in tree_up.keys() for v in CUT_VARS)
    if has_ml_vars:
        vars_to_load = RS_VARS
        cut_function = apply_cuts_exp
        cut_label    = _exp_cut_description()
        print(f"[RS] ML variables found — applying EXP cuts ({MODE} mode)")
    else:
        vars_to_load = RS_VARS_SIM
        cut_function = apply_cuts_sim
        cut_label    = f"SIM cuts (mass [{MASS_MIN},{MASS_MAX}], vz > {VZ_MIN})"
        print(f"[RS] ML variables not found — applying SIM kinematic cuts")

    print(f"[RS] ratio_clip_max = {ratio_clip_max:.4f}  "
          f"mass_weighter = {'ON' if mass_weighter is not None else 'OFF (iter 1)'}")
    print(f"[RS] Streaming over {n_events:,} entries")
    M = float(max(ratio_clip_max, 1.0))

    n_acc_total   = 0
    n_cut_total   = 0
    n_processed   = 0
    accepted_mass = []

    for start in tqdm(range(0, n_events, batch_size), desc="RS chunks"):
        stop   = min(start + batch_size, n_events)
        arrays = tree_up.arrays(vars_to_load, library="np",
                                entry_start=start, entry_stop=stop)

        cut_mask     = cut_function(arrays)
        n_cut_total += int(cut_mask.sum())
        if not np.any(cut_mask):
            continue

        feats        = _extract_active(arrays, cut_mask)
        n_processed += (stop - start)

        scaled = scaler.transform(feats)
        scaled = np.clip(scaled, -CLIP_ZSCORE, CLIP_ZSCORE)
        xb     = torch.tensor(scaled, dtype=torch.float32, device=device)

        exp_lp = exp_model.log_prob(xb, base_dist)
        sim_lp = sim_model.log_prob(xb, base_dist)
        r_mom  = torch.exp(torch.clamp(exp_lp - sim_lp, min=-50.0, max=50.0))
        r_mom  = torch.clamp(r_mom, max=ratio_clip_max).cpu().numpy()

        if mass_weighter is not None:
            batch_mass = arrays["mass"][cut_mask]
            w_mass     = mass_weighter.weights(batch_mass)
        else:
            w_mass = np.ones(r_mom.shape[0], dtype=np.float32)

        combined = r_mom * w_mass
        probs    = np.clip(combined / M, 0.0, 1.0)

        accepts = (np.random.rand(int(cut_mask.sum())) < probs)
        if not np.any(accepts):
            continue

        cut_indices  = np.nonzero(cut_mask)[0]
        acc_local    = cut_indices[accepts]
        n_acc_total += int(acc_local.shape[0])

        accepted_mass.append(arrays["mass"][cut_mask][accepts])

        for k in acc_local:
            t_in.GetEntry(int(start + k))
            t_out.Fill()

        if n_acc_total > 0 and (n_acc_total % 10000) == 0:
            t_out.FlushBaskets()

    t_out.Write("", ROOT.TObject.kOverwrite)
    f_out.Close()
    f_in_root.Close()
    f_up.close()

    dt = time.time() - t0
    print(f"[RS] Cuts ({cut_label}): {n_cut_total:,}/{n_processed:,} "
          f"({100*n_cut_total/max(n_processed,1):.1f}%)")
    print(f"[RS] Accepted: {n_acc_total:,}  "
          f"rate: {100*n_acc_total/max(n_cut_total,1):.1f}%  "
          f"time: {dt:.1f}s")

    accepted_mass = np.concatenate(accepted_mass) if accepted_mass else np.array([])
    return n_acc_total, accepted_mass



def adaptive_rs_loop(exp_mass, exp_model, sim_model, base_dist, scaler):
    print("\n" + "=" * 60)
    print("  Adaptive RS: mass KDE ratio + KS minimisation")
    print(f"  Search range : [{RATIO_CLIP_MIN}, {RATIO_CLIP_MAX_CAP}]")
    print(f"  Max iters    : {ADAPTIVE_MAX_ITER}    tol: {ADAPTIVE_TOL}")
    print(f"  Mass weight cap : {MASS_WEIGHT_CAP}")
    print(f"  BDT estimators  : {BDT_N_ESTIMATORS}  lr: {BDT_LEARNING_RATE}")
    print("=" * 60)

    clip_lo  = RATIO_CLIP_MIN
    clip_hi  = RATIO_CLIP_MAX_CAP
    clip_cur = RATIO_CLIP_INIT

    best_ks   = float("inf")
    best_clip = clip_cur
    best_file = OUTPUT_FILE
    history   = []

    mass_weighter = BDTMassWeighter(exp_mass)
    prev_sim_mass = None

    for iteration in range(1, ADAPTIVE_MAX_ITER + 1):
        iter_file = OUTPUT_FILE.replace(".root", f"_clip{clip_cur:.3f}.root")

        print(f"\n[Iter {iteration}/{ADAPTIVE_MAX_ITER}]  "
              f"ratio_clip_max = {clip_cur:.4f}  "
              f"search [{clip_lo:.3f}, {clip_hi:.3f}]")

        if prev_sim_mass is not None and len(prev_sim_mass) >= 10:
            mass_weighter.update(prev_sim_mass)
            mw_active = mass_weighter
        else:
            print("[Iter] No prior accepted sample — running without mass weight.")
            mw_active = None

        # Step 2: run RS
        n_acc, sim_mass = rejection_sampling_streaming(
            input_file     = INPUT_FILE,
            output_file    = iter_file,
            tree_name      = TREE_NAME,
            exp_model      = exp_model,
            sim_model      = sim_model,
            base_dist      = base_dist,
            scaler         = scaler,
            ratio_clip_max = clip_cur,
            mass_weighter  = mw_active,
            batch_size     = BATCH_SIZE_RS,
        )

        if n_acc < 10:
            print(f"  Too few accepted events ({n_acc}) — raising clip.")
            clip_lo  = clip_cur
            clip_cur = min((clip_cur + clip_hi) / 2.0, RATIO_CLIP_MAX_CAP)
            continue

        prev_sim_mass = sim_mass   # save for next iteration's KDE update

        # Step 3: evaluate KS
        ks = compute_mass_ks(exp_mass, sim_mass)
        history.append((clip_cur, ks, n_acc))
        print(f"  Mass KS = {ks:.6f}   n_acc = {n_acc:,}")

        if ks < best_ks - ADAPTIVE_TOL:
            best_ks   = ks
            best_clip = clip_cur
            best_file = iter_file
            print(f"  ✓ New best  KS={best_ks:.6f}  clip={best_clip:.4f}")
        else:
            print(f"  No improvement  (best KS={best_ks:.6f} @ clip={best_clip:.4f})")

        # Step 4: bisect
        if len(history) >= 2 and ks < history[-2][1]:
            clip_lo  = clip_cur
            clip_cur = min(clip_cur * 1.5, clip_hi)
        else:
            clip_hi  = clip_cur
            clip_cur = (clip_lo + clip_cur) / 2.0

        if (clip_hi - clip_lo) < 0.05:
            print(f"\n[Adaptive] Interval collapsed to "
                  f"[{clip_lo:.3f}, {clip_hi:.3f}] — converged.")
            break

    # ---- summary ----
    print("\n" + "=" * 60)
    print("  Adaptive search complete")
    print(f"  Best ratio_clip_max : {best_clip:.4f}")
    print(f"  Best mass KS        : {best_ks:.6f}")
    print("  Full history (clip, KS, n_acc):")
    for clip, ks, n in history:
        marker = "  ← best" if abs(clip - best_clip) < 1e-6 else ""
        print(f"    clip={clip:8.4f}  KS={ks:.6f}  n_acc={n:,}{marker}")
    print("=" * 60)

    if best_file != OUTPUT_FILE:
        shutil.copy2(best_file, OUTPUT_FILE)
        print(f"\nCopied best output → {OUTPUT_FILE}")

    return best_clip, best_ks


# =========================================================
#                         Main
# =========================================================
def main():
    print(f"=== Running in MODE = '{MODE}' ===")
    print(f"  Stage 1 : 6-D momentum NF (train once, freeze)")
    print(f"  Stage 2 : adaptive RS with mass KDE ratio weight "
          f"(minimise KS distance)")
    print(f"  EXP cuts: {_exp_cut_description()}")
    print(f"  Output  : {OUTPUT_FILE}\n")

    n_exp_total = count_entries(EXP_FILE, TREE_NAME)
    n_sim_total = count_entries(SIM_FILE, TREE_NAME)
    print(f"EXP total: {n_exp_total:,}   SIM total: {n_sim_total:,}")

    # ----------------------------------------------------------
    # STEP 1 — Load training data (6-D momentum, after cuts)
    # ----------------------------------------------------------
    print("\n=== STEP 1: Load training data ===")
    exp_points = load_first_n_flat(EXP_FILE, TREE_NAME, ALL_VARS,
                                   n_exp_total, cut_type='exp')
    n_exp_after_cuts = exp_points.shape[0]
    sim_points = load_random_window(SIM_FILE, TREE_NAME, ALL_VARS_SIM,
                                    10000 * n_exp_after_cuts,
                                    seed=SEED, cut_type='sim')
    print(f"EXP train: {exp_points.shape}   SIM train: {sim_points.shape}")

    # ----------------------------------------------------------
    # STEP 2 — Scaler (fit on EXP, apply to both)
    # ----------------------------------------------------------
    print("\n=== STEP 2: Fit scaler ===")
    exp_scaled, sim_scaled, scaler = fit_scaler_on_exp_and_transform(
        exp_points, sim_points)
    exp_tensor, device = to_device_tensor(exp_scaled)
    sim_tensor, _      = to_device_tensor(sim_scaled)

    dim       = len(ACTIVE_VARS)
    base_dist = make_base_dist(dim, device=device)

    # ----------------------------------------------------------
    # STEP 3-5 — Build & train flows (trained once, then frozen)
    # ----------------------------------------------------------
    print(f"\n=== STEP 3: Build {dim}-D flow models ===")
    exp_model = FlowModel(dim, num_layers=6, hidden_dim=64).to(device)
    sim_model = FlowModel(dim, num_layers=6, hidden_dim=64).to(device)

    print("\n=== STEP 4: Train EXP flow ===")
    train_flow(exp_model, base_dist, exp_tensor,
               epochs=EPOCHS_EXP, batch_size=BATCH_SIZE_TRAIN, lr=LR)

    print("\n=== STEP 5: Train SIM flow ===")
    train_flow(sim_model, base_dist, sim_tensor,
               epochs=EPOCHS_SIM, batch_size=BATCH_SIZE_TRAIN, lr=LR)

    exp_model.eval()
    sim_model.eval()

    # ----------------------------------------------------------
    # STEP 6 — Load EXP mass for KS evaluation
    # ----------------------------------------------------------
    print("\n=== STEP 6: Load EXP mass for KS evaluation ===")
    with uproot.open(EXP_FILE) as f:
        exp_arr = f[TREE_NAME].arrays(
            list(set(CUT_VARS + ["mass"])),
            library="np", entry_stop=n_exp_total)
    exp_mass_vals = exp_arr["mass"][apply_cuts_exp(exp_arr)]
    print(f"  EXP mass events for KS: {len(exp_mass_vals):,}")

    # ----------------------------------------------------------
    # STEP 7 — Adaptive RS: alternate between updating w_mass
    #          and bisecting ratio_clip_max to minimise KS
    # ----------------------------------------------------------
    print("\n=== STEP 7: Adaptive rejection sampling (mass KDE + KS) ===")
    best_clip, best_ks = adaptive_rs_loop(
        exp_mass  = exp_mass_vals,
        exp_model = exp_model,
        sim_model = sim_model,
        base_dist = base_dist,
        scaler    = scaler,
    )

    print(f"\nAll steps completed.")
    print(f"  Final ratio_clip_max : {best_clip:.4f}")
    print(f"  Final mass KS        : {best_ks:.6f}")
    print(f"  Output               : {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
