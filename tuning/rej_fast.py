#!/usr/bin/env python3
"""
rej_fast.py — Two-phase combinatoric background reweighting for SpinQuest.
Phase 1: NF-based rejection sampling over full SIM to produce an intermediate file.
Phase 2: BDT mass reweighting with adaptive bisection of bdt_clip to minimise KS.
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
assert MODE in ("jpsi", "psip", "dy", "comb")


SIM_FILES = {
    "jpsi": "/Users/spin/CombBkgDNP/comb/data/jpsi/raw_mc_jpsi_target_pythia8.root",
    "psip": "/Users/spin/CombBkgDNP/comb/data/psip/raw_mc_psip_target_pythia8.root",
    "dy":   "/Users/spin/CombBkgDNP/comb/dy/dy.root",
    "comb": "/Users/spin/CombBkgDNP/comb/data/mc_comb_muon_gun_march19.root",
}
SIM_FILE  = SIM_FILES[MODE]
EXP_FILE  = "/Users/spin/spinquest-combinatoric-bkg/data/raw_input/exp_tagged_tgt_data_II.root"

NF_ACCEPTED_FILE = f"data/nf_accepted_{MODE}.root"       # Phase 1 output
OUTPUT_FILE      = f"data/out_test_{MODE}_momentum_adaptive.root"  # Phase 2 final
TREE_NAME        = "tree"


ML_JPSI_CUT    = 0.8
ML_PSIP_CUT    = 0.8
ML_DY_COMB_CUT = 0.8
VZ_MIN         = -600.0

_MASS_RANGES = {
    "jpsi": (2.2, 4.2),
    "psip": (2.8, 4.8),
    "dy":   (1.5, 6.0),
    "comb": (1.5, 6.0),
}
MASS_MIN, MASS_MAX = _MASS_RANGES[MODE]

MOMENTUM_RANGES = {
    "rec_dimu_mu_pos_px": (-3.2, 0.5),
    "rec_dimu_mu_pos_py": (-2.5, 2.5),
    "rec_dimu_mu_pos_pz": (20.0, 70.0),
    "rec_dimu_mu_neg_px": (-0.5, 3.2),
    "rec_dimu_mu_neg_py": (-2.5, 2.5),
    "rec_dimu_mu_neg_pz": (20.0, 70.0),
}

ACTIVE_VARS = [
    "rec_dimu_mu_pos_px", "rec_dimu_mu_pos_py", "rec_dimu_mu_pos_pz",
    "rec_dimu_mu_neg_px", "rec_dimu_mu_neg_py", "rec_dimu_mu_neg_pz",
]


CLIP_ZSCORE      = 15.0
BATCH_SIZE_TRAIN = 1_000
EPOCHS_EXP       = 700
EPOCHS_SIM       = 700
LR               = 1e-4
BATCH_SIZE_RS    = 1_000


# ── FROZEN BEST CONFIG (KS=0.0435, bdt_clip=2.75, n_acc=804k) ──────────────
BDT_TRAIN_SAMPLE  = 100_000   # sample from NF-accepted for BDT training
BDT_N_ESTIMATORS  = 200
BDT_LEARNING_RATE = 0.1
MASS_WEIGHT_CAP   = 10.0

NF_RATIO_CLIP     = 4.5       # fixed NF clip (best from small-dataset tuning)
BDT_CLIP_MIN      = 0.5
BDT_CLIP_MAX_CAP  = 5.0
BDT_CLIP_INIT     = (BDT_CLIP_MIN + BDT_CLIP_MAX_CAP) / 2  # 2.75 — midpoint bisection
ADAPTIVE_MAX_ITER = 5         # bisection iterations
N_REFINE          = 0         # refinement iterations (set >0 to enable; found to hurt, disabled)
ADAPTIVE_TOL      = 1e-4
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)



class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        in_dim  = dim // 2
        out_dim = dim - in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * out_dim),
        )

    def forward(self, x):
        x1, x2 = x[:, :self.dim // 2], x[:, self.dim // 2:]
        theta   = self.net(x1)
        s, t    = theta[:, :x2.shape[1]], theta[:, x2.shape[1]:]
        s       = torch.clamp(s, -7.0, 7.0)
        y2      = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=-1), s.sum(dim=-1)


class Permute(nn.Module):
    def __init__(self, perm):
        super().__init__()
        self.register_buffer("perm", perm)

    def forward(self, x):
        return x[:, self.perm]


class FlowModel(nn.Module):
    def __init__(self, dim, num_layers=6, hidden_dim=64):
        super().__init__()
        mods = []
        for _ in range(num_layers):
            mods.append(AffineCoupling(dim, hidden_dim))
            mods.append(Permute(torch.randperm(dim)))
        self.transforms = nn.ModuleList(mods)

    def forward_flow(self, x):
        z, ldt = x, torch.zeros(x.shape[0], device=x.device)
        for t in self.transforms:
            if isinstance(t, AffineCoupling):
                z, ld = t(z); ldt += ld
            else:
                z = t(z)
        return z, ldt

    def log_prob(self, x, base):
        z, ldt = self.forward_flow(x)
        return base.log_prob(z) + ldt


def make_base(dim, device):
    return D.Independent(D.Normal(torch.zeros(dim, device=device),
                                   torch.ones(dim, device=device)), 1)


def train_flow(model, base, data, epochs, batch_size, lr):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n   = data.shape[0]
    bs  = min(int(batch_size), n)
    for _ in tqdm(range(int(epochs)), desc="Training"):
        idx  = torch.randint(0, n, (bs,), device=data.device)
        loss = -model.log_prob(data[idx], base).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()



CUT_VARS_EXP = ["ml_p_jpsi", "ml_p_psip", "ml_p_dy",
                 "mass", "rec_track_neg_vz", "rec_track_pos_vz"]
CUT_VARS_SIM = ["mass", "rec_track_neg_vz", "rec_track_pos_vz"]

def apply_cuts_exp(a):
    ml = (a["ml_p_dy"] < ML_DY_COMB_CUT) & \
         (a["ml_p_jpsi"] < ML_JPSI_CUT)  & \
         (a["ml_p_psip"] < ML_PSIP_CUT)
    m  = ml & \
         (a["mass"] > MASS_MIN) & (a["mass"] < MASS_MAX) & \
         (a["rec_track_neg_vz"] > VZ_MIN) & (a["rec_track_pos_vz"] > VZ_MIN) & \
         (np.abs(a["rec_track_pos_vz"] - a["rec_track_neg_vz"]) < 200)
    for v, (lo, hi) in MOMENTUM_RANGES.items():
        m &= (a[v] > lo) & (a[v] < hi)
    return m

def apply_cuts_sim(a):
    m = (a["mass"] > MASS_MIN) & (a["mass"] < MASS_MAX) & \
        (a["rec_track_neg_vz"] > VZ_MIN) & (a["rec_track_pos_vz"] > VZ_MIN) & \
        (np.abs(a["rec_track_pos_vz"] - a["rec_track_neg_vz"]) < 200)
    for v, (lo, hi) in MOMENTUM_RANGES.items():
        m &= (a[v] > lo) & (a[v] < hi)
    return m

ALL_VARS_EXP = list(set(ACTIVE_VARS + CUT_VARS_EXP + ["mass"]))
ALL_VARS_SIM = list(set(ACTIVE_VARS + CUT_VARS_SIM + ["mass"]))


def _extract_active(arrays, mask):
    return np.column_stack([arrays[v][mask] for v in ACTIVE_VARS])



class BDTMassWeighter:
    def __init__(self, exp_mass):
        self._exp = np.clip(exp_mass, MASS_MIN, MASS_MAX).reshape(-1, 1)
        self._rw  = None

    def update(self, sim_mass, n_sample=None):
        sim = np.clip(sim_mass, MASS_MIN, MASS_MAX).reshape(-1, 1)
        # Use ALL EXP events; subsample SIM up to n_sample (asymmetric is fine for GBReweighter)
        n_sim = len(sim)
        if n_sample:
            n_sim = min(n_sim, n_sample)
        rng = np.random.default_rng(SEED)
        si  = rng.choice(len(sim), n_sim, replace=False)
        self._rw = hep_reweight.GBReweighter(
            n_estimators=BDT_N_ESTIMATORS, learning_rate=BDT_LEARNING_RATE)
        self._rw.fit(sim[si], self._exp)
        print(f"  [BDT] trained on {n_sim:,} SIM vs {len(self._exp):,} EXP events")

    def weights(self, mass):
        if self._rw is None:
            return np.ones(len(mass), dtype=np.float32)
        w = self._rw.predict_weights(
            np.clip(mass, MASS_MIN, MASS_MAX).reshape(-1, 1))
        w = np.clip(w, 0.0, MASS_WEIGHT_CAP)
        m = w.mean()
        if m > 1e-8:
            w /= m
        return w.astype(np.float32)



@torch.no_grad()
def phase1_nf_rs(exp_model, sim_model, base, scaler, device):
    """Stream full SIM, accept with r_momentum / NF_RATIO_CLIP. Save to NF_ACCEPTED_FILE."""
    print(f"\n{'='*60}")
    print(f"  PHASE 1: NF rejection sampling  (clip={NF_RATIO_CLIP})")
    print(f"  Input : {SIM_FILE}")
    print(f"  Output: {NF_ACCEPTED_FILE}")
    print(f"{'='*60}")

    t0 = time.time()
    f_in_root = ROOT.TFile.Open(SIM_FILE, "READ")
    t_in      = f_in_root.Get(TREE_NAME)
    f_out     = ROOT.TFile(NF_ACCEPTED_FILE, "RECREATE")
    t_out     = t_in.CloneTree(0)
    t_out.SetAutoFlush(10_000)

    f_up     = uproot.open(SIM_FILE)
    tree_up  = f_up[TREE_NAME]
    n_events = int(tree_up.num_entries)
    print(f"  Streaming {n_events:,} entries ...")

    n_acc = 0
    acc_mass = []

    for start in tqdm(range(0, n_events, BATCH_SIZE_RS), desc="Phase1 RS"):
        stop   = min(start + BATCH_SIZE_RS, n_events)
        arrays = tree_up.arrays(ALL_VARS_SIM, library="np",
                                entry_start=start, entry_stop=stop)
        mask = apply_cuts_sim(arrays)
        if not np.any(mask):
            continue

        feats  = _extract_active(arrays, mask)
        scaled = np.clip(scaler.transform(feats), -CLIP_ZSCORE, CLIP_ZSCORE)
        xb     = torch.tensor(scaled, dtype=torch.float32, device=device)

        r_mom  = torch.exp(torch.clamp(
            exp_model.log_prob(xb, base) - sim_model.log_prob(xb, base),
            -50.0, 50.0))
        r_mom  = torch.clamp(r_mom, max=NF_RATIO_CLIP).cpu().numpy()

        probs  = np.clip(r_mom / NF_RATIO_CLIP, 0.0, 1.0)
        acc    = np.random.rand(int(mask.sum())) < probs
        if not np.any(acc):
            continue

        cut_idx = np.nonzero(mask)[0]
        for k in cut_idx[acc]:
            t_in.GetEntry(int(start + k))
            t_out.Fill()

        acc_mass.append(arrays["mass"][mask][acc])
        n_acc += int(acc.sum())
        if n_acc % 50_000 == 0 and n_acc > 0:
            t_out.FlushBaskets()

    t_out.Write("", ROOT.TObject.kOverwrite)
    f_out.Close()
    f_in_root.Close()
    f_up.close()

    acc_mass = np.concatenate(acc_mass) if acc_mass else np.array([])
    dt = time.time() - t0
    print(f"  Phase 1 done: {n_acc:,} accepted  "
          f"({100*n_acc/max(n_events,1):.1f}%)  time: {dt:.1f}s")
    return acc_mass



def phase2_bdt_rs(exp_mass, nf_mass, bdt_clip, mass_weighter, iteration):
    """
    Stream NF_ACCEPTED_FILE. Accept each event with
        p = clip(w_mass / bdt_clip, 0, 1)
    Return accepted mass array.
    """
    out_tmp = OUTPUT_FILE.replace(".root", f"_iter{str(iteration)}.root")

    f_in_root = ROOT.TFile.Open(NF_ACCEPTED_FILE, "READ")
    t_in      = f_in_root.Get(TREE_NAME)
    f_out     = ROOT.TFile(out_tmp, "RECREATE")
    t_out     = t_in.CloneTree(0)
    t_out.SetAutoFlush(10_000)

    f_up    = uproot.open(NF_ACCEPTED_FILE)
    tree_up = f_up[TREE_NAME]
    n_events = int(tree_up.num_entries)

    n_acc    = 0
    acc_mass = []

    for start in tqdm(range(0, n_events, BATCH_SIZE_RS),
                      desc=f"Phase2 iter {iteration}"):
        stop   = min(start + BATCH_SIZE_RS, n_events)
        arrays = tree_up.arrays(["mass"], library="np",
                                entry_start=start, entry_stop=stop)
        mass   = arrays["mass"]

        w_mass = mass_weighter.weights(mass)
        probs  = np.clip(w_mass / bdt_clip, 0.0, 1.0)
        acc    = np.random.rand(len(mass)) < probs

        for k in np.nonzero(acc)[0]:
            t_in.GetEntry(int(start + k))
            t_out.Fill()

        acc_mass.append(mass[acc])
        n_acc += int(acc.sum())
        if n_acc % 50_000 == 0 and n_acc > 0:
            t_out.FlushBaskets()

    t_out.Write("", ROOT.TObject.kOverwrite)
    f_out.Close()
    f_in_root.Close()
    f_up.close()

    acc_mass = np.concatenate(acc_mass) if acc_mass else np.array([])
    ks = float(ks_2samp(exp_mass, acc_mass)[0]) if len(acc_mass) >= 2 else 1.0
    print(f"  [Iter {iteration}] bdt_clip={bdt_clip:.4f}  "
          f"n_acc={n_acc:,}  KS={ks:.6f}")
    return acc_mass, ks, out_tmp



def main():
    print(f"=== rej_fast.py  MODE={MODE} ===")
    print(f"  Phase 1 : NF RS over full SIM (clip={NF_RATIO_CLIP})")
    print(f"  Phase 2 : BDT mass RS over NF-accepted only ({ADAPTIVE_MAX_ITER} iters)")
    print(f"  BDT     : {BDT_N_ESTIMATORS} trees, trained on {BDT_TRAIN_SAMPLE:,} samples\n")

    print("=== Loading EXP data ===")
    with uproot.open(EXP_FILE) as f:
        exp_arr = f[TREE_NAME].arrays(ALL_VARS_EXP, library="np")
    mask_exp   = apply_cuts_exp(exp_arr)
    exp_pts    = _extract_active(exp_arr, mask_exp)
    exp_mass   = exp_arr["mass"][mask_exp]
    print(f"  EXP: {mask_exp.sum():,} events after cuts")

    print("\n=== Loading SIM for flow training ===")
    with uproot.open(SIM_FILE) as f:
        n_total = f[TREE_NAME].num_entries
        # sample up to 10× EXP size
        n_load  = min(int(n_total), 10_000 * len(exp_pts))
        rng     = np.random.default_rng(SEED)
        start   = int(rng.integers(0, max(1, n_total - n_load + 1)))
        sim_arr = f[TREE_NAME].arrays(ALL_VARS_SIM, library="np",
                                      entry_start=start,
                                      entry_stop=start + n_load)
    mask_sim = apply_cuts_sim(sim_arr)
    sim_pts  = _extract_active(sim_arr, mask_sim)
    print(f"  SIM: {mask_sim.sum():,} events after cuts")

    print("\n=== Fitting scaler ===")
    scaler   = StandardScaler().fit(exp_pts)
    exp_sc   = np.clip(scaler.transform(exp_pts), -CLIP_ZSCORE, CLIP_ZSCORE)
    sim_sc   = np.clip(scaler.transform(sim_pts), -CLIP_ZSCORE, CLIP_ZSCORE)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_t    = torch.tensor(exp_sc, dtype=torch.float32, device=device)
    sim_t    = torch.tensor(sim_sc, dtype=torch.float32, device=device)
    dim      = len(ACTIVE_VARS)
    base     = make_base(dim, device)

    print(f"\n=== Training EXP flow ({EPOCHS_EXP} epochs) ===")
    exp_model = FlowModel(dim).to(device)
    train_flow(exp_model, base, exp_t, EPOCHS_EXP, BATCH_SIZE_TRAIN, LR)
    exp_model.eval()

    print(f"\n=== Training SIM flow ({EPOCHS_SIM} epochs) ===")
    sim_model = FlowModel(dim).to(device)
    train_flow(sim_model, base, sim_t, EPOCHS_SIM, BATCH_SIZE_TRAIN, LR)
    sim_model.eval()

    import os
    if os.path.exists(NF_ACCEPTED_FILE):
        print(f"\n=== Skipping Phase 1 — {NF_ACCEPTED_FILE} already exists ===")
        with uproot.open(NF_ACCEPTED_FILE) as f:
            nf_mass = f[TREE_NAME].arrays(["mass"], library="np")["mass"]
        print(f"  Loaded {len(nf_mass):,} NF-accepted events")
    else:
        nf_mass = phase1_nf_rs(exp_model, sim_model, base, scaler, device)

    print(f"\n{'='*60}")
    print(f"  PHASE 2: adaptive BDT mass RS")
    print(f"  NF-accepted file: {NF_ACCEPTED_FILE}  ({len(nf_mass):,} events)")
    print(f"{'='*60}")

    mass_weighter = BDTMassWeighter(exp_mass)

    best_ks     = float("inf")
    best_file   = None
    best_mass   = nf_mass
    bdt_clip    = BDT_CLIP_INIT
    clip_lo, clip_hi = BDT_CLIP_MIN, BDT_CLIP_MAX_CAP

    mass_weighter.update(nf_mass, n_sample=BDT_TRAIN_SAMPLE)

    for it in range(1, ADAPTIVE_MAX_ITER + 1):
        print(f"\n--- Phase 2, Iter {it}/{ADAPTIVE_MAX_ITER} "
              f"  bdt_clip={bdt_clip:.4f}  search=[{clip_lo:.3f},{clip_hi:.3f}] ---")

        acc_mass, ks, tmp_file = phase2_bdt_rs(
            exp_mass, nf_mass, bdt_clip, mass_weighter, it)

        if ks < best_ks:
            best_ks   = ks
            best_file = tmp_file
            best_mass = acc_mass
            print(f"  ✓ New best  KS={best_ks:.6f}  clip={bdt_clip:.4f}")
            clip_lo = bdt_clip
            mass_weighter.update(best_mass, n_sample=BDT_TRAIN_SAMPLE)
        else:
            print(f"  No improvement (best KS={best_ks:.6f})")
            clip_hi = bdt_clip

        if (clip_hi - clip_lo) < ADAPTIVE_TOL:
            print("  Bisection converged.")
            break

        bdt_clip = (clip_lo + clip_hi) / 2.0

    best_clip = clip_lo
    print(f"\n{'='*60}")
    print(f"  REFINEMENT: {N_REFINE} iterations at best clip={best_clip:.4f}")
    print(f"{'='*60}")
    for ri in range(1, N_REFINE + 1):
        mass_weighter.update(best_mass, n_sample=BDT_TRAIN_SAMPLE)
        acc_mass, ks, tmp_file = phase2_bdt_rs(
            exp_mass, nf_mass, best_clip, mass_weighter, f"R{ri}")
        if ks < best_ks:
            best_ks   = ks
            best_file = tmp_file
            best_mass = acc_mass
            print(f"  ✓ Refinement new best  KS={best_ks:.6f}")
        else:
            print(f"  No improvement (best KS={best_ks:.6f})")

    if best_file:
        shutil.copy(best_file, OUTPUT_FILE)
        print(f"\nCopied best output → {OUTPUT_FILE}")

    print(f"\n{'='*60}")
    print(f"  rej_fast.py complete")
    print(f"  Best KS (mass) : {best_ks:.6f}")
    print(f"  Output         : {OUTPUT_FILE}")
    print(f"  NF-accepted    : {NF_ACCEPTED_FILE} (reusable)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
