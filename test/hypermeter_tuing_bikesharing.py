"""
Hyperparameter tuning for InstaSHAP on Bike Sharing dataset.
Tunes: GAM order (k=1,2,3) × number of interactions (N).

Strategy:
  For each order k, run SIAN once with max_rounds to discover interactions in
  importance order. Then incrementally train InstaSHAP with top-N interactions
  and evaluate R² error on validation set.

Outputs:
  - tuning_results.json   (all numerical results)
  - tuning_trend.png      (line plot: R² error vs N, one line per order k)
"""

#%% 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

# ---- import your InstaSHAP code ----
import sys
from pathlib import Path

sys.path.append(str(Path("..").resolve()))

from src.instashap import ShapleySampler, InstaSHAP, surrogate
from sian.models import TrainingArgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================================================================
#  Config
# ========================================================================
DATA_ROOT = "../data/"
DATASET_NAME = "bike_sharing"
SEED = 37
BATCH_SIZE = 64
INSTASHAP_EPOCHS = 100
INSTASHAP_LR = 5e-3
MAX_ROUNDS = 50          # max interactions SIAN discovers per order
ORDERS = [1, 2, 3]       # GAM orders to try
# For each order, we test N = d (singletons only), d+2, d+4, ..., up to all discovered
# d=13 singletons are always included

RESULTS_DIR = "./tuning_results/"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ========================================================================
#  Data
# ========================================================================
from src.dataloader import CustomBikeDataset

dataset_obj = CustomBikeDataset(root_dir=DATA_ROOT, dataset_name=DATASET_NAME, seed=SEED)
dataset_obj.shuffle_and_split_trnval(trnval_shuffle_seed=SEED)
trnX, trnY, valX, valY = dataset_obj.pull_trnval_data()
tstX, tstY = dataset_obj.tstX, dataset_obj.tstY

d = dataset_obj.get_D()  # 13
print(f"Features d={d}, Train={trnX.shape[0]}, Val={valX.shape[0]}, Test={tstX.shape[0]}")

X_tr_t = torch.from_numpy(trnX).float()
Y_tr_t = torch.from_numpy(trnY).float()
y_mean = Y_tr_t.mean()
Y_centered = Y_tr_t - y_mean

train_ds = TensorDataset(X_tr_t, Y_centered)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

X_val_t = torch.from_numpy(valX).float().to(device)
Y_val_t = torch.from_numpy(valY).float()

X_tst_t = torch.from_numpy(tstX).float().to(device)
Y_tst_t = torch.from_numpy(tstY).float()


# ========================================================================
#  Helper: build transform matrix from a subset of interactions
# ========================================================================
def build_transform_matrix(interactions_subset, d):
    """interactions_subset: list of tuples, index 0 = () placeholder."""
    n_inter = len(interactions_subset) - 1  # skip index 0
    tm = torch.zeros(d, n_inter)
    for j in range(1, len(interactions_subset)):
        for feat_idx in interactions_subset[j]:
            tm[feat_idx, j - 1] = 1.0
    return tm


def eval_r2_error(model, X_t, Y_np, y_mean_val):
    """Compute R² error (%) = (1 - R²) * 100."""
    model.eval()
    with torch.no_grad():
        full_mask = torch.ones_like(X_t)
        preds = model(X_t, full_mask).cpu().squeeze() + y_mean_val
    y_true = torch.from_numpy(Y_np).float().squeeze()
    ss_res = ((y_true - preds) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return (1 - r2.item()) * 100


def train_instashap_quick(interactions_subset, transform_matrix, train_loader,
                          num_epochs, lr, device):
    """Train and return an InstaSHAP model."""
    model = InstaSHAP(interactions_subset, transform_matrix, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    sampler = ShapleySampler(num_features=d)

    model.train()
    for epoch in range(num_epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            S = sampler.sample(xb.shape[0], paired_sampling=True).to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, S), yb)
            loss.backward()
            optimizer.step()
    return model


# ========================================================================
#  Step 1: Run SIAN for each order k (do this once, cache results)
# ========================================================================
all_discovered = {}  # k -> list of interaction tuples (including () at index 0)

for order_k in ORDERS:
    print(f"\n{'='*50}")
    print(f"  SIAN interaction discovery: order k={order_k}, max_rounds={MAX_ROUNDS}")
    print(f"{'='*50}")

    exp_folder = os.path.join(RESULTS_DIR, f"sian_order{order_k}/")
    os.makedirs(exp_folder, exist_ok=True)

    mlp_args = TrainingArgs(batch_size=32, number_of_epochs=10,
                            learning_rate=5e-3, device=device)
    mlp_args.model_config.net_name = "MLP"
    mlp_args.model_config.sizes = [-1, 128, 256, 128, -1]
    mlp_args.model_config.is_masked = True
    mlp_args.saving_settings.exp_folder = exp_folder

    my_surrogate = surrogate(mlp_args=mlp_args, dataset_obj=dataset_obj,
                             max_number_of_rounds=MAX_ROUNDS, order=order_k)
    interactions = my_surrogate.get_interactions(device=device)

    all_discovered[order_k] = interactions
    print(f"  Discovered {len(interactions)-1} interactions (+ phi_0)")
    for idx, inter in enumerate(interactions):
        if idx == 0:
            continue
        labels = dataset_obj.get_readable_labels()
        feat_names = [labels.get(f, str(f)) for f in inter]
        print(f"    [{idx}] {inter}  ({' × '.join(feat_names)})")


# ========================================================================
#  Step 2: For each (order, N), train InstaSHAP and evaluate
# ========================================================================
results = []  # list of dicts

for order_k in ORDERS:
    interactions_full = all_discovered[order_k]
    total_available = len(interactions_full) - 1  # excluding phi_0

    # Separate singletons from higher-order interactions
    singletons = [i for i in range(1, len(interactions_full))
                  if len(interactions_full[i]) == 1]
    higher_order = [i for i in range(1, len(interactions_full))
                    if len(interactions_full[i]) > 1]

    n_singletons = len(singletons)
    n_higher = len(higher_order)

    # N values to test: singletons only, then add higher-order ones incrementally
    N_values = [n_singletons]  # baseline: singletons only (= GAM-1 equivalent)
    for extra in range(1, n_higher + 1):
        N_values.append(n_singletons + extra)

    print(f"\n{'='*50}")
    print(f"  Tuning order k={order_k}: {n_singletons} singletons + up to {n_higher} higher-order")
    print(f"  N values to test: {N_values}")
    print(f"{'='*50}")

    for N in N_values:
        # take first N interactions (singletons first, then higher-order in discovery order)
        selected_indices = singletons + higher_order[:N - n_singletons]
        interactions_subset = [()]  # phi_0 placeholder
        for idx in selected_indices:
            interactions_subset.append(interactions_full[idx])

        tm = build_transform_matrix(interactions_subset, d)

        print(f"\n  k={order_k}, N={N} ({N - n_singletons} higher-order)  training...", end=" ")

        model = train_instashap_quick(
            interactions_subset, tm, train_loader,
            num_epochs=INSTASHAP_EPOCHS, lr=INSTASHAP_LR, device=device
        )

        val_r2_err = eval_r2_error(model, X_val_t, valY, y_mean)
        tst_r2_err = eval_r2_error(model, X_tst_t, tstY, y_mean)

        print(f"Val R²err={val_r2_err:.2f}%  Test R²err={tst_r2_err:.2f}%")

        entry = {
            "order_k": order_k,
            "N_total": N,
            "N_singletons": n_singletons,
            "N_higher_order": N - n_singletons,
            "val_r2_error_pct": round(val_r2_err, 4),
            "tst_r2_error_pct": round(tst_r2_err, 4),
            "interactions_used": [list(interactions_subset[i])
                                  for i in range(1, len(interactions_subset))],
        }
        results.append(entry)

        # free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ========================================================================
#  Step 3: Save results
# ========================================================================
results_path = os.path.join(RESULTS_DIR, "tuning_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")


# ========================================================================
#  Step 4: Plot
# ========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
markers = {1: "o", 2: "s", 3: "^"}

for order_k in ORDERS:
    subset = [r for r in results if r["order_k"] == order_k]
    n_higher = [r["N_higher_order"] for r in subset]
    val_errs = [r["val_r2_error_pct"] for r in subset]
    tst_errs = [r["tst_r2_error_pct"] for r in subset]

    ax1.plot(n_higher, val_errs, marker=markers[order_k], color=colors[order_k],
             label=f"GAM-{order_k}", linewidth=2, markersize=6)
    ax2.plot(n_higher, tst_errs, marker=markers[order_k], color=colors[order_k],
             label=f"GAM-{order_k}", linewidth=2, markersize=6)

ax1.set_xlabel("Number of Higher-Order Interactions Added")
ax1.set_ylabel("R² Error (%)")
ax1.set_title("Validation Set")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Number of Higher-Order Interactions Added")
ax2.set_ylabel("R² Error (%)")
ax2.set_title("Test Set")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle("Bike Sharing: InstaSHAP R² Error vs. Number of Interactions", fontsize=13, y=1.02)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "tuning_trend.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved to {plot_path}")


# ========================================================================
#  Step 5: Print summary table
# ========================================================================
print("\n" + "=" * 70)
print(f"{'Order k':<10}{'N_higher':<12}{'Val R²err%':<14}{'Test R²err%':<14}")
print("-" * 70)
for r in results:
    print(f"{r['order_k']:<10}{r['N_higher_order']:<12}{r['val_r2_error_pct']:<14.2f}{r['tst_r2_error_pct']:<14.2f}")

# find best config
best = min(results, key=lambda r: r["val_r2_error_pct"])
print("-" * 70)
print(f"Best (by val): order k={best['order_k']}, "
      f"N_higher={best['N_higher_order']}, "
      f"Val={best['val_r2_error_pct']:.2f}%, Test={best['tst_r2_error_pct']:.2f}%")