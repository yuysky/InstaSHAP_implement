"""
Reproduce experiments from the InstaSHAP paper (ICLR 2025).
- Experiment 1: 10D Synthetic (Section 6.1, Figure 3)
- Experiment 2: Bike Sharing synergy (Section 6.2)
"""
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from itertools import combinations
from copy import deepcopy
import os, sys

# ---- import your InstaSHAP code ----
import sys
from pathlib import Path

sys.path.append(str(Path("..").resolve()))

from src.instashap import ShapleySampler, InstaSHAP, phi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========================================================================
# FastSHAP baseline (simple MLP that directly outputs d SHAP values)
# ========================================================================
class FastSHAP(nn.Module):
    """
    FastSHAP explainer: a single network x -> phi(x) in R^d,
    trained with Shapley-kernel weighted MSE using the same masked objective.
    """
    def __init__(self, num_features, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_features),
        )
        self.phi_0 = nn.Parameter(torch.zeros(1))

    def forward(self, x, S):
        shap_vals = self.net(x)                   # (B, d)
        output = (shap_vals * S).sum(dim=1, keepdim=True) + self.phi_0
        return output

    def get_shapley_values(self, x):
        self.eval()
        with torch.no_grad():
            return self.net(x)


# ========================================================================
#  Synthetic Data Generator  (Appendix D.2)
# ========================================================================
def generate_synthetic_data(n_samples, d=10, k_star=1, rho=0.0, seed=42):
    """
    Generate 10D synthetic data with correlated-pairs covariance.
    f(x) = sum_{S in I<=k*} beta_S * prod_{i in S} x_i
    Normalised so Var(f) = 1.
    """
    rng = np.random.default_rng(seed)

    # build block-diagonal covariance  (5 pairs)
    Sigma = np.eye(d)
    for i in range(0, d, 2):
        if i + 1 < d:
            Sigma[i, i + 1] = rho
            Sigma[i + 1, i] = rho

    X = rng.multivariate_normal(np.zeros(d), Sigma, size=n_samples).astype(np.float32)

    # enumerate all subsets of size <= k*
    subsets = []
    for k in range(1, k_star + 1):
        for combo in combinations(range(d), k):
            subsets.append(combo)

    betas = rng.standard_normal(len(subsets)).astype(np.float32)

    # compute target
    Y = np.zeros(n_samples, dtype=np.float32)
    for beta, S in zip(betas, subsets):
        prod = np.ones(n_samples, dtype=np.float32)
        for i in S:
            prod *= X[:, i]
        Y += beta * prod

    # normalise to unit variance
    std = Y.std()
    if std > 1e-8:
        Y /= std
        betas /= std

    return X, Y, subsets, betas, Sigma


# ========================================================================
#  Ground-truth SHAP-1 values for the synthetic data
# ========================================================================
def compute_true_shap1(X, subsets, betas, Sigma):
    """
    For multilinear f on correlated Gaussians, the SHAP-1 value for feature i
    is:  phi_i(x) = sum_{S containing i}  tilde_f_S(x_S) / |S|
    where tilde_f_S is the purified ANOVA component.

    For the simple correlated-pairs structure and multilinear f,
    we can compute the conditional expectations analytically.
    
    We use a brute-force permutation-sampling estimator as ground truth
    (expensive but correct for moderate d).
    """
    n, d = X.shape
    n_perm = 200  # enough for d=10
    rng = np.random.default_rng(0)

    shap_values = np.zeros((n, d), dtype=np.float32)
    
    # precompute the mean vector (zero for our data)
    mu = np.zeros(d)
    Sigma_inv = np.linalg.inv(Sigma)

    def f_eval(x_batch):
        """evaluate f on (batch, d)"""
        out = np.zeros(x_batch.shape[0], dtype=np.float32)
        for beta, S in zip(betas, subsets):
            prod = np.ones(x_batch.shape[0], dtype=np.float32)
            for i in S:
                prod *= x_batch[:, i]
            out += beta * prod
        return out

    def conditional_expectation(x_row, S_mask):
        """
        E[f(x) | x_S] by sampling missing features from p(x_Sc | x_S).
        S_mask: boolean array of length d, True = observed.
        """
        S_idx = np.where(S_mask)[0]
        Sc_idx = np.where(~S_mask)[0]
        
        if len(Sc_idx) == 0:
            return f_eval(x_row.reshape(1, -1))[0]
        if len(S_idx) == 0:
            # return E[f]
            n_mc = 100
            samples = rng.multivariate_normal(mu, Sigma, size=n_mc).astype(np.float32)
            return f_eval(samples).mean()

        # conditional Gaussian:  x_Sc | x_S ~ N(mu_cond, Sigma_cond)
        Sigma_SS = Sigma[np.ix_(S_idx, S_idx)]
        Sigma_ScSc = Sigma[np.ix_(Sc_idx, Sc_idx)]
        Sigma_ScS = Sigma[np.ix_(Sc_idx, S_idx)]
        
        Sigma_SS_inv = np.linalg.inv(Sigma_SS)
        mu_cond = Sigma_ScS @ Sigma_SS_inv @ (x_row[S_idx] - mu[S_idx]) + mu[Sc_idx]
        Sigma_cond = Sigma_ScSc - Sigma_ScS @ Sigma_SS_inv @ Sigma_ScS.T
        # ensure PSD
        Sigma_cond = (Sigma_cond + Sigma_cond.T) / 2 + np.eye(len(Sc_idx)) * 1e-8

        n_mc = 50
        samples_Sc = rng.multivariate_normal(mu_cond, Sigma_cond, size=n_mc).astype(np.float32)
        full_samples = np.tile(x_row, (n_mc, 1)).astype(np.float32)
        full_samples[:, Sc_idx] = samples_Sc
        return f_eval(full_samples).mean()

    # permutation sampling for SHAP
    for idx in tqdm(range(n), desc="Computing true SHAP"):
        x_row = X[idx]
        for _ in range(n_perm):
            perm = rng.permutation(d)
            S_mask = np.zeros(d, dtype=bool)
            prev_val = conditional_expectation(x_row, S_mask)
            for feat in perm:
                S_mask[feat] = True
                cur_val = conditional_expectation(x_row, S_mask)
                shap_values[idx, feat] += (cur_val - prev_val)
                prev_val = cur_val
        shap_values[idx] /= n_perm

    return shap_values


# ========================================================================
#  Training loop that records MSE vs true SHAP per epoch
# ========================================================================
def train_and_eval(model, train_loader, true_shap_test, X_test_tensor,
                   num_epochs=300, lr=5e-3, d=10):
    """
    Train a model (FastSHAP or InstaSHAP) and record per-epoch MSE 
    against ground-truth SHAP-1 values.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    sampler = ShapleySampler(num_features=d)
    
    mse_history = []
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            S_batch = sampler.sample(x_batch.shape[0], paired_sampling=True).to(device)
            optimizer.zero_grad()
            out = model(x_batch, S_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)

        # evaluate SHAP MSE on test set
        model.eval()
        with torch.no_grad():
            pred_shap = model.get_shapley_values(X_test_tensor.to(device))
            # For InstaSHAP: pred_shap has shape (n, num_interactions)
            # We need to map back to per-feature SHAP values
            if pred_shap.shape[1] != d:
                # InstaSHAP outputs per-interaction values; aggregate to per-feature
                tm = model.transform_matrix.cpu().numpy()   # (d, num_inter)
                pred_shap_np = pred_shap.cpu().numpy()      # (n, num_inter)
                per_feature = np.zeros((pred_shap_np.shape[0], d), dtype=np.float32)
                for j in range(tm.shape[1]):
                    feats_in_inter = np.where(tm[:, j] == 1)[0]
                    n_feats = len(feats_in_inter)
                    for fi in feats_in_inter:
                        per_feature[:, fi] += pred_shap_np[:, j] / n_feats
                pred_shap_np = per_feature
            else:
                pred_shap_np = pred_shap.cpu().numpy()

            mse = ((pred_shap_np - true_shap_test) ** 2).mean()
            mse_history.append(mse)
        model.train()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}  train_loss={epoch_loss/len(train_loader.dataset):.5f}  shap_mse={mse:.5f}")

    return mse_history


# ########################################################################
#  EXPERIMENT 1 :  10D Synthetic  (Figure 3)
# ########################################################################
def run_synthetic_experiment():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: 10D Synthetic  (reproducing Figure 3)")
    print("=" * 60)

    d = 10
    n_train = 5000
    n_test = 500       # small for fast ground-truth SHAP computation
    num_epochs = 300
    batch_size = 256

    configs = [
        (1, 0.0),
        (1, 0.707),
        (2, 0.0),
        (2, 0.707),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    for ax_idx, (k_star, rho) in enumerate(configs):
        print(f"\n--- k*={k_star}, rho={rho} ---")

        # generate data
        X_train, Y_train, subsets, betas, Sigma = generate_synthetic_data(
            n_train, d=d, k_star=k_star, rho=rho, seed=42)
        X_test, Y_test, _, _, _ = generate_synthetic_data(
            n_test, d=d, k_star=k_star, rho=rho, seed=99)
        # use same betas for test
        Y_test = np.zeros(n_test, dtype=np.float32)
        for beta, S in zip(betas, subsets):
            prod = np.ones(n_test, dtype=np.float32)
            for i in S:
                prod *= X_test[:, i]
            Y_test += beta * prod

        # ground-truth SHAP
        print("Computing ground-truth SHAP values on test set...")
        true_shap = compute_true_shap1(X_test, subsets, betas, Sigma)

        # torch tensors
        X_tr_t = torch.from_numpy(X_train).float()
        Y_tr_t = torch.from_numpy(Y_train).float().unsqueeze(1)
        X_te_t = torch.from_numpy(X_test).float()
        train_ds = TensorDataset(X_tr_t, Y_tr_t)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # ---- FastSHAP ----
        print("Training FastSHAP...")
        fastshap = FastSHAP(d, hidden=128).to(device)
        mse_fast = train_and_eval(fastshap, train_loader, true_shap, X_te_t,
                                  num_epochs=num_epochs, lr=5e-3, d=d)

        # ---- InstaSHAP ----
        # For synthetic: we *know* the true interactions = the subsets used to generate data
        # plus all singletons (order 1 always included)
        interactions_list = [()]   # index 0 = empty (phi_0 placeholder)
        # add singletons
        for i in range(d):
            interactions_list.append((i,))
        # add higher-order subsets if k_star > 1
        if k_star >= 2:
            for combo in combinations(range(d), 2):
                interactions_list.append(combo)

        transform_matrix = torch.zeros(d, len(interactions_list) - 1)
        for j in range(1, len(interactions_list)):
            for feat_idx in interactions_list[j]:
                transform_matrix[feat_idx, j - 1] = 1.0

        print("Training InstaSHAP...")
        instashap = InstaSHAP(interactions_list, transform_matrix, device=device).to(device)
        mse_insta = train_and_eval(instashap, train_loader, true_shap, X_te_t,
                                   num_epochs=num_epochs, lr=5e-3, d=d)

        # ---- Plot ----
        ax = axes[ax_idx]
        ax.semilogy(mse_fast, label="FastSHAP", linestyle="--")
        ax.semilogy(mse_insta, label="InstaSHAP", linestyle="-")
        ax.set_title(f"SHAP-1 Error (k*={k_star}, ρ={rho:.3f})")
        ax.set_xlabel("Number of Epochs")
        ax.set_ylabel("MSE Error (log)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig3_synthetic.png", dpi=150)
    plt.show()
    print("Saved fig3_synthetic.png")


# ########################################################################
#  EXPERIMENT 2 :  Bike Sharing  (Section 6.2)
# ########################################################################
def run_bike_sharing_experiment():
    """
    Requires the bike sharing dataset at ../data/bike_sharing/hour.csv
    Trains MLP, GAM-1 (FastSHAP), and InstaSHAP-GAM.
    Reports R2 error and plots hour×workday interaction.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Bike Sharing  (Section 6.2)")
    print("=" * 60)

    # Try to load the data
    try:
        from src.dataloader import CustomBikeDataset
        dataset_obj = CustomBikeDataset(root_dir="../data/", dataset_name="bike_sharing", seed=37)
    except Exception as e:
        print(f"Could not load bike sharing data: {e}")
        print("Skipping bike sharing experiment. Place hour.csv in ../data/bike_sharing/")
        return

    dataset_obj.shuffle_and_split_trnval(trnval_shuffle_seed=37)
    trnX, trnY, valX, valY = dataset_obj.pull_trnval_data()
    tstX, tstY = dataset_obj.tstX, dataset_obj.tstY

    d = trnX.shape[1]  # 13 features

    X_tr = torch.from_numpy(trnX).float()
    Y_tr = torch.from_numpy(trnY).float()
    y_mean = Y_tr.mean()
    Y_centered = Y_tr - y_mean

    train_ds = TensorDataset(X_tr, Y_centered)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # --- 1. FastSHAP (GAM-1 equivalent) ---
    print("Training FastSHAP (GAM-1)...")
    fastshap = FastSHAP(d, hidden=128).to(device)
    opt = optim.Adam(fastshap.parameters(), lr=5e-3)
    sampler = ShapleySampler(d)
    criterion = nn.MSELoss()
    for epoch in range(100):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            S = sampler.sample(xb.shape[0], paired_sampling=True).to(device)
            opt.zero_grad()
            loss = criterion(fastshap(xb, S), yb)
            loss.backward()
            opt.step()

    # --- 2. InstaSHAP: use SIAN to discover interactions ---
    from src.instashap import surrogate
    from sian.models import TrainingArgs

    results_path = "./results/"
    exp_folder = results_path + "bike_experiment/"
    os.makedirs(exp_folder, exist_ok=True)

    mlp_args = TrainingArgs(batch_size=32, number_of_epochs=10, learning_rate=5e-3, device=device)
    mlp_args.model_config.net_name = "MLP"
    mlp_args.model_config.sizes = [-1, 128, 256, 128, -1]
    mlp_args.model_config.is_masked = True
    mlp_args.saving_settings.exp_folder = exp_folder

    my_surrogate = surrogate(mlp_args=mlp_args, dataset_obj=dataset_obj,
                             max_number_of_rounds=18, order=3)
    interactions = my_surrogate.get_interactions(device=device)
    transform_matrix = my_surrogate.get_transform_matrix()

    print(f"SIAN discovered interactions: {interactions}")

    print("Training InstaSHAP with discovered interactions...")
    instashap = InstaSHAP(interactions, transform_matrix, device=device).to(device)
    instashap.train_instaSHAP(train_loader, num_epochs=100, lr=5e-3, device=device)

    # --- Evaluate R2 ---
    def eval_r2(model, X_np, Y_np, label):
        model.eval()
        Xt = torch.from_numpy(X_np).float().to(device)
        Yt = torch.from_numpy(Y_np).float().squeeze()
        with torch.no_grad():
            full_mask = torch.ones_like(Xt)
            preds = model(Xt, full_mask).cpu() + y_mean
        ss_res = ((Yt - preds.squeeze()) ** 2).sum()
        ss_tot = ((Yt - Yt.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        r2_err = (1 - r2) * 100
        print(f"  {label}:  R2 = {r2:.4f},  R2 error = {r2_err:.2f}%")
        return r2_err.item()

    print("\n--- Test Set R2 ---")
    eval_r2(fastshap, tstX, tstY, "FastSHAP (GAM-1)")
    eval_r2(instashap, tstX, tstY, "InstaSHAP-GAM")

    # --- Plot hour×workday interaction ---
    print("Plotting hour×workday interaction heatmap...")
    instashap.eval()

    # Find the hour(4)×workday(7) interaction in discovered interactions
    hw_inter_idx = None
    for idx_i in range(1, len(interactions)):
        inter = interactions[idx_i]
        if set(inter) == {4, 7}:
            hw_inter_idx = idx_i - 1  # 0-based index into model list
            break

    if hw_inter_idx is None:
        print("hour×workday interaction not discovered by SIAN. Plotting all 1D SHAP values instead.")
        # fallback: plot per-feature SHAP values
        X_te = torch.from_numpy(tstX).float().to(device)
        with torch.no_grad():
            shap_vals = instashap.get_shapley_values(X_te).cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = dataset_obj.get_readable_labels()
        importances = np.abs(shap_vals).mean(axis=0)
        # only plot first d singletons
        imp_1d = importances[:d]
        order = np.argsort(imp_1d)[::-1]
        ax.barh([labels.get(i, str(i)) for i in order], imp_1d[order])
        ax.set_title("Mean |SHAP| per feature (InstaSHAP)")
        plt.tight_layout()
        plt.savefig("fig4_bike_feature_importance.png", dpi=150)
        plt.show()
    else:
        phi_hw = instashap.model[hw_inter_idx]
        hours = np.arange(0, 24, dtype=np.float32)
        workdays = np.array([0.0, 1.0], dtype=np.float32)
        grid = np.array([[h, w] for h in hours for w in workdays], dtype=np.float32)
        grid_t = torch.from_numpy(grid).float().to(device)
        with torch.no_grad():
            vals = phi_hw(grid_t).cpu().numpy().squeeze()
        vals_2d = vals.reshape(24, 2)
        fig, ax = plt.subplots(figsize=(8, 5))
        for w_idx, w_label in enumerate(["Weekend/Holiday", "Workday"]):
            ax.plot(hours, vals_2d[:, w_idx], label=w_label, marker='o', markersize=3)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("φ_{hour,workday}")
        ax.set_title("Bike Sharing: Hour × Workday Interaction (InstaSHAP)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("fig4_bike_interaction.png", dpi=150)
        plt.show()
        print("Saved fig4_bike_interaction.png")


if __name__ == "__main__":
    run_synthetic_experiment()
    run_bike_sharing_experiment()

# %%
