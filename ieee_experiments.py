"""
ieee_experiments.py
--------------------
Generates all novel results and figures for the IEEE paper:
  - shap_dominance_bar.png  (permutation-based feature dominance)
  - threshold_curve.png     (F-beta vs threshold)
  - dominance_table.txt     (dominance ratios)
  - ensemble_results.txt    (stratified ensemble AUC/F1)

Run: python3 ieee_experiments.py
"""

import subprocess, sys

needed = ["pandas","numpy","matplotlib","seaborn",
          "scikit-learn","openpyxl","scipy"]
for pkg in needed:
    try:
        __import__(pkg.replace("-","_"))
    except ImportError:
        print(f"  installing {pkg}...")
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_theme(style="whitegrid")

print("=" * 60)
print("IEEE Paper Experiment Suite")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
train = pd.read_excel("train.xlsx")
test  = pd.read_excel("test.xlsx")

TARGET   = "layoff_happened"
CAT_COLS = ["industry", "country"]
NUM_COLS = ["funding_amount", "employee_count", "growth_rate", "valuation"]
FEAT_COLS = CAT_COLS + NUM_COLS

print(f"\nTrain: {train.shape[0]} rows | Test: {test.shape[0]} rows")
print(f"Layoff rate: {train[TARGET].mean():.2%}\n")


# ─────────────────────────────────────────────────────────────
# 2. FEATURE PREPARATION
# ─────────────────────────────────────────────────────────────
def prepare(df, enc=None, fit=True):
    df = df.copy()
    enc = enc or {}

    for col in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            enc[col] = le
        else:
            le = enc[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in set(le.classes_) else -1)

    if fit:
        imp = SimpleImputer(strategy="median")
        df[NUM_COLS] = imp.fit_transform(df[NUM_COLS])
        enc["imp"] = imp
    else:
        df[NUM_COLS] = enc["imp"].transform(df[NUM_COLS])

    for col in ["funding_amount","employee_count","valuation"]:
        df[col] = np.log1p(df[col].clip(lower=0))

    if fit:
        sc = StandardScaler()
        df[NUM_COLS] = sc.fit_transform(df[NUM_COLS])
        enc["sc"] = sc
    else:
        df[NUM_COLS] = enc["sc"].transform(df[NUM_COLS])

    return df[FEAT_COLS], enc


X_all, ENC = prepare(train[FEAT_COLS], fit=True)
y_all = train[TARGET]

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

# ─────────────────────────────────────────────────────────────
# 3. TRAIN BASELINE RANDOM FOREST
# ─────────────────────────────────────────────────────────────
print("Training baseline Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10,
    class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

val_probs = rf.predict_proba(X_val)[:, 1]
val_preds = rf.predict(X_val)
baseline_auc = roc_auc_score(y_val, val_probs)
baseline_f1  = f1_score(y_val, val_preds)
print(f"  Baseline AUC = {baseline_auc:.4f}  |  F1 = {baseline_f1:.4f}")


# ─────────────────────────────────────────────────────────────
# 4. PERMUTATION-BASED FEATURE DOMINANCE  →  shap_dominance_bar.png
# ─────────────────────────────────────────────────────────────
print("\nComputing permutation feature importance ...")

perm = permutation_importance(
    rf, X_val, y_val,
    n_repeats=30, random_state=42,
    scoring="roc_auc", n_jobs=-1)

feat_names   = list(FEAT_COLS)
perm_means   = perm.importances_mean          # shape (n_features,)
perm_stds    = perm.importances_std

# Also collect built-in impurity importance for cross-check
impurity_imp = rf.feature_importances_

dominant_idx  = int(np.argmax(perm_means))
dominant_feat = feat_names[dominant_idx]

print(f"\n  Dominant feature : {dominant_feat}")
print(f"  Perm importance  : {perm_means[dominant_idx]:.4f} ± {perm_stds[dominant_idx]:.4f}")
print("\n  Dominance ratios D({}, other):".format(dominant_feat))

with open("dominance_table.txt", "w") as f:
    f.write("Feature                 Perm-Imp   Impurity-Imp   Dom-Ratio\n")
    f.write("-" * 58 + "\n")
    for j, feat in enumerate(feat_names):
        ratio = perm_means[dominant_idx] / max(perm_means[j], 1e-9)
        line  = f"  {feat:<22} {perm_means[j]:.4f}     {impurity_imp[j]:.4f}         {ratio:.2f}"
        print(line)
        f.write(f"{feat}\t{perm_means[j]:.4f}\t{impurity_imp[j]:.4f}\t{ratio:.2f}\n")
print("  → saved dominance_table.txt")

# ── Figure 1: Dominance bar chart with error bars ──────────────
order  = np.argsort(perm_means)
colors = ["#e74c3c" if feat_names[i] == dominant_feat else "#3498db"
          for i in order]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: permutation importance
axes[0].barh([feat_names[i] for i in order], perm_means[order],
             xerr=perm_stds[order], color=colors,
             edgecolor="white", linewidth=1.2,
             error_kw=dict(ecolor="#2c3e50", capsize=4, lw=1.5))
for idx_bar, i in enumerate(order):
    ratio = perm_means[dominant_idx] / max(perm_means[i], 1e-9)
    tag   = f" {perm_means[i]:.3f}"
    if i != dominant_idx:
        tag += f"  (D={ratio:.1f}×)"
    axes[0].text(perm_means[i] + perm_stds[i] + 0.001,
                 idx_bar, tag, va="center", fontsize=8.5)
axes[0].set_xlabel("Mean AUC Drop (Permutation)", fontsize=11)
axes[0].set_title("Permutation Feature Importance\n(proxy for SHAP global importance)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlim(0, perm_means.max() * 1.55)
axes[0].legend(handles=[
    plt.Rectangle((0,0),1,1, color="#e74c3c", label="Dominant feature"),
    plt.Rectangle((0,0),1,1, color="#3498db", label="Other features")],
    fontsize=9, loc="lower right")

# Right: impurity importance for comparison
order2  = np.argsort(impurity_imp)
colors2 = ["#e74c3c" if feat_names[i] == dominant_feat else "#95a5a6"
           for i in order2]
axes[1].barh([feat_names[i] for i in order2], impurity_imp[order2],
             color=colors2, edgecolor="white", linewidth=1.2)
for idx_bar, i in enumerate(order2):
    axes[1].text(impurity_imp[i] + 0.002, idx_bar,
                 f" {impurity_imp[i]:.3f}", va="center", fontsize=8.5)
axes[1].set_xlabel("Gini Impurity Importance", fontsize=11)
axes[1].set_title("Built-in (Gini) Feature Importance\n(cross-validation reference)",
                  fontsize=11, fontweight="bold")
axes[1].set_xlim(0, impurity_imp.max() * 1.35)

plt.suptitle("Feature Dominance Analysis — Random Forest Classifier",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("shap_dominance_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → saved shap_dominance_bar.png")


# ─────────────────────────────────────────────────────────────
# 5. THRESHOLD OPTIMIZATION  →  threshold_curve.png
# ─────────────────────────────────────────────────────────────
print("\nRunning threshold optimization ...")

def fbeta(prec, rec, beta):
    denom = beta**2 * prec + rec
    return (1 + beta**2) * prec * rec / denom if denom > 0 else 0.0

thresholds = np.arange(0.25, 0.75, 0.01)
betas = [1.0, 1.5, 2.0]
results = {b: [] for b in betas}

best_tau  = {}
best_fb   = {}

for b in betas:
    best_fb[b]  = -1
    best_tau[b] = 0.5
    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        fb = fbeta(p, r, b)
        results[b].append(fb)
        if fb > best_fb[b]:
            best_fb[b] = fb
            best_tau[b] = tau

print(f"\n  β=1.0  →  τ* = {best_tau[1.0]:.2f}  |  F1    = {best_fb[1.0]:.4f}")
print(f"  β=1.5  →  τ* = {best_tau[1.5]:.2f}  |  F1.5  = {best_fb[1.5]:.4f}")
print(f"  β=2.0  →  τ* = {best_tau[2.0]:.2f}  |  F2    = {best_fb[2.0]:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
colors_map = {1.0: "#3498db", 1.5: "#e74c3c", 2.0: "#2ecc71"}
for b in betas:
    ax.plot(thresholds, results[b], lw=2.2, color=colors_map[b],
            label=f"β = {b}  (τ* = {best_tau[b]:.2f})")
    ax.axvline(best_tau[b], color=colors_map[b], linestyle="--", alpha=0.5, lw=1.2)

ax.axvline(0.5, color="gray", linestyle=":", lw=1.5, label="Default τ = 0.50")
ax.set_xlabel("Decision Threshold τ", fontsize=12)
ax.set_ylabel("$F_\\beta$ Score", fontsize=12)
ax.set_title("Cost-Sensitive Threshold Optimization\n$F_\\beta$ Score vs Decision Threshold",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0.25, 0.75)
ax.set_ylim(0, max(max(v) for v in results.values()) * 1.15)
plt.tight_layout()
plt.savefig("threshold_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → saved threshold_curve.png")

# Print comparison table
tau_opt = best_tau[1.5]
p_def = precision_score(y_val, (val_probs >= 0.50).astype(int), zero_division=0)
r_def = recall_score   (y_val, (val_probs >= 0.50).astype(int), zero_division=0)
f_def = f1_score       (y_val, (val_probs >= 0.50).astype(int))

p_opt = precision_score(y_val, (val_probs >= tau_opt).astype(int), zero_division=0)
r_opt = recall_score   (y_val, (val_probs >= tau_opt).astype(int), zero_division=0)
f_opt = f1_score       (y_val, (val_probs >= tau_opt).astype(int))

print(f"\n  Threshold  | Precision | Recall | F1")
print(f"  τ = 0.50   | {p_def:.4f}    | {r_def:.4f} | {f_def:.4f}")
print(f"  τ = {tau_opt:.2f}   | {p_opt:.4f}    | {r_opt:.4f} | {f_opt:.4f}")


# ─────────────────────────────────────────────────────────────
# 6. INDUSTRY-STRATIFIED ENSEMBLE
# ─────────────────────────────────────────────────────────────
print("\nBuilding industry-stratified ensemble ...")

MIN_SAMPLES = 50
train_copy = train.copy()
ind_counts = train_copy["industry"].value_counts()
rare_inds  = ind_counts[ind_counts < MIN_SAMPLES].index
train_copy["industry_stratum"] = train_copy["industry"].apply(
    lambda x: "Other" if x in rare_inds else x)

strata     = train_copy["industry_stratum"].unique()
stratum_models = {}

for stratum in strata:
    mask = train_copy["industry_stratum"] == stratum
    sub  = train_copy[mask]
    if sub[TARGET].nunique() < 2:
        continue
    X_sub, enc_sub = prepare(sub[FEAT_COLS], fit=True)
    y_sub = sub[TARGET]
    m = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        class_weight="balanced", random_state=42, n_jobs=-1)
    m.fit(X_sub, y_sub)
    stratum_models[stratum] = (m, enc_sub)
    print(f"  Trained stratum '{stratum}' — {len(sub)} samples")

# Predict on validation set with ensemble
def ensemble_predict(df_raw, global_model, global_enc,
                     stratum_models, rare_inds, alpha=0.55):
    df = df_raw.copy()
    df["industry_stratum"] = df["industry"].apply(
        lambda x: "Other" if x in rare_inds else x)

    global_X, _ = prepare(df[FEAT_COLS], enc=global_enc, fit=False)
    global_p    = global_model.predict_proba(global_X)[:, 1]

    blended = np.zeros(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        stratum = row["industry_stratum"]
        gp      = global_p[i]
        if stratum in stratum_models:
            m, enc = stratum_models[stratum]
            row_df  = pd.DataFrame([row[FEAT_COLS]])
            row_X, _ = prepare(row_df, enc=enc, fit=False)
            sp = m.predict_proba(row_X)[0, 1]
            blended[i] = alpha * sp + (1 - alpha) * gp
        else:
            blended[i] = gp
    return blended

# Grid search alpha
val_raw = train.iloc[X_val.index] if hasattr(X_val, "index") else train.sample(len(X_val), random_state=42)
# Safer: reconstruct val set from train using same split indices
from sklearn.model_selection import train_test_split as tts
_, val_df = tts(train, test_size=0.2, random_state=42, stratify=train[TARGET])

best_alpha, best_ens_auc = 0.5, 0
for alpha in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]:
    ens_p = ensemble_predict(val_df, rf, ENC, stratum_models, rare_inds, alpha)
    auc   = roc_auc_score(val_df[TARGET], ens_p)
    if auc > best_ens_auc:
        best_ens_auc = auc
        best_alpha   = alpha

print(f"\n  Best α = {best_alpha}  |  Ensemble AUC = {best_ens_auc:.4f}")

ens_probs = ensemble_predict(val_df, rf, ENC, stratum_models, rare_inds, best_alpha)
ens_y     = val_df[TARGET].values

# Threshold optimization for ensemble
best_ens_tau = 0.5
best_ens_f1  = -1
for tau in thresholds:
    f = f1_score(ens_y, (ens_probs >= tau).astype(int), zero_division=0)
    if f > best_ens_f1:
        best_ens_f1  = f
        best_ens_tau = tau

ens_f1_default = f1_score(ens_y, (ens_probs >= 0.5).astype(int))
print(f"  Ensemble F1 @ τ=0.50  = {ens_f1_default:.4f}")
print(f"  Ensemble F1 @ τ={best_ens_tau:.2f}  = {best_ens_f1:.4f}")

# ─────────────────────────────────────────────────────────────
# 7. SUMMARY TABLE  →  ensemble_results.txt
# ─────────────────────────────────────────────────────────────
with open("ensemble_results.txt", "w") as f:
    f.write("Progressive Model Improvement\n")
    f.write("=" * 55 + "\n")
    f.write(f"{'Configuration':<40} {'AUC':>6} {'F1':>6}\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Baseline RF (τ=0.50)':<40} {baseline_auc:>6.4f} {baseline_f1:>6.4f}\n")
    f.write(f"{'Baseline RF (τ*={:.2f})'.format(tau_opt):<40} {baseline_auc:>6.4f} {f_opt:>6.4f}\n")
    f.write(f"{'Stratified Ensemble (τ=0.50)':<40} {best_ens_auc:>6.4f} {ens_f1_default:>6.4f}\n")
    f.write(f"{'Stratified Ensemble (τ*={:.2f})'.format(best_ens_tau):<40} {best_ens_auc:>6.4f} {best_ens_f1:>6.4f}\n")
print("\n  → saved ensemble_results.txt")

# ─────────────────────────────────────────────────────────────
# 8. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE — Generated files for IEEE paper:")
print("  shap_dominance_bar.png → Fig. 1 (dominance + impurity bars)")
print("  threshold_curve.png    → Fig. 2 (F-beta vs threshold)")
print("  dominance_table.txt    → Table III data")
print("  ensemble_results.txt   → Table IV data")
print("=" * 60)
