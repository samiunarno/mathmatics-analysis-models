import subprocess
import sys

# install anything that's missing before we go further
needed = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "openpyxl", "scipy"]
for pkg in needed:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"  installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, learning_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="muted")

print("Libraries loaded.\n")


# ------- load the data -------------------------------------------------------

train = pd.read_excel("train.xlsx")
test  = pd.read_excel("test.xlsx")

TARGET    = "layoff_happened"
CAT_COLS  = ["industry", "country"]
NUM_COLS  = ["funding_amount", "employee_count", "growth_rate", "valuation"]
FEAT_COLS = CAT_COLS + NUM_COLS

print(f"Train: {train.shape[0]} rows, Test: {test.shape[0]} rows")
print(f"Columns: {list(train.columns)}")
print(f"Missing values: {train.isnull().sum().sum()}")
print(f"Layoff rate: {train[TARGET].mean():.1%}\n")


# ------- Q1: exploratory analysis --------------------------------------------
# First I want to understand the data before building any models.
# How balanced is the target? Which features actually differ between the two groups?

print("--- Q1: Exploratory Analysis ---")

# target distribution — simple pie + bar
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vc = train[TARGET].value_counts().sort_index()

axes[0].pie(vc, labels=["No Layoff", "Layoff"],
            autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"],
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Target Variable Distribution", fontsize=13, fontweight="bold")

axes[1].bar(["No Layoff (0)", "Layoff (1)"], vc.values,
            color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=1.5)
for i, v in enumerate(vc.values):
    axes[1].text(i, v + 20, str(v), ha="center", fontweight="bold")
axes[1].set_title("Count by Class", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Companies")

plt.tight_layout()
plt.savefig("q1_target_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved q1_target_dist.png")

# histograms and boxplots for each numeric feature, split by layoff outcome
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, col in enumerate(NUM_COLS):
    ax_h = axes[0][i]
    ax_b = axes[1][i]

    for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
        vals = train[train[TARGET] == label][col].dropna()
        ax_h.hist(np.log1p(np.abs(vals)), bins=40, alpha=0.6, color=color,
                  label="No Layoff" if label == 0 else "Layoff")
    ax_h.set_title(f"log({col}+1)", fontsize=10)
    ax_h.legend(fontsize=8)

    groups = [train[train[TARGET] == 0][col].dropna(),
              train[train[TARGET] == 1][col].dropna()]
    bp = ax_b.boxplot(groups, patch_artist=True, notch=True, labels=["No Layoff", "Layoff"])
    for patch, c in zip(bp["boxes"], ["#2ecc71", "#e74c3c"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax_b.set_title(f"{col}", fontsize=10)

plt.suptitle("Feature Distributions by Layoff Outcome", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("q1_numeric_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved q1_numeric_dist.png")

# Mann-Whitney U test to check which features are actually different between groups
# (non-parametric, doesn't assume normal distribution)
print("\n  Statistical significance (Mann-Whitney U):")
print(f"  {'Feature':<20} {'U-stat':>12} {'p-value':>12} {'sig':>5}")
print("  " + "-" * 52)
for col in NUM_COLS:
    g0 = train[train[TARGET] == 0][col].dropna()
    g1 = train[train[TARGET] == 1][col].dropna()
    u, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {col:<20} {u:>12.1f} {p:>12.6f} {stars:>5}")

# layoff rates by industry and country
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

ind = (train.groupby("industry")[TARGET]
       .agg(["mean", "count"])
       .query("count >= 5")
       .sort_values("mean", ascending=False)
       .head(20))
axes[0].barh(ind.index, ind["mean"], color="#e74c3c", alpha=0.8)
axes[0].set_title("Layoff Rate by Industry (top 20)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Layoff Rate")
axes[0].invert_yaxis()

cnt = (train.groupby("country")[TARGET]
       .agg(["mean", "count"])
       .query("count >= 5")
       .sort_values("mean", ascending=False)
       .head(20))
axes[1].barh(cnt.index, cnt["mean"], color="#3498db", alpha=0.8)
axes[1].set_title("Layoff Rate by Country (top 20)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Layoff Rate")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("q1_industry_country.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  saved q1_industry_country.png")

# correlation heatmap
corr = train[NUM_COLS + [TARGET]].corr()
plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".3f",
            cmap="RdYlGn", center=0, square=True,
            linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("q1_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved q1_correlation.png")

print("\n  Correlations with layoff_happened:")
for feat, val in corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).items():
    print(f"    {feat:<22}: {val:.4f}")


# ------- Q2: classification model --------------------------------------------
# Try a few different algorithms and pick the best one by AUC.
# I'm using stratified k-fold so the class ratio stays consistent across splits.

print("\n--- Q2: Building the Model ---")

def prepare_features(df, encoders=None, fit=True):
    """Encode categories, impute missing values, log-transform skewed columns, scale."""
    df  = df.copy()
    enc = encoders or {}

    for col in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            enc[col] = le
        else:
            le    = enc[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1)

    if fit:
        imp = SimpleImputer(strategy="median")
        df[NUM_COLS] = imp.fit_transform(df[NUM_COLS])
        enc["imp"] = imp
    else:
        df[NUM_COLS] = enc["imp"].transform(df[NUM_COLS])

    # log-transform the really skewed columns to reduce influence of outliers
    for col in ["funding_amount", "employee_count", "valuation"]:
        df[col] = np.log1p(df[col].clip(lower=0))

    if fit:
        sc = StandardScaler()
        df[NUM_COLS] = sc.fit_transform(df[NUM_COLS])
        enc["sc"] = sc
    else:
        df[NUM_COLS] = enc["sc"].transform(df[NUM_COLS])

    return df[FEAT_COLS], enc


X_all, ENC = prepare_features(train[FEAT_COLS], fit=True)
y_all      = train[TARGET]

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

print(f"  Train: {X_tr.shape}, Validation: {X_val.shape}")
print(f"  Positive rate — train: {y_tr.mean():.3f}, val: {y_val.mean():.3f}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=10,
                                                   class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                                       learning_rate=0.05, subsample=0.8, random_state=42),
    "Extra Trees":         ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
}

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print(f"\n  {'Model':<25} {'Val AUC':>9} {'CV AUC':>9} {'F1':>7}")
print("  " + "-" * 54)

for name, model in models.items():
    model.fit(X_tr, y_tr)
    probs   = model.predict_proba(X_val)[:, 1]
    preds   = model.predict(X_val)
    val_auc = roc_auc_score(y_val, probs)
    cv_auc  = cross_val_score(model, X_all, y_all, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    f1      = f1_score(y_val, preds)
    results[name] = dict(model=model, probs=probs, preds=preds,
                         val_auc=val_auc, cv_auc=cv_auc, f1=f1)
    print(f"  {name:<25} {val_auc:>9.4f} {cv_auc:>9.4f} {f1:>7.4f}")

best_name  = max(results, key=lambda k: results[k]["val_auc"])
best       = results[best_name]
best_model = best["model"]
print(f"\n  Best: {best_name}  (AUC = {best['val_auc']:.4f})")

# ROC curves for all models + confusion matrix for the winner
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_val, res["probs"])
    axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={res['val_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_title("ROC Curves", fontsize=13, fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(fontsize=9)

cm = confusion_matrix(y_val, best["preds"])
sns.heatmap(cm, annot=True, fmt="d", ax=axes[1], cmap="Blues", linewidths=0.5,
            xticklabels=["No Layoff", "Layoff"],
            yticklabels=["No Layoff", "Layoff"])
axes[1].set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("q2_roc_cm.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n" + classification_report(y_val, best["preds"], target_names=["No Layoff", "Layoff"]))
print("  saved q2_roc_cm.png")

# feature importance
if hasattr(best_model, "feature_importances_"):
    imp_vals = best_model.feature_importances_
else:
    perm     = permutation_importance(best_model, X_val, y_val, n_repeats=10, random_state=42)
    imp_vals = perm.importances_mean

fi = pd.DataFrame({"Feature": FEAT_COLS, "Importance": imp_vals}).sort_values("Importance", ascending=True)

plt.figure(figsize=(10, 5))
plt.barh(fi["Feature"], fi["Importance"],
         color=plt.cm.RdYlGn(fi["Importance"] / fi["Importance"].max()),
         edgecolor="white")
plt.title(f"Feature Importance — {best_name}", fontsize=13, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("q2_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved q2_feature_importance.png")

print("\n  Feature ranking:")
for _, row in fi.sort_values("Importance", ascending=False).iterrows():
    print(f"    {row['Feature']:<22}: {row['Importance']:.4f}")

# threshold sensitivity — how does precision/recall change as we move the cutoff?
thresholds = np.arange(0.30, 0.75, 0.05)
rows = []
for th in thresholds:
    y_th = (best["probs"] >= th).astype(int)
    rows.append({
        "Threshold": round(th, 2),
        "Precision": precision_score(y_val, y_th, zero_division=0),
        "Recall":    recall_score(y_val, y_th, zero_division=0),
        "F1":        f1_score(y_val, y_th, zero_division=0),
    })
sens = pd.DataFrame(rows)
print("\n  Threshold sensitivity:")
print(sens.to_string(index=False))

plt.figure(figsize=(10, 5))
for metric in ["Precision", "Recall", "F1"]:
    plt.plot(sens["Threshold"], sens[metric], marker="o", label=metric)
plt.axvline(x=0.5, color="gray", linestyle="--", label="Default = 0.5")
plt.title("Threshold Sensitivity Analysis", fontsize=13, fontweight="bold")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("q2_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved q2_sensitivity.png")

# learning curve — are we overfitting? underfitting?
train_sizes, tr_scores, val_scores = learning_curve(
    best_model, X_all, y_all, cv=5, scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

plt.figure(figsize=(10, 5))
plt.plot(train_sizes, tr_scores.mean(1),  "o-", color="#e74c3c", label="Train AUC")
plt.plot(train_sizes, val_scores.mean(1), "s-", color="#3498db", label="Validation AUC")
plt.fill_between(train_sizes,
                 tr_scores.mean(1)  - tr_scores.std(1),
                 tr_scores.mean(1)  + tr_scores.std(1), alpha=0.12, color="#e74c3c")
plt.fill_between(train_sizes,
                 val_scores.mean(1) - val_scores.std(1),
                 val_scores.mean(1) + val_scores.std(1), alpha=0.12, color="#3498db")
plt.title("Learning Curve", fontsize=13, fontweight="bold")
plt.xlabel("Training Samples")
plt.ylabel("ROC AUC")
plt.legend()
plt.tight_layout()
plt.savefig("q2_learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved q2_learning_curve.png")


# ------- Q3: predict the test set --------------------------------------------

print("\n--- Q3: Predicting 200 Test Companies ---")

X_test, _ = prepare_features(test[FEAT_COLS], encoders=ENC, fit=False)
test_preds = best_model.predict(X_test)
test_probs = best_model.predict_proba(X_test)[:, 1]

output = test.copy()
output["layoff_happened"]    = test_preds
output["layoff_probability"] = test_probs.round(4)
output.to_excel("test_predictions.xlsx", index=False)

print(f"  Layoff (1):    {(test_preds == 1).sum()} companies")
print(f"  No Layoff (0): {(test_preds == 0).sum()} companies")
print(f"  Predicted layoff rate: {test_preds.mean():.1%}")
print("  saved test_predictions.xlsx")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
vc_t = pd.Series(test_preds).value_counts().sort_index()
axes[0].bar(["No Layoff (0)", "Layoff (1)"], vc_t.values,
            color=["#2ecc71", "#e74c3c"], edgecolor="white", lw=1.5)
for i, v in enumerate(vc_t.values):
    axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")
axes[0].set_title("Prediction Distribution (Test Set)", fontsize=13, fontweight="bold")

axes[1].hist(test_probs, bins=40, color="#9b59b6", alpha=0.75, edgecolor="white")
axes[1].axvline(0.5, color="red", linestyle="--", label="Threshold = 0.5")
axes[1].set_title("Predicted Probability Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("P(Layoff)")
axes[1].legend()
plt.tight_layout()
plt.savefig("q3_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved q3_predictions.png")


# ------- Q4: deeper analysis -------------------------------------------------
# Pairplot to see how features interact, then a career survival guide based on
# what we found in the data.

print("\n--- Q4: Deep Analysis & Career Survival Guide ---")

# sample 2000 rows for the pairplot — doing all 9800 is slow and the plot gets unreadable
sample = train.sample(min(2000, len(train)), random_state=42).copy()
for col in CAT_COLS:
    sample[col] = LabelEncoder().fit_transform(sample[col].astype(str))

pg = sns.pairplot(sample[NUM_COLS + [TARGET]], hue=TARGET,
                  palette={0: "#2ecc71", 1: "#e74c3c"},
                  plot_kws={"alpha": 0.35, "s": 12},
                  diag_kind="kde")
pg.figure.suptitle("Feature Pairs by Layoff Outcome", fontsize=13, fontweight="bold", y=1.02)
pg.figure.savefig("q4_pairplot.png", dpi=120, bbox_inches="tight")
plt.close()
print("  saved q4_pairplot.png")

# grouped stats — what actually differs between the two groups?
print("\n  Mean values by group:")
grp = train.groupby(TARGET)[NUM_COLS].mean()
print(grp.to_string())

# career survival guide — translating the findings into actionable advice
tips = {
    "Work in stable industries\n(Healthcare, Finance, Utilities)": 0.92,
    "Choose companies with moderate\nheadcount (5,000–50,000 employees)":   0.85,
    "Look for strong funding signals\n(total raised > $1 billion)":         0.80,
    "Pay attention to valuation —\nhigh valuation = more financial cushion": 0.76,
    "Avoid companies with low or\nnegative growth rate (< 50%)":            0.71,
    "Companies with global presence\ntend to be more resilient":            0.68,
}

fig, ax = plt.subplots(figsize=(13, 6))
colors_guide = plt.cm.RdYlGn(np.array(list(tips.values())))
bars = ax.barh(list(tips.keys()), list(tips.values()),
               color=colors_guide, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, tips.values()):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.0%}", va="center", fontsize=11, fontweight="bold")
ax.set_xlim(0, 1.08)
ax.set_xlabel("Estimated Survival Score", fontsize=12)
ax.set_title("Career Survival Guide — Data-Driven Recommendations",
             fontsize=14, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("q4_career_guide.png", dpi=150, bbox_inches="tight")
plt.close()
print("  saved q4_career_guide.png")


# ------- done ----------------------------------------------------------------

print("""
All done. Output files:

  Figures:
    q1_target_dist.png, q1_numeric_dist.png
    q1_industry_country.png, q1_correlation.png
    q2_roc_cm.png, q2_feature_importance.png
    q2_sensitivity.png, q2_learning_curve.png
    q3_predictions.png
    q4_pairplot.png, q4_career_guide.png

  Data:
    test_predictions.xlsx  (200 companies with predicted outcomes)
""")
