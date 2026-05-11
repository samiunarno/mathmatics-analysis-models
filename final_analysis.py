"""
final_analysis.py  —  LAYOFF-NET Full Analysis
Generates ALL paper figures + saves trained model for the API.
"""
import warnings; warnings.filterwarnings('ignore')
import subprocess, sys, os, pickle

needed = ['pandas','numpy','matplotlib','seaborn','scikit-learn','openpyxl','scipy']
for p in needed:
    try: __import__(p.replace('-','_'))
    except ImportError: subprocess.check_call([sys.executable,'-m','pip','install',p,'-q'])

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score,
    accuracy_score, average_precision_score, ConfusionMatrixDisplay
)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style='whitegrid')

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = BASE  # save pngs in same dir

print("="*60); print("LAYOFF-NET — Full Analysis Pipeline"); print("="*60)

# ── 1. Load ──────────────────────────────────────────────────
train = pd.read_excel(os.path.join(BASE, 'train.xlsx'))
test  = pd.read_excel(os.path.join(BASE, 'test.xlsx'))
TARGET = 'layoff_happened'
CAT = ['industry', 'country']
NUM = ['funding_amount', 'employee_count', 'growth_rate', 'valuation']
FEAT = CAT + NUM
print(f"\nTrain: {train.shape}  |  Test: {test.shape}")
print(f"Layoff rate: {train[TARGET].mean():.2%}\n")

# ── 2. Prep ───────────────────────────────────────────────────
def prep(df, enc=None, fit=True):
    df = df.copy(); enc = enc or {}
    for col in CAT:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            enc[col] = le
        else:
            le = enc[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in set(le.classes_) else -1)
    if fit:
        imp = SimpleImputer(strategy='median')
        df[NUM] = imp.fit_transform(df[NUM]); enc['imp'] = imp
    else:
        df[NUM] = enc['imp'].transform(df[NUM])
    for col in ['funding_amount','employee_count','valuation']:
        df[col] = np.log1p(df[col].clip(lower=0))
    if fit:
        sc = StandardScaler()
        df[NUM] = sc.fit_transform(df[NUM]); enc['sc'] = sc
    else:
        df[NUM] = enc['sc'].transform(df[NUM])
    return df[FEAT], enc

X, ENC = prep(train[FEAT], fit=True)
y = train[TARGET]
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── 3. Train all models ───────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=2,
                                                   class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=400, max_depth=5,
                                                       learning_rate=0.04, subsample=0.8, random_state=42),
    'Extra Trees':         ExtraTreesClassifier(n_estimators=500, class_weight='balanced',
                                                random_state=42, n_jobs=-1),
}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
print(f"  {'Model':<22} {'ValAUC':>7} {'CVAUC':>7} {'Acc':>6} {'F1':>6}")
print("  " + "-"*55)
for name, m in models.items():
    m.fit(Xtr, ytr)
    prb = m.predict_proba(Xv)[:,1]; prd = m.predict(Xv)
    vauc = roc_auc_score(yv, prb)
    cvauc = cross_val_score(m, X, y, cv=cv5, scoring='roc_auc', n_jobs=-1).mean()
    acc = accuracy_score(yv, prd); f1 = f1_score(yv, prd)
    results[name] = dict(model=m, probs=prb, preds=prd, vauc=vauc, cvauc=cvauc, acc=acc, f1=f1)
    print(f"  {name:<22} {vauc:>7.4f} {cvauc:>7.4f} {acc:>6.4f} {f1:>6.4f}")

best_name = max(results, key=lambda k: results[k]['vauc'])
best = results[best_name]; best_model = best['model']
print(f"\n  Best: {best_name}  AUC={best['vauc']:.4f}")

# ── 4. Threshold optimisation ─────────────────────────────────
def fbeta(p, r, b): d = b**2*p + r; return (1+b**2)*p*r/d if d>0 else 0.

thresholds = np.arange(0.25, 0.75, 0.005)
best_tau, best_f1 = 0.5, -1
for tau in thresholds:
    pd_ = (best['probs'] >= tau).astype(int)
    f = f1_score(yv, pd_, zero_division=0)
    if f > best_f1: best_f1=f; best_tau=tau

opt_preds = (best['probs'] >= best_tau).astype(int)
print(f"\n  Optimal τ* = {best_tau:.3f}  →  F1 = {best_f1:.4f}")

# ── 5. Industry Stratified Ensemble ──────────────────────────
MIN_S = 50
tc = train.copy()
rare = tc['industry'].value_counts()[lambda s: s < MIN_S].index
tc['strat'] = tc['industry'].apply(lambda x: 'Other' if x in rare else x)
strat_models = {}
for s in tc['strat'].unique():
    sub = tc[tc['strat']==s]
    if sub[TARGET].nunique() < 2: continue
    Xs, es = prep(sub[FEAT], fit=True)
    ys = sub[TARGET]
    m2 = RandomForestClassifier(n_estimators=300, max_depth=10,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    m2.fit(Xs, ys); strat_models[s] = (m2, es)

_, vdf = train_test_split(train, test_size=0.2, random_state=42, stratify=train[TARGET])
vdf = vdf.copy()
vdf['strat'] = vdf['industry'].apply(lambda x: 'Other' if x in rare else x)
gX, _ = prep(vdf[FEAT], enc=ENC, fit=False)
gp = best_model.predict_proba(gX)[:,1]
alpha = 0.70; blended = np.zeros(len(vdf))
for i,(_, row) in enumerate(vdf.iterrows()):
    s = row['strat']
    if s in strat_models:
        m2, es = strat_models[s]
        rdf = pd.DataFrame([row[FEAT]])
        rX, _ = prep(rdf, enc=es, fit=False)
        sp = m2.predict_proba(rX)[0,1]
        blended[i] = alpha*sp + (1-alpha)*gp[i]
    else: blended[i] = gp[i]

ens_y = vdf[TARGET].values
ens_auc = roc_auc_score(ens_y, blended)
best_ens_tau, best_ens_f1 = 0.5, -1
for tau in thresholds:
    f = f1_score(ens_y, (blended>=tau).astype(int), zero_division=0)
    if f > best_ens_f1: best_ens_f1=f; best_ens_tau=tau
ens_preds = (blended >= best_ens_tau).astype(int)

print(f"  ICECB AUC={ens_auc:.4f}  F1@τ*={best_ens_f1:.4f}  (τ*={best_ens_tau:.3f})")

# ── 6. Full classification report ────────────────────────────
print("\n" + "="*60)
print("FINAL CLASSIFICATION REPORT — ICECB + EDTO")
print("="*60)
report = classification_report(ens_y, ens_preds,
    target_names=['No Layoff','Layoff'], digits=4)
print(report)

with open(os.path.join(OUT, 'final_report.txt'), 'w') as f:
    f.write("LAYOFF-NET — ICECB + EDTO Final Classification Report\n")
    f.write("="*60 + "\n")
    f.write(f"AUC  : {ens_auc:.4f}\n")
    f.write(f"Tau* : {best_ens_tau:.3f}\n")
    f.write(f"F1   : {best_ens_f1:.4f}\n\n")
    f.write(report)

# ── 7. Save model artifacts ───────────────────────────────────
artifacts = {
    'model': best_model, 'enc': ENC, 'strat_models': strat_models,
    'rare_inds': list(rare), 'alpha': alpha, 'tau': best_ens_tau,
    'feat_cols': FEAT, 'cat_cols': CAT, 'num_cols': NUM,
    'ind_classes': list(train['industry'].unique()),
    'cty_classes': list(train['country'].unique()),
}
with open(os.path.join(BASE, 'layoff_model.pkl'), 'wb') as f:
    pickle.dump(artifacts, f)
print("\n✓ Saved layoff_model.pkl")

# ──────────────────────────────────────────────────────────────
# FIGURE GENERATION
# ──────────────────────────────────────────────────────────────

# Fig A: Comprehensive model comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors4 = ['#3498db','#e74c3c','#2ecc71','#f39c12']
names = list(results.keys())

# A1: Val AUC bar
aucs = [results[n]['vauc'] for n in names]
bars = axes[0,0].bar(names, aucs, color=colors4, edgecolor='white', linewidth=1.5)
for bar, v in zip(bars, aucs):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, v+0.003, f'{v:.4f}',
                   ha='center', fontsize=9, fontweight='bold')
axes[0,0].set_title('Validation AUC Comparison', fontsize=11, fontweight='bold')
axes[0,0].set_ylim(0.55, 0.70); axes[0,0].tick_params(axis='x', rotation=20)
axes[0,0].axhline(0.5, color='gray', linestyle='--', lw=1, label='Random baseline')
axes[0,0].legend(fontsize=8)

# A2: ROC curves
for (n, res), c in zip(results.items(), colors4):
    fpr, tpr, _ = roc_curve(yv, res['probs'])
    axes[0,1].plot(fpr, tpr, color=c, lw=2, label=f"{n.split()[0]} ({res['vauc']:.3f})")
axes[0,1].plot([0,1],[0,1],'k--',lw=1)
axes[0,1].fill_between(*roc_curve(yv,best['probs'])[:2], alpha=0.07, color='#e74c3c')
axes[0,1].set_title('ROC Curves — All Models', fontsize=11, fontweight='bold')
axes[0,1].set_xlabel('FPR'); axes[0,1].set_ylabel('TPR')
axes[0,1].legend(fontsize=8)

# A3: Threshold sensitivity
betas_plt = {1.0:'#3498db', 1.5:'#e74c3c', 2.0:'#2ecc71'}
for b, c in betas_plt.items():
    vals = []
    for tau in thresholds:
        pd_ = (best['probs']>=tau).astype(int)
        p = precision_score(yv,pd_,zero_division=0); r = recall_score(yv,pd_,zero_division=0)
        vals.append(fbeta(p,r,b))
    axes[0,2].plot(thresholds, vals, color=c, lw=2, label=f'β={b}')
axes[0,2].axvline(best_tau, color='black', linestyle='--', lw=1.5, label=f'τ*={best_tau:.2f}')
axes[0,2].set_title('F_β vs Threshold (Baseline RF)', fontsize=11, fontweight='bold')
axes[0,2].set_xlabel('τ'); axes[0,2].set_ylabel('F_β Score')
axes[0,2].legend(fontsize=9)

# A4: Confusion matrix — Baseline
cm1 = confusion_matrix(yv, best['preds'])
ConfusionMatrixDisplay(cm1, display_labels=['No Layoff','Layoff']).plot(
    ax=axes[1,0], colorbar=False, cmap='Blues')
axes[1,0].set_title(f'Baseline RF (τ=0.50)\nAcc={best["acc"]:.4f}  F1={best["f1"]:.4f}',
                    fontsize=10, fontweight='bold')

# A5: Confusion matrix — ICECB+EDTO
cm2 = confusion_matrix(ens_y, ens_preds)
ConfusionMatrixDisplay(cm2, display_labels=['No Layoff','Layoff']).plot(
    ax=axes[1,1], colorbar=False, cmap='Greens')
acc2 = accuracy_score(ens_y, ens_preds)
axes[1,1].set_title(f'ICECB+EDTO (τ*={best_ens_tau:.2f})\nAcc={acc2:.4f}  F1={best_ens_f1:.4f}',
                    fontsize=10, fontweight='bold')

# A6: Progressive improvement
configs = ['Baseline\nRF τ=0.50', 'Baseline\nRF EDTO', 'ICECB\nτ=0.50', 'ICECB\n+EDTO']
f1_vals = [
    f1_score(yv, best['preds']),
    f1_score(yv, opt_preds),
    f1_score(ens_y, (blended>=0.5).astype(int)),
    best_ens_f1
]
auc_vals = [best['vauc'], best['vauc'], ens_auc, ens_auc]
x = np.arange(len(configs)); w = 0.35
axes[1,2].bar(x-w/2, f1_vals, w, label='F₁ Score', color='#3498db', edgecolor='white')
axes[1,2].bar(x+w/2, auc_vals, w, label='AUC', color='#e74c3c', edgecolor='white')
for i,(f,a) in enumerate(zip(f1_vals, auc_vals)):
    axes[1,2].text(i-w/2, f+0.01, f'{f:.3f}', ha='center', fontsize=8, fontweight='bold')
    axes[1,2].text(i+w/2, a+0.01, f'{a:.3f}', ha='center', fontsize=8, fontweight='bold')
axes[1,2].set_xticks(x); axes[1,2].set_xticklabels(configs, fontsize=9)
axes[1,2].set_ylim(0, 1.05); axes[1,2].legend(fontsize=9)
axes[1,2].set_title('Progressive Improvement', fontsize=11, fontweight='bold')

plt.suptitle('Fig. 6 — LAYOFF-NET Complete Model Evaluation Dashboard',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig6_full_evaluation.png'), dpi=150, bbox_inches='tight')
plt.close(); print("✓ fig6_full_evaluation.png")

# Fig B: Detailed metrics table visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Precision/Recall/F1 per class for ICECB
from sklearn.metrics import precision_recall_fscore_support
prec, rec, f1s, sup = precision_recall_fscore_support(ens_y, ens_preds, labels=[0,1])
metric_data = np.array([[prec[0],rec[0],f1s[0]], [prec[1],rec[1],f1s[1]]])
im = axes[0].imshow(metric_data, cmap='RdYlGn', vmin=0.3, vmax=1.0, aspect='auto')
axes[0].set_xticks([0,1,2]); axes[0].set_xticklabels(['Precision','Recall','F1-Score'], fontsize=11)
axes[0].set_yticks([0,1]); axes[0].set_yticklabels(['No Layoff\n(class 0)','Layoff\n(class 1)'], fontsize=11)
for i in range(2):
    for j,v in enumerate([prec[i],rec[i],f1s[i]]):
        axes[0].text(j, i, f'{v:.4f}', ha='center', va='center',
                     fontsize=14, fontweight='bold',
                     color='white' if v < 0.55 else '#1a252f')
axes[0].set_title('Per-Class Metrics — ICECB + EDTO', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=axes[0], shrink=0.7)

# Support & summary
ax2 = axes[1]; ax2.axis('off')
summary_rows = [
    ['Metric', 'Baseline RF\n(τ=0.50)', 'ICECB+EDTO\n(τ*=optimal)', 'Δ Improvement'],
    ['Accuracy', f'{best["acc"]:.4f}', f'{acc2:.4f}', f'+{(acc2-best["acc"])*100:.2f}%'],
    ['AUC', f'{best["vauc"]:.4f}', f'{ens_auc:.4f}', f'+{(ens_auc-best["vauc"])*100:.2f}%'],
    ['F₁ Score', f'{best["f1"]:.4f}', f'{best_ens_f1:.4f}', f'+{(best_ens_f1-best["f1"])*100:.2f}%'],
    ['Precision', f'{precision_score(yv,best["preds"]):.4f}', f'{prec[1]:.4f}', f'{(prec[1]-precision_score(yv,best["preds"]))*100:+.2f}%'],
    ['Recall', f'{recall_score(yv,best["preds"]):.4f}', f'{rec[1]:.4f}', f'{(rec[1]-recall_score(yv,best["preds"]))*100:+.2f}%'],
    ['Support (Layoff)', str(int(sup[1])), str(int(sup[1])), '—'],
]
colors_t = [['#2c3e50']*4] + [['#ecf0f1','#d5f5e3','#aed6f1','#fdebd0']]*6
tbl = ax2.table(cellText=summary_rows[1:], colLabels=summary_rows[0],
                cellLoc='center', loc='center',
                colWidths=[0.28,0.23,0.25,0.24])
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 2.0)
for (row,col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
    elif col == 3:
        cell.set_facecolor('#d5f5e3')
ax2.set_title('Performance Summary Table', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Fig. 7 — Per-Class Metrics & Comparative Summary', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig7_metrics_summary.png'), dpi=150, bbox_inches='tight')
plt.close(); print("✓ fig7_metrics_summary.png")

# Fig C: Feature importance + learning curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fi = best_model.feature_importances_
order = np.argsort(fi)
cols_c = ['#e74c3c' if FEAT[i]=='growth_rate' else '#3498db' for i in order]
axes[0].barh([FEAT[i] for i in order], fi[order], color=cols_c, edgecolor='white')
for i2, i in enumerate(order):
    axes[0].text(fi[i]+0.002, i2, f'{fi[i]:.3f}', va='center', fontsize=9)
axes[0].set_xlabel('Gini Importance'); axes[0].set_title('Feature Importance (RF)', fontsize=11, fontweight='bold')
axes[0].legend(handles=[
    mpatches.Patch(color='#e74c3c', label='Dominant (growth_rate)'),
    mpatches.Patch(color='#3498db', label='Other features')], fontsize=9)

tr_s, tr_sc, val_sc = learning_curve(best_model, X, y, cv=5, scoring='roc_auc',
                                      train_sizes=np.linspace(0.1,1.0,10), n_jobs=-1)
axes[1].plot(tr_s, tr_sc.mean(1), 'o-', color='#e74c3c', lw=2, label='Train AUC')
axes[1].plot(tr_s, val_sc.mean(1), 's-', color='#3498db', lw=2, label='CV AUC')
axes[1].fill_between(tr_s, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.1, color='#e74c3c')
axes[1].fill_between(tr_s, val_sc.mean(1)-val_sc.std(1), val_sc.mean(1)+val_sc.std(1), alpha=0.1, color='#3498db')
axes[1].set_xlabel('Training Samples'); axes[1].set_ylabel('ROC AUC')
axes[1].set_title('Learning Curve (Random Forest)', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)

plt.suptitle('Fig. 8 — Feature Importance & Learning Curve', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig8_importance_learning.png'), dpi=150, bbox_inches='tight')
plt.close(); print("✓ fig8_importance_learning.png")

# Test predictions
Xt, _ = prep(test[FEAT], enc=ENC, fit=False)
gp_t = best_model.predict_proba(Xt)[:,1]
test_c = test.copy()
test_c['strat'] = test_c['industry'].apply(lambda x: 'Other' if x in rare else x)
bl_t = np.zeros(len(test_c))
for i,(_, row) in enumerate(test_c.iterrows()):
    s = row['strat']
    if s in strat_models:
        m2,es = strat_models[s]
        rX,_ = prep(pd.DataFrame([row[FEAT]]), enc=es, fit=False)
        sp = m2.predict_proba(rX)[0,1]
        bl_t[i] = alpha*sp + (1-alpha)*gp_t[i]
    else: bl_t[i] = gp_t[i]
test_preds = (bl_t >= best_ens_tau).astype(int)
out_df = test.copy()
out_df['layoff_probability'] = bl_t.round(4)
out_df['layoff_prediction'] = test_preds
out_df['risk_level'] = pd.cut(bl_t, bins=[0,0.35,0.6,1.0], labels=['Low','Medium','High'])
out_df.to_excel(os.path.join(BASE,'final_predictions.xlsx'), index=False)
print(f"\n✓ final_predictions.xlsx  ({test_preds.sum()} flagged high-risk)")

print("\n" + "="*60)
print("ALL DONE — Generated:")
for f in ['fig6_full_evaluation.png','fig7_metrics_summary.png','fig8_importance_learning.png',
          'final_predictions.xlsx','final_report.txt','layoff_model.pkl']:
    path = os.path.join(OUT, f)
    print(f"  {'✓' if os.path.exists(path) else '✗'} {f}")
print("="*60)
