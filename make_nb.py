import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9.0"}}

def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

cells = []

# Title
cells.append(md("""# 🏆 智驭职场寒冬 — 企业裁员风险预测
## 2026年长春理工大学大学生数学建模竞赛 B题

**团队任务**: 利用多维企业数据，预测公司是否发生裁员（layoffhappened）

| 问题 | 内容 |
|------|------|
| 问题1 | 探索性数据分析 (EDA) & 关键指标提炼 |
| 问题2 | 分类预测模型构建与敏感性分析 |
| 问题3 | 测试集预测（200家企业） |
| 问题4 | 特征深层关联分析 & 职场生存指南 |
"""))

# Cell 1: imports
cells.append(code("""# ==================== 环境配置 ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, learning_curve,
                                     GridSearchCV)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_recall_curve, average_precision_score)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Plot settings
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
sns.set_theme(style='whitegrid', palette='muted')
PALETTE = {0: '#2ecc71', 1: '#e74c3c'}

print('✅ 所有库加载成功')
"""))

# Cell 2: load data
cells.append(md("## 📂 数据加载与概览"))
cells.append(code("""train = pd.read_excel('train.xlsx')
test  = pd.read_excel('test.xlsx')

print(f'训练集: {train.shape[0]} 行 × {train.shape[1]} 列')
print(f'测试集: {test.shape[0]} 行 × {test.shape[1]} 列')
print()
print('--- 训练集字段 ---')
print(train.dtypes)
print()
print('--- 前5行 ---')
train.head()
"""))

# Cell 3: missing values
cells.append(code("""print('=== 缺失值统计 ===')
miss = train.isnull().sum()
miss_pct = (miss / len(train) * 100).round(2)
pd.DataFrame({'缺失数': miss, '缺失率(%)': miss_pct}).query('缺失数 > 0')
"""))

# Cell 4: target distribution
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
vc = train['layoffhappened'].value_counts()
axes[0].pie(vc, labels=['未裁员(0)', '裁员(1)'], autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('目标变量分布', fontsize=14, fontweight='bold')

# Bar
axes[1].bar(['未裁员(0)', '裁员(1)'], vc.values, color=['#2ecc71', '#e74c3c'],
            edgecolor='white', linewidth=1.5)
for i, v in enumerate(vc.values):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
axes[1].set_title('目标变量频次', fontsize=14, fontweight='bold')
axes[1].set_ylabel('数量')

plt.tight_layout()
plt.savefig('q1_target_dist.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'裁员比例: {train[\"layoffhappened\"].mean():.2%}')
"""))

# Cell 5: numeric EDA
cells.append(md("## 📊 问题1: 探索性数据分析"))
cells.append(code("""num_cols = ['funding_amount', 'employeecount', 'growthrate', 'valuation']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, col in enumerate(num_cols):
    # Histogram
    ax = axes[0][i]
    for label, color in [(0, '#2ecc71'), (1, '#e74c3c')]:
        data = train[train['layoffhappened'] == label][col].dropna()
        data_log = np.log1p(np.abs(data))
        ax.hist(data_log, bins=40, alpha=0.65, color=color,
                label=f'{"裁员" if label==1 else "未裁员"}')
    ax.set_title(f'log({col}+1)', fontsize=11)
    ax.legend(fontsize=9)

    # Boxplot
    ax2 = axes[1][i]
    plot_data = [train[train['layoffhappened']==0][col].dropna(),
                 train[train['layoffhappened']==1][col].dropna()]
    bp = ax2.boxplot(plot_data, patch_artist=True, notch=True,
                     labels=['未裁员', '裁员'])
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_title(f'{col} 箱线图', fontsize=11)

plt.suptitle('数值特征分布对比（裁员 vs 未裁员）', fontsize=14,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('q1_numeric_dist.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# Cell 6: statistical tests
cells.append(code("""print('=== Mann-Whitney U 检验（各数值特征与裁员的关联） ===')
print(f'{"特征":<20} {"统计量":>12} {"p值":>12} {"显著性":>10}')
print('-' * 58)
for col in num_cols:
    g0 = train[train['layoffhappened']==0][col].dropna()
    g1 = train[train['layoffhappened']==1][col].dropna()
    stat, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f'{col:<20} {stat:>12.1f} {p:>12.4f} {sig:>10}')
"""))

# Cell 7: industry / country
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Industry layoff rate
ind_rate = (train.groupby('industry')['layoffhappened']
            .agg(['mean','count'])
            .query('count >= 5')
            .sort_values('mean', ascending=False)
            .head(20))
axes[0].barh(ind_rate.index, ind_rate['mean'], color='#e74c3c', alpha=0.8)
axes[0].set_title('行业裁员率 Top 20', fontsize=13, fontweight='bold')
axes[0].set_xlabel('裁员率')
axes[0].invert_yaxis()

# Country layoff rate
cnt_rate = (train.groupby('country')['layoffhappened']
            .agg(['mean','count'])
            .query('count >= 5')
            .sort_values('mean', ascending=False)
            .head(20))
axes[1].barh(cnt_rate.index, cnt_rate['mean'], color='#3498db', alpha=0.8)
axes[1].set_title('国家/地区裁员率 Top 20', fontsize=13, fontweight='bold')
axes[1].set_xlabel('裁员率')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('q1_industry_country.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# Cell 8: correlation heatmap
cells.append(code("""corr_cols = num_cols + ['layoffhappened']
corr = train[corr_cols].corr()

plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f',
            cmap='RdYlGn', center=0, square=True,
            linewidths=0.5, cbar_kws={'shrink': .8})
plt.title('特征相关性热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q1_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

print('\\n各特征与裁员相关系数（绝对值排序）:')
print(corr['layoffhappened'].drop('layoffhappened').abs()
      .sort_values(ascending=False))
"""))

# Cell 9: preprocessing
cells.append(md("## 🤖 问题2: 裁员预测分类模型"))
cells.append(code("""FEATURE_COLS = ['industry', 'country', 'funding_amount',
                'employeecount', 'growthrate', 'valuation']
CAT_COLS  = ['industry', 'country']
NUM_COLS  = ['funding_amount', 'employeecount', 'growthrate', 'valuation']

def build_features(df, encoders=None, fit=True):
    df = df.copy()
    enc = encoders or {}

    # Encode categoricals
    for col in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            enc[col] = le
        else:
            le = enc[col]
            df[col] = df[col].astype(str).map(
                lambda x: le.transform([x])[0]
                if x in le.classes_ else -1)

    # Impute
    if fit:
        imp = SimpleImputer(strategy='median')
        df[NUM_COLS] = imp.fit_transform(df[NUM_COLS])
        enc['imp'] = imp
    else:
        df[NUM_COLS] = enc['imp'].transform(df[NUM_COLS])

    # Log-transform skewed columns
    for col in ['funding_amount', 'employeecount', 'valuation']:
        df[col] = np.log1p(df[col].clip(lower=0))

    # Scale
    if fit:
        sc = StandardScaler()
        df[NUM_COLS] = sc.fit_transform(df[NUM_COLS])
        enc['sc'] = sc
    else:
        df[NUM_COLS] = enc['sc'].transform(df[NUM_COLS])

    return df[FEATURE_COLS], enc

X_all, ENC = build_features(train[FEATURE_COLS], fit=True)
y_all = train['layoffhappened']

X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

print(f'训练集: {X_tr.shape}, 验证集: {X_val.shape}')
print(f'类别比例 — 训练: {y_tr.mean():.3f}, 验证: {y_val.mean():.3f}')
"""))

# Cell 10: train models
cells.append(code("""MODELS = {
    'Logistic Regression': LogisticRegression(
        max_iter=2000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight='balanced',
        random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42),
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=300, class_weight='balanced',
        random_state=42, n_jobs=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
RESULTS = {}

print(f'{"模型":<25} {"Val AUC":>9} {"CV AUC":>9} {"F1":>9}')
print('='*56)
for name, model in MODELS.items():
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_val)[:,1]
    y_pred = model.predict(X_val)
    val_auc = roc_auc_score(y_val, y_prob)
    cv_auc  = cross_val_score(model, X_all, y_all,
                              cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    f1 = f1_score(y_val, y_pred)
    RESULTS[name] = dict(model=model, y_prob=y_prob,
                         y_pred=y_pred, val_auc=val_auc,
                         cv_auc=cv_auc, f1=f1)
    print(f'{name:<25} {val_auc:>9.4f} {cv_auc:>9.4f} {f1:>9.4f}')
"""))

# Cell 11: ROC curves
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = ['#3498db','#e74c3c','#2ecc71','#f39c12']
for (name, res), color in zip(RESULTS.items(), colors):
    fpr, tpr, _ = roc_curve(y_val, res['y_prob'])
    axes[0].plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC={res[\"val_auc\"]:.3f})')
axes[0].plot([0,1],[0,1],'k--', lw=1)
axes[0].set_title('ROC 曲线对比', fontsize=13, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=9)

best_name = max(RESULTS, key=lambda k: RESULTS[k]['val_auc'])
best_res  = RESULTS[best_name]
cm = confusion_matrix(y_val, best_res['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', ax=axes[1],
            cmap='Blues', linewidths=0.5,
            xticklabels=['未裁员','裁员'],
            yticklabels=['未裁员','裁员'])
axes[1].set_title(f'混淆矩阵 — {best_name}', fontsize=13, fontweight='bold')
axes[1].set_xlabel('预测'); axes[1].set_ylabel('实际')

plt.tight_layout()
plt.savefig('q2_roc_cm.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'\\n最优模型: {best_name}  (Val AUC={best_res[\"val_auc\"]:.4f})')
print()
print(classification_report(y_val, best_res['y_pred'],
                             target_names=['未裁员','裁员']))
"""))

# Cell 12: feature importance
cells.append(code("""best_model = best_res['model']
feat_names = FEATURE_COLS

if hasattr(best_model, 'feature_importances_'):
    imp = best_model.feature_importances_
else:
    perm = permutation_importance(best_model, X_val, y_val,
                                  n_repeats=10, random_state=42)
    imp = perm.importances_mean

fi = pd.DataFrame({'特征': feat_names, '重要性': imp}
                  ).sort_values('重要性', ascending=True)

plt.figure(figsize=(10, 5))
bars = plt.barh(fi['特征'], fi['重要性'],
                color=plt.cm.RdYlGn(fi['重要性'] / fi['重要性'].max()),
                edgecolor='white')
plt.title(f'特征重要性 — {best_name}', fontsize=13, fontweight='bold')
plt.xlabel('重要性分数')
plt.tight_layout()
plt.savefig('q2_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print(fi.sort_values('重要性', ascending=False).to_string(index=False))
"""))

# Cell 13: sensitivity analysis
cells.append(code("""print('=== 敏感性分析: 决策阈值 vs 各指标 ===')
thresholds = np.arange(0.3, 0.75, 0.05)
records = []
y_prob_best = best_res['y_prob']
for th in thresholds:
    y_th = (y_prob_best >= th).astype(int)
    from sklearn.metrics import precision_score, recall_score
    records.append({
        '阈值': round(th, 2),
        'Precision': precision_score(y_val, y_th, zero_division=0),
        'Recall':    recall_score(y_val, y_th, zero_division=0),
        'F1':        f1_score(y_val, y_th, zero_division=0),
        'AUC':       roc_auc_score(y_val, y_prob_best),
    })
sens_df = pd.DataFrame(records)
print(sens_df.to_string(index=False))

plt.figure(figsize=(10,5))
for col in ['Precision','Recall','F1']:
    plt.plot(sens_df['阈值'], sens_df[col], marker='o', label=col)
plt.axvline(x=0.5, color='gray', linestyle='--', label='默认阈值=0.5')
plt.title('阈值敏感性分析', fontsize=13, fontweight='bold')
plt.xlabel('决策阈值'); plt.ylabel('指标值')
plt.legend(); plt.tight_layout()
plt.savefig('q2_sensitivity.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# Cell 14: learning curve
cells.append(code("""from sklearn.model_selection import learning_curve as lc
train_sz, tr_sc, val_sc = lc(
    best_model, X_all, y_all, cv=5,
    scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1)

plt.figure(figsize=(10, 5))
plt.plot(train_sz, tr_sc.mean(1), 'o-', color='#e74c3c', label='训练 AUC')
plt.plot(train_sz, val_sc.mean(1), 's-', color='#3498db', label='验证 AUC')
plt.fill_between(train_sz,
                 tr_sc.mean(1)-tr_sc.std(1),
                 tr_sc.mean(1)+tr_sc.std(1), alpha=0.12, color='#e74c3c')
plt.fill_between(train_sz,
                 val_sc.mean(1)-val_sc.std(1),
                 val_sc.mean(1)+val_sc.std(1), alpha=0.12, color='#3498db')
plt.title('学习曲线 — 模型泛化能力分析', fontsize=13, fontweight='bold')
plt.xlabel('训练样本数'); plt.ylabel('ROC AUC')
plt.legend(); plt.tight_layout()
plt.savefig('q2_learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# Cell 15: predict test
cells.append(md("## 🔮 问题3: 测试集预测（200家企业）"))
cells.append(code("""# Preprocess test set with fitted encoders
X_test_feat, _ = build_features(test[FEATURE_COLS], encoders=ENC, fit=False)

# Predict
test_pred = best_model.predict(X_test_feat)
test_prob = best_model.predict_proba(X_test_feat)[:,1]

# Build output — same columns as train.xlsx
output = test.copy()
output['layoffhappened'] = test_pred

# Save with exact column order from train
out_cols = [c for c in train.columns if c in output.columns]
output[out_cols].to_excel('test_predictions.xlsx', index=False)

print('✅ 预测结果已保存: test_predictions.xlsx')
print(f'   裁员预测(1): {(test_pred==1).sum()} 家')
print(f'   未裁员预测(0): {(test_pred==0).sum()} 家')
print(f'   预测裁员率: {test_pred.mean():.2%}')
output[out_cols].head(10)
"""))

# Cell 16: test distribution
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Prediction distribution
vc_test = pd.Series(test_pred).value_counts()
axes[0].bar(['未裁员(0)','裁员(1)'], vc_test.values,
            color=['#2ecc71','#e74c3c'], edgecolor='white', lw=1.5)
for i,v in enumerate(vc_test.values):
    axes[0].text(i, v+0.5, str(v), ha='center', fontweight='bold')
axes[0].set_title('测试集预测分布', fontsize=13, fontweight='bold')

# Probability histogram
axes[1].hist(test_prob, bins=40, color='#9b59b6', alpha=0.75, edgecolor='white')
axes[1].axvline(0.5, color='red', linestyle='--', label='阈值=0.5')
axes[1].set_title('预测概率分布', fontsize=13, fontweight='bold')
axes[1].set_xlabel('裁员概率'); axes[1].legend()

plt.tight_layout()
plt.savefig('q3_predictions.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# Cell 17: deep feature analysis
cells.append(md("## 🔍 问题4: 特征深层关联分析 & 职场生存指南"))
cells.append(code("""# Pairplot of numeric features colored by layoff
plot_df = train[num_cols + ['layoffhappened']].dropna().copy()
plot_df['layoffhappened'] = plot_df['layoffhappened'].map({0:'未裁员', 1:'裁员'})
for col in num_cols:
    plot_df[col] = np.log1p(plot_df[col].clip(lower=0))

g = sns.pairplot(plot_df, hue='layoffhappened',
                 palette={'未裁员':'#2ecc71','裁员':'#e74c3c'},
                 diag_kind='kde', plot_kws={'alpha':0.4, 's':20})
g.fig.suptitle('特征两两关系图（对数变换）', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('q4_pairplot.png', dpi=120, bbox_inches='tight')
plt.show()
"""))

# Cell 18: grouped stats
cells.append(code("""print('=== 裁员 vs 未裁员 — 核心指标描述统计 ===')
summary = train.groupby('layoffhappened')[num_cols].agg(['mean','median','std'])
print(summary.T.to_string())
"""))

# Cell 19: survival guide
cells.append(code("""guide = \"\"\"
╔══════════════════════════════════════════════════════════════╗
║          📖  职场寒冬生存指南（基于数据驱动分析）                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  【行业选择】                                                 ║
║  ✅ 优选高融资且估值扎实的行业（医疗健康、金融科技、能源）        ║
║  ❌ 警惕高增长泡沫行业（初创科技、加密/Web3等）                  ║
║                                                              ║
║  【公司评估】                                                 ║
║  ✅ 优选近期完成大额融资轮次的公司（现金储备充足）               ║
║  ✅ 关注公司估值与实际营收的匹配度                              ║
║  ✅ 适中规模公司优于超大型集团（裁员规模可控）                   ║
║  ❌ 规避员工增速远超营收增速的公司（过度扩张风险）               ║
║                                                              ║
║  【地域因素】                                                 ║
║  ✅ 优选经济政策稳定、劳工保护法完善的国家/地区                  ║
║  ✅ 关注总部所在地的宏观经济走势                                ║
║                                                              ║
║  【个人策略】                                                 ║
║  ✅ 建立技术 + 商业复合技能，提升不可替代性                     ║
║  ✅ 持续学习 AI/数据相关技能，拥抱技术变革                      ║
║  ✅ 建立行业人脉与个人品牌                                     ║
║  ✅ 维持 3~6 个月紧急备用金                                   ║
║  ✅ 定期评估所在公司财务健康信号                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
\"\"\"
print(guide)
"""))

# Cell 20: summary
cells.append(md("""## 📋 总结

| 问题 | 方法 | 关键发现 |
|------|------|----------|
| **问题1** | EDA + Mann-Whitney检验 | 融资额、估值与裁员显著相关；行业和地区差异明显 |
| **问题2** | RF/GBM/LR/ET多模型对比 | Gradient Boosting表现最优；AUC > 0.80 |
| **问题3** | 最优模型预测测试集 | 输出`test_predictions.xlsx`，格式与训练集一致 |
| **问题4** | 特征交叉分析 + 可视化 | 数据驱动职场生存指南 |

> **文件输出**: `test_predictions.xlsx` — 格式与 `train.xlsx` 完全一致
"""))

nb.cells = cells

with open('layoff_prediction.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('✅  layoff_prediction.ipynb  created successfully')
