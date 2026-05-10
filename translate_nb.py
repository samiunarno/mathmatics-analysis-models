import json, re

with open("layoff_prediction.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# ─── translation map (Chinese → English) ────────────────────────────────────
TR = {
    # markdown titles / labels
    "智驭职场寒冬 — 企业裁员风险预测": "Navigating the Job Market Winter — Corporate Layoff Risk Prediction",
    "2026年长春理工大学大学生数学建模竞赛 B题": "2026 Changchun University of Science & Technology Math Modeling Competition — Problem B",
    "**团队任务**: 利用多维企业数据，预测公司是否发生裁员（layoffhappened）": "**Team Objective**: Predict whether a company experienced layoffs (`layoffhappened`) using multi-dimensional corporate data.",
    "问题 | 内容": "Problem | Description",
    "问题1 | 探索性数据分析 (EDA) & 关键指标提炼": "Q1 | Exploratory Data Analysis & Key Indicator Extraction",
    "问题2 | 分类预测模型构建与敏感性分析": "Q2 | Classification Model Construction & Sensitivity Analysis",
    "问题3 | 测试集预测（200家企业）": "Q3 | Test Set Prediction (200 Companies)",
    "问题4 | 特征深层关联分析 & 职场生存指南": "Q4 | Deep Feature Relationship Analysis & Career Survival Guide",
    "## 📂 数据加载与概览": "## 📂 Data Loading & Overview",
    "## 📊 问题1: 探索性数据分析": "## 📊 Problem 1: Exploratory Data Analysis (EDA)",
    "## 🤖 问题2: 裁员预测分类模型": "## 🤖 Problem 2: Layoff Classification Prediction Model",
    "## 🔮 问题3: 测试集预测（200家企业）": "## 🔮 Problem 3: Test Set Prediction (200 Companies)",
    "## 🔍 问题4: 特征深层关联分析 & 职场生存指南": "## 🔍 Problem 4: Deep Feature Analysis & Career Survival Guide",
    "## 📋 总结": "## 📋 Summary",
    "问题 | 方法 | 关键发现": "Problem | Method | Key Finding",
    "**问题1** | EDA + Mann-Whitney检验 | 融资额、估值与裁员显著相关；行业和地区差异明显": "**Q1** | EDA + Mann-Whitney Test | Funding & valuation are significantly correlated with layoffs; clear industry & regional differences",
    "**问题2** | RF/GBM/LR/ET多模型对比 | Gradient Boosting表现最优；AUC > 0.80": "**Q2** | RF / GBM / LR / ET multi-model comparison | Gradient Boosting achieves best AUC > 0.80",
    "**问题3** | 最优模型预测测试集 | 输出`test_predictions.xlsx`，格式与训练集一致": "**Q3** | Best model applied to test set | Outputs `test_predictions.xlsx` matching train.xlsx format",
    "**问题4** | 特征交叉分析 + 可视化 | 数据驱动职场生存建议": "**Q4** | Cross-feature analysis + visualization | Data-driven career survival recommendations",
    "> **文件输出**: `test_predictions.xlsx` — 格式与 `train.xlsx` 完全一致": "> **Output file**: `test_predictions.xlsx` — columns match `train.xlsx` exactly",
    # code cell strings
    "# ==================== 环境配置 ====================": "# ==================== Environment Setup ====================",
    "print('✅ 所有库加载成功')": "print('✅ All libraries loaded successfully')",
    "训练集": "Train set",
    "测试集": "Test set",
    "行 × ": " rows × ",
    "列": " columns",
    "--- 训练集字段 ---": "--- Training Set Columns ---",
    "--- 前5行 ---": "--- First 5 Rows ---",
    "=== 缺失值统计 ===": "=== Missing Value Summary ===",
    "'缺失数': miss, '缺失率(%)': miss_pct": "'Missing Count': miss, 'Missing Rate(%)': miss_pct",
    "缺失数 > 0": "`Missing Count` > 0",
    "'未裁员(0)', '裁员(1)'": "'No Layoff (0)', 'Layoff (1)'",
    "目标变量分布": "Target Variable Distribution",
    "目标变量频次": "Target Variable Frequency",
    "裁员比例": "Layoff Rate",
    "数量": "Count",
    "数值特征分布对比（裁员 vs 未裁员）": "Numeric Feature Distribution (Layoff vs No-Layoff)",
    "'裁员' if label==1 else '未裁员'": "'Layoff' if label==1 else 'No Layoff'",
    "log({col}+1)": "log({col}+1)",
    "箱线图": "Boxplot",
    "'未裁员', '裁员'": "'No Layoff', 'Layoff'",
    "=== Mann-Whitney U 检验（各数值特征与裁员的关联） ===": "=== Mann-Whitney U Test (Numeric Features vs Layoff) ===",
    '"特征":<20} {"统计量":>12} {"p值":>12} {"显著性"': '"Feature":<20} {"Statistic":>12} {"p-value":>12} {"Significance"',
    "行业裁员率 Top 20": "Layoff Rate by Industry — Top 20",
    "裁员率": "Layoff Rate",
    "国家/地区裁员率 Top 20": "Layoff Rate by Country — Top 20",
    "特征相关性热力图": "Feature Correlation Heatmap",
    "各特征与裁员相关系数（绝对值排序）": "Feature Correlations with Layoff (sorted by abs value)",
    "# Encode categoricals": "# Encode categoricals",
    "训练集:": "Train:",
    "验证集:": "  Val:",
    "类别比例 — 训练:": "Class ratio — Train:",
    "验证:": "Val:",
    "f'训练集: {X_tr.shape}, 验证集: {X_val.shape}'": "f'Train: {X_tr.shape}, Validation: {X_val.shape}'",
    "f'类别比例 — 训练: {y_tr.mean():.3f}, 验证: {y_val.mean():.3f}'": "f'Class ratio — Train: {y_tr.mean():.3f}, Val: {y_val.mean():.3f}'",
    "ROC 曲线对比": "ROC Curve Comparison",
    "False Positive Rate": "False Positive Rate",
    "True Positive Rate": "True Positive Rate",
    "混淆矩阵 — ": "Confusion Matrix — ",
    "预测": "Predicted",
    "实际": "Actual",
    "最优模型:": "Best model:",
    "特征重要性 — ": "Feature Importance — ",
    "重要性分数": "Importance Score",
    "=== 敏感性分析: 决策阈值 vs 各指标 ===": "=== Sensitivity Analysis: Decision Threshold vs Metrics ===",
    "阈值": "Threshold",
    "阈值敏感性分析": "Threshold Sensitivity Analysis",
    "默认阈值=0.5": "Default threshold=0.5",
    "学习曲线 — 模型泛化能力分析": "Learning Curve — Model Generalization Analysis",
    "训练 AUC": "Train AUC",
    "验证 AUC": "Validation AUC",
    "训练样本数": "Training Samples",
    "# Preprocess test set with fitted encoders": "# Preprocess test set using fitted encoders",
    "# Predict": "# Predict",
    "# Build output — same columns as train.xlsx": "# Build output — same column order as train.xlsx",
    "# Save with exact column order from train": "# Save with exact column order from train",
    "✅ 预测结果已保存: test_predictions.xlsx": "✅ Predictions saved to: test_predictions.xlsx",
    "裁员预测(1):": "Layoff (1):",
    "未裁员预测(0):": "No Layoff (0):",
    "预测裁员率:": "Predicted Layoff Rate:",
    "家": " companies",
    "测试集预测分布": "Test Set Prediction Distribution",
    "预测概率分布": "Prediction Probability Distribution",
    "裁员概率": "Layoff Probability",
    "# Pairplot of numeric features colored by layoff": "# Pairplot of numeric features colored by layoff status",
    "{0:'未裁员', 1:'裁员'}": "{0:'No Layoff', 1:'Layoff'}",
    "特征两两关系图（对数变换）": "Pairwise Feature Relationships (Log-transformed)",
    "'未裁员':'#2ecc71','裁员':'#e74c3c'": "'No Layoff':'#2ecc71','Layoff':'#e74c3c'",
    "=== 裁员 vs 未裁员 — 核心指标描述统计 ===": "=== Layoff vs No-Layoff — Key Metric Descriptive Statistics ===",
    # survival guide
    "📖  职场寒冬生存指南（基于数据驱动分析）": "📖  Career Winter Survival Guide (Data-Driven)",
    "【行业选择】": "[Industry Selection]",
    "✅ 优选高融资且估值扎实的行业（医疗健康、金融科技、能源）": "✅ Prefer high-funding, solid-valuation industries (Healthcare, Fintech, Energy)",
    "❌ 警惕高增长泡沫行业（初创科技、加密/Web3等）": "❌ Avoid bubble industries with inflated growth (Early-stage Tech, Crypto/Web3)",
    "【公司评估】": "[Company Evaluation]",
    "✅ 优选近期完成大额融资轮次的公司（现金储备充足）": "✅ Prefer companies that recently closed large funding rounds (strong cash reserves)",
    "✅ 关注公司估值与实际营收的匹配度": "✅ Check that company valuation aligns with real revenue",
    "✅ 适中规模公司优于超大型集团（裁员规模可控）": "✅ Mid-size companies are safer than mega-corps (controlled layoff scale)",
    "❌ 规避员工增速远超营收增速的公司（过度扩张风险）": "❌ Avoid companies whose headcount growth far outpaces revenue growth (over-expansion risk)",
    "【地域因素】": "[Geographic Factors]",
    "✅ 优选经济政策稳定、劳工保护法完善的国家/地区": "✅ Choose countries/regions with stable economic policy and strong labor protections",
    "✅ 关注总部所在地的宏观经济走势": "✅ Monitor macroeconomic trends in the company's HQ country",
    "【个人策略】": "[Personal Strategy]",
    "✅ 建立技术 + 商业复合技能，提升不可替代性": "✅ Build Tech + Business hybrid skills to become indispensable",
    "✅ 持续学习 AI/数据相关技能，拥抱技术变革": "✅ Continuously learn AI/data skills and embrace technological change",
    "✅ 建立行业人脉与个人品牌": "✅ Build an industry network and personal brand",
    "✅ 维持 3~6 个月紧急备用金": "✅ Maintain 3–6 months of emergency savings",
    "✅ 定期评估所在公司财务健康信号": "✅ Regularly assess your company's financial health signals",
    # model print statements
    "print(f'最优模型: {best_name}  (Val AUC={best_res[\"val_auc\"]:.4f})')": "print(f'Best model: {best_name}  (Val AUC={best_res[\"val_auc\"]:.4f})')",
    "target_names=['未裁员','裁员']": "target_names=['No Layoff','Layoff']",
    "xticklabels=['未裁员','裁员'],\n            yticklabels=['未裁员','裁员']": "xticklabels=['No Layoff','Layoff'],\n            yticklabels=['No Layoff','Layoff']",
    "axes[0].set_title('测试集预测分布'": "axes[0].set_title('Test Set Prediction Distribution'",
    "axes[1].set_title('预测概率分布'": "axes[1].set_title('Prediction Probability Distribution'",
    "axes[1].set_xlabel('裁员概率')": "axes[1].set_xlabel('Layoff Probability')",
}

def translate(text):
    for zh, en in TR.items():
        text = text.replace(zh, en)
    return text

for cell in nb["cells"]:
    if isinstance(cell["source"], list):
        cell["source"] = [translate(line) for line in cell["source"]]
    else:
        cell["source"] = translate(cell["source"])

with open("layoff_prediction.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("✅ Notebook fully translated to English.")
