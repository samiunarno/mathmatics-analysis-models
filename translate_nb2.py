import json, re

with open("layoff_prediction.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

# Additional replacements missed in first pass
TR2 = {
    # Cell 3
    "--- 训练集字段 ---": "--- Training Set Fields ---",
    "字段": "Fields",
    # Cell 7
    "'裁员' if label==1 else '未裁员'": "'Layoff' if label==1 else 'No Layoff'",
    "label='裁员'": "label='Layoff'",
    "label='未裁员'": "label='No Layoff'",
    "label=f'裁员'": "label=f'Layoff'",
    "label=f'未裁员'": "label=f'No Layoff'",
    # Cell 9
    "国家/地区裁员率 Top 20": "Layoff Rate by Country/Region — Top 20",
    "国": "Country",
    "地区": "Region",
    # Cell 13
    "模型": "Model",
    # Cell 14
    "xticklabels=['未裁员','裁员']": "xticklabels=['No Layoff','Layoff']",
    "yticklabels=['未裁员','裁员']": "yticklabels=['No Layoff','Layoff']",
    "labels=['未裁员', '裁员']": "labels=['No Layoff', 'Layoff']",
    "labels=['未裁员','裁员']": "labels=['No Layoff','Layoff']",
    "'裁员'": "'Layoff'",
    "'未裁员'": "'No Layoff'",
    # Cell 15
    "'特征': feat_names, '重要性': imp": "'Feature': feat_names, 'Importance': imp",
    ".sort_values('重要性'": ".sort_values('Importance'",
    "fi['特征']": "fi['Feature']",
    "fi['重要性']": "fi['Importance']",
    "'重要性'": "'Importance'",
    "'特征'": "'Feature'",
    "特征重要性 — ": "Feature Importance — ",
    # Cell 16
    "默认阈值=0.5": "Default Threshold=0.5",
    "决策阈值": "Decision Threshold",
    "阈值敏感性分析": "Threshold Sensitivity Analysis",
    "指标值": "Metric Value",
    "plt.xlabel('决策阈值')": "plt.xlabel('Decision Threshold')",
    "plt.ylabel('指标值')": "plt.ylabel('Metric Value')",
    # Cell 19
    "结果已保存": "predictions saved to",
    "✅ 预测结果已保存: test_predictions.xlsx": "✅ Predictions saved: test_predictions.xlsx",
    "裁员预测(1)": "Layoff (1)",
    "未裁员预测(0)": "No Layoff (0)",
    "预测裁员率": "Predicted Layoff Rate",
    # Cell 20
    "axes[0].bar(['未裁员(0)','裁员(1)']": "axes[0].bar(['No Layoff(0)','Layoff(1)']",
    "预测分布": "Prediction Distribution",
    "概率分布": "Probability Distribution",
    "测试集预测分布": "Test Prediction Distribution",
    "预测概率分布": "Prediction Probability Distribution",
    "裁员概率": "Layoff Probability",
    # Cell 22
    "{0:'未裁员', 1:'裁员'}": "{0:'No Layoff', 1:'Layoff'}",
    "'未裁员':'#2ecc71','裁员':'#e74c3c'": "'No Layoff':'#2ecc71','Layoff':'#e74c3c'",
    # Cell 24 (survival guide)
    "关注公司估值与实际营收的匹配度": "Check that company valuation matches real revenue",
    "实际营收的匹配度": "real revenue",
    "优选经济政策稳定、劳工保护法完善的国家/地区": "Choose countries with stable policy and strong labor laws",
    "劳工保护法完善的国": "strong labor law countries",
    # Cell 25
    "问题 | 方法 | 关键发现": "Problem | Method | Key Finding",
    "特征交叉分析 + 可视化": "Cross-feature analysis + visualization",
    "数据驱动职场生存指南": "Data-driven career survival guide",
    "| **Q4** | Cross-feature analysis + visualization | 数据驱动职场生存指南": "| **Q4** | Cross-feature analysis + visualization | Data-driven career survival guide",
    # misc remaining
    "f'训练集: {X_tr.shape}, 验证集: {X_val.shape}'": "f'Train: {X_tr.shape}, Val: {X_val.shape}'",
    "f'类别比例 — 训练: {y_tr.mean():.3f}, 验证: {y_val.mean():.3f}'": "f'Class ratio — Train: {y_tr.mean():.3f}, Val: {y_val.mean():.3f}'",
    "f'Train set: {train.shape[0]} 行 × {train.shape[1]}  columns'": "f'Train set: {train.shape[0]} rows × {train.shape[1]} columns'",
    "f'Test set: {test.shape[0]} 行 × {test.shape[1]}  columns'": "f'Test set: {test.shape[0]} rows × {test.shape[1]} columns'",
    "行 × {train.shape[1]}  columns": "rows × {train.shape[1]} columns",
    "行 × {test.shape[1]}  columns": "rows × {test.shape[1]} columns",
    "print(f'Layoff Rate: {train[\"layoffhappened\"].mean():.2%}')": "print(f'Layoff Rate: {train[\"layoffhappened\"].mean():.2%}')",
    "print(f'裁员率: {train[\"layoffhappened\"].mean():.2%}')": "print(f'Layoff Rate: {train[\"layoffhappened\"].mean():.2%}')",
}

def translate(text):
    for zh, en in TR2.items():
        text = text.replace(zh, en)
    # Remove any stray Chinese labels in f-strings
    text = re.sub(r"label='裁员'", "label='Layoff'", text)
    text = re.sub(r"label='未裁员'", "label='No Layoff'", text)
    text = re.sub(r"'裁员'", "'Layoff'", text)
    text = re.sub(r"'未裁员'", "'No Layoff'", text)
    text = re.sub(r'"裁员"', '"Layoff"', text)
    text = re.sub(r'"未裁员"', '"No Layoff"', text)
    return text

for cell in nb["cells"]:
    if isinstance(cell["source"], list):
        cell["source"] = [translate(line) for line in cell["source"]]
    else:
        cell["source"] = translate(cell["source"])

with open("layoff_prediction.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Final check
remaining = []
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    zh = re.findall(r"[\u4e00-\u9fff]+", src)
    if zh:
        remaining.append(f"Cell {i}: {zh[:8]}")

if remaining:
    print("⚠️  Still has Chinese:")
    for r in remaining:
        print(" ", r)
else:
    print("✅ All Chinese text translated to English.")
