import pandas as pd
import numpy as np
import pulp
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
path1, path2 = "files/Attachment_1.xlsx", "files/Attachment_2.xlsx"

if not os.path.exists(path1) or not os.path.exists(path2):
    print("Files not found")
else:
    df_ord = pd.read_excel(path1, sheet_name="Enterprise_Order")
    df_sup = pd.read_excel(path1, sheet_name="Supplier_Supply")
    df_loss = pd.read_excel(path2)
    
    sup_vals = df_sup.iloc[:, 2:].values
    ord_vals = df_ord.iloc[:, 2:].values
    suppliers = df_ord.iloc[:, 0].values
    m_types = df_ord.iloc[:, 1].values
    transporters = df_loss.iloc[:, 0].tolist()

    X_time = np.array(range(1, 241)).reshape(-1, 1)
    future_time = np.array(range(241, 265)).reshape(-1, 1)
    future_preds = []

    for i in range(len(df_sup)):
        y_val = sup_vals[i].reshape(-1, 1)
        reg_model = LinearRegression().fit(X_time, y_val)
        pred = reg_model.predict(future_time)
        future_preds.append(np.mean(np.maximum(0, pred)))

    total_vol = np.sum(sup_vals, axis=1)
    fulfillment = np.clip(total_vol / (np.sum(ord_vals, axis=1) + 1e-9), 0, 1)
    stability = 1 - (np.std(sup_vals, axis=1) / (np.max(np.std(sup_vals, axis=1)) + 1e-9))

    df_master = pd.DataFrame({
        'Supplier_ID': suppliers,
        'Material_Type': m_types,
        'Fulfillment_Rate': fulfillment,
        'Stability_Index': stability,
        'Predicted_Trend': future_preds,
        'Total_Supply': total_vol
    })

    df_master['Final_Score'] = (df_master['Fulfillment_Rate'] * 0.4) + \
                               (df_master['Stability_Index'] * 0.3) + \
                               (df_master['Predicted_Trend'] / (max(future_preds) + 1e-9) * 0.3)

    top_50 = df_master.sort_values('Final_Score', ascending=False).head(50)
    top_50.to_excel("Q1_Top_50_Suppliers.xlsx", index=False)

    def solve_model(target_df, mode="Q2"):
        s_list = target_df['Supplier_ID'].tolist()
        m_dict = target_df.set_index('Supplier_ID')['Material_Type'].to_dict()
        caps = dict(zip(suppliers, np.max(sup_vals, axis=1)))
        
        WEEKS, DEMAND, MAX_T = 24, 28200, 6000
        prices = {'A': 1.2, 'B': 1.1, 'C': 1.0}

        sense = pulp.LpMaximize if mode == "Q4" else pulp.LpMinimize
        prob = pulp.LpProblem(f"Optimization_{mode}", sense)

        x = pulp.LpVariable.dicts("Ord", ((s, w) for s in s_list for w in range(1, 25)), lowBound=0)
        y = pulp.LpVariable.dicts("Trn", ((s, t, w) for s in s_list for t in transporters for w in range(1, 25)), lowBound=0)
        d_val = pulp.LpVariable("DynDemand", lowBound=28200) if mode == "Q4" else 28200

        if mode == "Q2":
            prob += pulp.lpSum(x[s,w] * prices[m_dict[s]] for s in s_list for w in range(1, 25))
        else:
            prob += d_val

        for w in range(1, 25):
            prob += pulp.lpSum(y[s,t,w] for s in s_list for t in transporters) >= d_val
            for t in transporters:
                prob += pulp.lpSum(y[s,t,w] for s in s_list) <= MAX_T
            for s in s_list:
                prob += pulp.lpSum(y[s,t,w] for t in transporters) == x[s,w]
                if mode == "Q4": prob += x[s,w] <= caps.get(s, 0)

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        a_rows = [[s, m_dict[s]] + [round(x[s,w].varValue, 2) for w in range(1,25)] for s in s_list]
        pd.DataFrame(a_rows, columns=['ID', 'Type'] + [f'W{i}' for i in range(1,25)]).to_excel(f"Attachment_A_{mode}.xlsx", index=False)
        
        b_rows = []
        for s in s_list:
            for t in transporters:
                if sum(y[s,t,w].varValue for w in range(1,25)) > 0:
                    b_rows.append([s, t] + [round(y[s,t,w].varValue, 2) for w in range(1,25)])
        pd.DataFrame(b_rows, columns=['S_ID', 'T_ID'] + [f'W{i}' for i in range(1,25)]).to_excel(f"Attachment_B_{mode}.xlsx", index=False)

    solve_model(top_50, mode="Q2")
    solve_model(top_50, mode="Q4")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Supply Chain Analysis Dashboard', fontsize=22, fontweight='bold')

    sns.barplot(ax=axes[0, 0], x='Supplier_ID', y='Final_Score', data=top_50.head(10), palette='viridis')
    axes[0, 0].tick_params(axis='x', rotation=30)

    sns.scatterplot(ax=axes[0, 1], x='Fulfillment_Rate', y='Predicted_Trend', hue='Material_Type', size='Total_Supply', data=top_50, sizes=(100, 500), alpha=0.7)

    top_50['Material_Type'].value_counts().plot.pie(ax=axes[1, 0], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'], explode=[0.05]*len(top_50['Material_Type'].unique()))
    axes[1, 0].set_ylabel('')

    sns.histplot(ax=axes[1, 1], data=top_50, x='Stability_Index', kde=True, color='teal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()