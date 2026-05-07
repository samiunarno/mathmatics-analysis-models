import pandas as pd
import numpy as np
import pulp
import os
import warnings
warnings.filterwarnings('ignore')

class SupplyChainOptimizer:
    def __init__(self, file1="Attachment_1.xlsx", file2="Attachment_2.xlsx"):
        self.file1 = file1
        self.file2 = file2
        self.WEEKS = 24
        self.DEMAND = 28200
        self.CAPACITY = 6000
        self.price_ratio = {'A': 1.2, 'B': 1.1, 'C': 1.0}
        self.strategic_weights = {'A': 0.1, 'B': 1.0, 'C': 5.0}

    def execute(self):
        print("Starting execution...")
        if not os.path.exists(self.file1) or not os.path.exists(self.file2):
            print("Error: Excel files not found in the directory.")
            return

        top_suppliers, transporters, supplier_capacity = self.solve_q1()
        
        if top_suppliers is not None:
            self.solve_q2_q3(top_suppliers, transporters, "Q2")
            self.solve_q2_q3(top_suppliers, transporters, "Q3")
            self.solve_q4(top_suppliers, transporters, supplier_capacity)
            
        print("All processes completed successfully.")

    def solve_q1(self):
        print("Processing Question 1...")
        try:
            df_orders = pd.read_excel(self.file1, sheet_name="Enterprise_Order").fillna(0)
            df_supply = pd.read_excel(self.file1, sheet_name="Supplier_Supply").fillna(0)
            df_loss = pd.read_excel(self.file2).fillna(0)
        except Exception as e:
            print(f"Error reading files: {e}")
            return None, None, None

        transporters = df_loss.iloc[:, 0].tolist()
        suppliers = df_orders.iloc[:, 0].values
        materials = df_orders.iloc[:, 1].values
        
        order_vals = df_orders.iloc[:, 2:].values
        supply_vals = df_supply.iloc[:, 2:].values
        
        total_ordered = np.sum(order_vals, axis=1)
        total_supplied = np.sum(supply_vals, axis=1)
        
        fulfillment = np.clip(total_supplied / (total_ordered + 1e-9), 0, 1)
        supply_std = np.std(supply_vals, axis=1)
        max_std = np.max(supply_std) if np.max(supply_std) > 0 else 1
        norm_stability = 1 - (supply_std / max_std) 
        
        df_eval = pd.DataFrame({
            'Supplier_ID': suppliers,
            'Material_Type': materials,
            'Total_Supply': total_supplied,
            'Fulfillment_Rate': fulfillment,
            'Stability': norm_stability
        })
        
        df_eval['Score'] = (df_eval['Fulfillment_Rate'] * 0.5) + (df_eval['Total_Supply'] / df_eval['Total_Supply'].max() * 0.3) + (df_eval['Stability'] * 0.2)
        top_50 = df_eval.sort_values(by='Score', ascending=False).head(50)
        top_50.to_excel("Q1_Top_50_Suppliers.xlsx", index=False)
        
        max_historical_supply = np.max(supply_vals, axis=1)
        supplier_capacity = dict(zip(suppliers, max_historical_supply))

        return top_50, transporters, supplier_capacity

    def solve_q2_q3(self, top_suppliers_df, transporters, mode):
        print(f"Processing Optimization Model {mode}...")
        suppliers = top_suppliers_df['Supplier_ID'].tolist()
        materials_dict = top_suppliers_df.set_index('Supplier_ID')['Material_Type'].to_dict()
        
        model = pulp.LpProblem(f"Optimization_{mode}", pulp.LpMinimize)

        order = pulp.LpVariable.dicts("Order", ((s, w) for s in suppliers for w in range(1, self.WEEKS + 1)), lowBound=0)
        transport = pulp.LpVariable.dicts("Transport", ((s, t, w) for s in suppliers for t in transporters for w in range(1, self.WEEKS + 1)), lowBound=0)

        if mode == "Q2":
            model += pulp.lpSum(order[s, w] * self.price_ratio[materials_dict[s]] for s in suppliers for w in range(1, self.WEEKS + 1))
        elif mode == "Q3":
            model += pulp.lpSum(order[s, w] * self.strategic_weights[materials_dict[s]] for s in suppliers for w in range(1, self.WEEKS + 1))

        for w in range(1, self.WEEKS + 1):
            model += pulp.lpSum(transport[s, t, w] for s in suppliers for t in transporters) >= self.DEMAND
            for t in transporters:
                model += pulp.lpSum(transport[s, t, w] for s in suppliers) <= self.CAPACITY
            for s in suppliers:
                model += pulp.lpSum(transport[s, t, w] for t in transporters) == order[s, w]

        model.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[model.status] == 'Optimal':
            self.export_to_excel(order, transport, suppliers, transporters, materials_dict, mode)
        else:
            print(f"Solver failed for {mode}")

    def solve_q4(self, top_suppliers_df, transporters, supplier_capacity):
        print("Processing Optimization Model Q4...")
        suppliers = top_suppliers_df['Supplier_ID'].tolist()
        materials_dict = top_suppliers_df.set_index('Supplier_ID')['Material_Type'].to_dict()
        
        model = pulp.LpProblem("Optimization_Q4", pulp.LpMaximize)

        order = pulp.LpVariable.dicts("Order", ((s, w) for s in suppliers for w in range(1, self.WEEKS + 1)), lowBound=0)
        transport = pulp.LpVariable.dicts("Transport", ((s, t, w) for s in suppliers for t in transporters for w in range(1, self.WEEKS + 1)), lowBound=0)
        max_demand_var = pulp.LpVariable("Max_Weekly_Demand", lowBound=self.DEMAND, upBound=len(transporters)*self.CAPACITY)

        model += max_demand_var

        for w in range(1, self.WEEKS + 1):
            model += pulp.lpSum(transport[s, t, w] for s in suppliers for t in transporters) >= max_demand_var
            for t in transporters:
                model += pulp.lpSum(transport[s, t, w] for s in suppliers) <= self.CAPACITY
            for s in suppliers:
                model += pulp.lpSum(transport[s, t, w] for t in transporters) == order[s, w]
                if s in supplier_capacity:
                    model += order[s, w] <= supplier_capacity[s]

        model.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[model.status] == 'Optimal':
            self.export_to_excel(order, transport, suppliers, transporters, materials_dict, "Q4")
            print(f"Q4 Maximum Feasible Weekly Capacity: {round(max_demand_var.varValue, 2)}")

    def export_to_excel(self, order, transport, suppliers, transporters, materials_dict, mode):
        order_data = [[s, materials_dict[s]] + [round(order[s, w].varValue, 2) for w in range(1, self.WEEKS + 1)] for s in suppliers]
        cols_A = ['Supplier_ID', 'Material_Type'] + [f'Week_{w}' for w in range(1, self.WEEKS + 1)]
        pd.DataFrame(order_data, columns=cols_A).to_excel(f"Attachment_A_{mode}.xlsx", index=False)
        
        transport_data = []
        for s in suppliers:
            for t in transporters:
                total_t = sum(transport[s, t, w].varValue for w in range(1, self.WEEKS + 1))
                if total_t > 0: 
                    row = [s, t] + [round(transport[s, t, w].varValue, 2) for w in range(1, self.WEEKS + 1)]
                    transport_data.append(row)
                    
        cols_B = ['Supplier_ID', 'Transporter_ID'] + [f'Week_{w}' for w in range(1, self.WEEKS + 1)]
        pd.DataFrame(transport_data, columns=cols_B).to_excel(f"Attachment_B_{mode}.xlsx", index=False)

if __name__ == "__main__":
    pipeline = SupplyChainOptimizer()
    pipeline.execute()