import json

nb_path = 'layoff_prediction.ipynb'
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

# Find the data loading cell (by id or content)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'train = pd.read_excel' in ''.join(cell['source']):
        # Replace the source with updated version
        new_source = [
            "train = pd.read_excel('train.xlsx')\n",
            "test  = pd.read_excel('test.xlsx')\n",
            "# Standardize target column name\n",
            "if 'layoff_happened' in train.columns:\n",
            "    train = train.rename(columns={'layoff_happened':'layoffhappened'})\n",
            "if 'layoff_happened' in test.columns:\n",
            "    test = test.rename(columns={'layoff_happened':'layoffhappened'})\n",
            "\n",
            "print(f'Train set: {train.shape[0]}  rows × {train.shape[1]}  columns')\n",
            "print(f'Test set: {test.shape[0]}  rows × {test.shape[1]}  columns')\n",
            "print()\n",
            "print('--- Train setFields ---')\n",
            "print(train.dtypes)\n",
            "print()\n",
            "print('--- First 5 Rows ---')\n",
            "train.head()\n"
        ]
        cell['source'] = new_source
        break

# Write back the notebook
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('✅ Notebook fixed: column name standardized')
