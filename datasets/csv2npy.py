import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = 'SWAT'

scaler = StandardScaler()

train_csv = pd.read_csv(f'D:/workspace/temp/py/llm/llms-for-ad/dataset/{dataset}/{dataset}_train.csv')
test_csv = pd.read_csv(f'D:/workspace/temp/py/llm/llms-for-ad/dataset/{dataset}/{dataset}_test.csv')
if dataset == 'SWAT':
    label_csv = pd.read_csv(f'D:/workspace/temp/py/llm/llms-for-ad/dataset/{dataset}/{dataset}_test.csv')
else:
    label_csv = pd.read_csv(f'D:/workspace/temp/py/llm/llms-for-ad/dataset/{dataset}/{dataset}_label.csv')

train_feature = train_csv.values[:, 1:] if dataset != 'SWAT' else train_csv.values[:, :-1]
train_feature = np.nan_to_num(train_feature)
scaler.fit(train_feature)
train_feature = scaler.transform(train_feature)
np.save(f'D:/workspace/temp/py/llm/llms-for-ad/dataset/{dataset}/{dataset}_train.npy', train_feature)

test_feature = test_csv.values[:, 1:] if dataset != 'SWAT' else test_csv.values[:, :-1]
test_feature = np.nan_to_num(test_feature)
scaler.fit(test_feature)
test_feature = scaler.transform(test_feature)
np.save(f'D:/workspace/temp/py/llm/llms-for-ad/dataset/{dataset}/{dataset}_test.npy', test_feature)

label = label_csv.values[:, 1:] if dataset != 'SWAT' else label_csv.values[:, -1:]
label = np.nan_to_num(label)
label = label.squeeze()
np.save(f'D:/workspace/temp/py/llm/llms-for-ad/dataset/{dataset}/{dataset}_test_label.npy', label)
