import pandas as pd
import numpy as np
import os

print("replace NaN data with medium values: start...")
proj_path = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_ex_vivo\Deep_Learning'
data = pd.read_csv(os.path.join(proj_path, 'Gleason.csv'))

def na_to_median(col):
    for i in list(data.ROI_Class.unique()):
        filter_col = data[data['ROI_Class'] == i][col].median()
        data[col].fillna(filter_col, inplace=True)
col_list = data.columns[data.isnull().any()].tolist()
for i in col_list:
    na_to_median(i)
col_names = data.columns.tolist()
data.to_csv(os.path.join(proj_path, 'Gleason.csv'), header=col_names, index=False)
print("replace NaN data with medium values: complete!!!")


