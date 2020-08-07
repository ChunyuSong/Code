import pandas as pd
import numpy as np
import os
import timeit
from datetime import datetime
from time import gmtime, strftime


def load_data():

    data = pd.read_csv(os.path.join(proj_dir, data_file))

    return data

def na_to_median(col):
    
    for i in list(data.ROI_Class.unique()):
        
        filter_col = data[data['ROI_Class'] == i][col].median()
        data[col].fillna(filter_col, inplace=True)

def replace_nan():

    col_list = data.columns[data.isnull().any()].tolist()
    
    for i in col_list:
        na_to_median(i)
        
    col_names = data.columns.tolist()
    
    data.to_csv(
                os.path.join(result_dir, save_file),
                header=col_names,
                index=False
                )


if __name__ == '__main__':

    proj_dir   = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\ALL_DBSI_DATA_new_thresh_all\Non_PCa_without_biopsy'
    result_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\ALL_DBSI_DATA_new_thresh_all\Non_PCa_without_biopsy'
    data_file  = 'benign_mpMRI_voxel_22_May_2020_00_51_26.csv'
    save_file  = 'benign_mpMRI_voxel.csv'

    
    print("extract voxel values and index from ROI and DBSI data: start...")
    
    start = timeit.default_timer()

    data = load_data()
    
    replace_nan()

    stop = timeit.default_timer()
    running_seconds = np.around(stop-start, 0)
    
    print('DNN running time:', running_seconds, 'seconds')
    print("replace nan: complete!!!")


