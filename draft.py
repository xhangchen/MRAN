import csv
import pandas as pd
import glob
import subprocess
import os
import torchmetrics
import pandas as pd
import torch

if __name__ == '__main__':
    # tot = pd.read_csv('/home/xhc/MRAN/example_total.csv')
    # path_li = glob.glob('/home/xhc/data/TCGA-LUSC/*/*.svs')
    # name_to_path = {i.split('/')[-1]: i for i in path_li}
    # bp = 1
    # for svs_i in list(tot.iloc[:, 0]):
    #     os.makedirs(f'/home/xhc/MRAN/WSI/{svs_i}', exist_ok=True)
    #     subprocess.call(f'ln {name_to_path[svs_i]} /home/xhc/MRAN/WSI/{svs_i}', shell=True)

    # tot = pd.read_csv('/home/xhc/MRAN/csv/example/sheet/total.csv')
    # tot.columns = ['File Name', 'Sample Type']
    # tot.to_csv('/home/xhc/MRAN/csv/example/sheet/total.csv', index=False)

    auc_ = torchmetrics.AUROC(pos_label=1)
    df = pd.read_csv('/home/xhc/MRAN/run/2023-01-0415/predict.csv')
    auc_.update(torch.tensor(list(df['p1'])), torch.tensor(list(df['label'])))

    print(auc_.compute().item())
