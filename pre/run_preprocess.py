import os
import subprocess

import pandas as pd
from tqdm import tqdm
import glob
import argparse

root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # xxx/MRAN

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--save_dir', type=str, default='csv/example/loc/', help='directory to save csv file')
parser.add_argument('--dataset_dir', type=str, default='WSI/example/', help='svs datasets storage directory')

if __name__ == '__main__':
    args = parser.parse_args()
    save_dir = os.path.join(root_dir, args.save_dir)
    dataset_dir = os.path.join(root_dir, args.dataset_dir)
    os.makedirs(save_dir, exist_ok=True)
    li_abspath = glob.glob(f'{dataset_dir}/*/*.svs')  # todo

    for cnt, i in enumerate(tqdm(li_abspath)):
        if (cnt + 1) % 10 == 0:
            subprocess.call(f'python3 preprocess.py --wsi_path {i} --save_dir {save_dir}', shell=True)
        else:
            subprocess.call(f'python3 preprocess.py --wsi_path {i} --save_dir {save_dir} &', shell=True)
