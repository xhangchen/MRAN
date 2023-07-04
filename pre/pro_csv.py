import os
import pandas as pd
import csv
import math
from collections import Counter
import glob
from tqdm import tqdm
from random import shuffle
import slideio
import argparse

root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # xxx/MRAN

parser = argparse.ArgumentParser(description='pro_csv')
parser.add_argument('--total_sheet_path', type=str, default='csv/example/sheet/total.csv', help='path to total sheet file')
parser.add_argument('--sheet_dir', type=str, default='csv/example/sheet', help='directory to save sheet file')
parser.add_argument('--split_dir', type=str, default='csv/example/split', help='directory to save split csv file')
parser.add_argument('--loc_path', type=str, default='csv/example/loc/', help='path to loc csv file')
parser.add_argument('--wsi_dir', type=str, default='WSI/example', help='path to svs file')

parser.add_argument('--s_train', type=int, default=1, help='proportion of training set')
parser.add_argument('--s_val', type=int, default=1, help='proportion of validation set')
parser.add_argument('--s_test', type=int, default=1, help='proportion of test set')
parser.add_argument('--k', type=int, default=1, help='k-fold Monte Carlo cross-validation')

args = parser.parse_args()
sheet_dir = os.path.join(root_dir, args.sheet_dir)
split_dir = os.path.join(root_dir, args.split_dir)
sheet_subdir = f'{sheet_dir}/{args.s_train}{args.s_val}{args.s_test}'
split_subdir = os.path.join(split_dir, f'{args.s_train}{args.s_val}{args.s_test}')
loc_path = os.path.join(root_dir, args.loc_path)
wsi_dir = os.path.join(root_dir, args.wsi_dir)

li_abspath = glob.glob(f'{wsi_dir}/*/*.svs')
name_to_path = {i.split('/')[-1][:-4]: i for i in li_abspath}


def make_case_sheet_abc(s_train, s_val, s_test, dir_id=0):
    tot_sheet = pd.read_csv(os.path.join(root_dir, args.total_sheet_path))
    tot_sheet = tot_sheet.sample(frac=1.0).reset_index(drop=True)
    tot_0 = tot_sheet[tot_sheet['Sample Type'] == "Solid Tissue Normal"].reset_index(drop=True)
    tot_1 = tot_sheet[tot_sheet['Sample Type'] == "Primary Tumor"].reset_index(drop=True)
    cnt_train_0 = math.floor(len(tot_0) * s_train / (s_train + s_val + s_test))
    cnt_eval_0 = math.floor(len(tot_0) * s_val / (s_train + s_val + s_test))
    cnt_train_1 = math.floor(len(tot_1) * s_train / (s_train + s_val + s_test))
    cnt_eval_1 = math.floor(len(tot_1) * s_val / (s_train + s_val + s_test))
    case_train = []
    case_val = []
    case_test = []
    for cnt, i in enumerate(tot_0['File Name']):
        if cnt < cnt_train_0:
            case_train.append(i[:12])
        elif cnt < cnt_train_0 + cnt_eval_0:
            case_val.append(i[:12])
        else:
            case_test.append(i[:12])

    for cnt, i in enumerate(tot_1['File Name']):
        if cnt < cnt_train_1:
            case_train.append(i[:12])
        elif cnt < cnt_train_1 + cnt_eval_1:
            case_val.append(i[:12])
        else:
            case_test.append(i[:12])

    n_train = pd.DataFrame(columns=list(tot_sheet))
    n_val = pd.DataFrame(columns=list(tot_sheet))
    n_test = pd.DataFrame(columns=list(tot_sheet))
    last_dir = os.path.join(sheet_subdir, str(dir_id))
    os.makedirs(last_dir, exist_ok=True)

    for i in range(len(tot_sheet)):

        t = list(tot_sheet.iloc[i])

        if t[0][:12] in case_train:
            n_train.loc[len(n_train) + 1] = t
        elif t[0][:12] in case_val:
            n_val.loc[len(n_val) + 1] = t
        else:
            n_test.loc[len(n_test) + 1] = t
    n_train.to_csv(f"{last_dir}/case_train.csv", index=False)
    n_val.to_csv(f"{last_dir}/case_val.csv", index=False)
    n_test.to_csv(f"{last_dir}/case_test.csv", index=False)


def check_csv_label_split(split_id):
    dir = f'{sheet_dir}/{args.s_train}{args.s_val}{args.s_test}/{split_id}'
    col = 1  # dir= ~_loc col should be 4

    train = pd.read_csv(f'{dir}/case_train.csv')
    val = pd.read_csv(f'{dir}/case_val.csv')
    test = pd.read_csv(f'{dir}/case_test.csv')

    c1 = Counter(train.iloc[:, col])
    c2 = Counter(val.iloc[:, col])
    c3 = Counter(test.iloc[:, col])
    # print(sum(c1.values()) + sum(c2.values()) + sum(c3.values()))
    print(f'{split_id}th fold case split:')
    if train.iloc[0, col] in {"Solid Tissue Normal", "Primary Tumor"}:
        print('train:\t', c1, '\tpostive/negative: ', c1['Primary Tumor'] / c1['Solid Tissue Normal'])
        print('val:\t', c2, '\tpostive/negative: ', c2['Primary Tumor'] / c2['Solid Tissue Normal'])
        print('test:\t', c3, '\tpostive/negative: ', c3['Primary Tumor'] / c3['Solid Tissue Normal'])
    else:
        print(c1, c1[1] / c1[0])
        print(c2, c2[1] / c2[0])
        print(c3, c3[1] / c3[0])


def sheet_to_csv2(sheet_path, csv_path, data_path):
    csv_dir = '/'.join(csv_path.split('/')[:-1])
    os.makedirs(csv_dir, exist_ok=True)
    # case: save loc
    data = pd.read_csv(sheet_path)

    col = ['wsi', 'bag_id', 'x', 'y', 'type']
    res = pd.DataFrame(columns=col)
    li_name = [i[:-4] for i in list(data['File Name'])]

    if data['Sample Type'][0] in {"Solid Tissue Normal", "Primary Tumor"}:
        li_type = [0 if i == 'Solid Tissue Normal' else 1 for i in list(data['Sample Type'])]
    else:
        li_type = [int(i) for i in list(data['Sample Type'])]
    d = dict(zip(li_name, li_type))

    for svs_i in tqdm(os.listdir(data_path)):
        name_i = svs_i.split('/')[-1][:-4]
        if d.get(name_i) is None:
            continue
        t = pd.read_csv(os.path.join(data_path, svs_i))
        path_i = name_to_path[name_i]
        t['wsi'] = path_i
        t['type'] = d[name_i]
        # res = res.append(t)
        res = pd.concat([res, t])
    res.to_csv(csv_path, index=False)


if __name__ == '__main__':
    for i in range(args.k):
        make_case_sheet_abc(args.s_train, args.s_val, args.s_test, dir_id=i)
        check_csv_label_split(i)
        for j in ['train', 'val', 'test']:
            sheet_path = os.path.join(sheet_subdir, str(i), f'case_{j}.csv')
            split_path = os.path.join(split_subdir, str(i), f'case_{j}.csv')
            sheet_to_csv2(sheet_path, split_path, loc_path)
