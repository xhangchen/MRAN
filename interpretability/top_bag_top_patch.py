import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys

root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # xxx/MRAN
sys.path.append(root_dir)

import yaml
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchmetrics
import argparse
import torch.backends.cudnn as cudnn
import pandas as pd
import glob

from utils.par import Struct
from torch.utils.data import DataLoader
from datasets.datasets import MyDataset, Data_embedding
from model.ResNet import MRAN
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils.func import save_attention

cudnn.benchmark = True  ##

parser = argparse.ArgumentParser(description='test on top-bag top-patch')
parser.add_argument('--wsi_dir', type=str, default='WSI/example', help='path to svs file')
parser.add_argument('--output_dir', type=str, default='example', help='directory to save csv file')
# parser.add_argument('--wsi_dir', type=str, default='/home/xhc/data/TCGA-LUSC', help='path to svs file')
# parser.add_argument('--output_dir', type=str, default='2023-01-0415', help='directory to save csv file')
parser.add_argument('--top_bag_num', type=int, default=40, help='select the top_bag_num bags with the highest attention scores')
parser.add_argument('--top_patch_num', type=int, default=20, help='select the top_patch_num patches with the highest attention scores in each selected bag')
args = parser.parse_args()

wsi_dir = os.path.join(root_dir, args.wsi_dir)
tb, tp = args.top_bag_num, args.top_patch_num
output_dir = os.path.join(root_dir, 'run', args.output_dir, 'tb_tp', f'{tb}_{tp}')
os.makedirs(output_dir, exist_ok=True)


def get_path(save_dir):
    a_path = os.path.join(save_dir, 'alpha.pt')
    cp_path = os.path.join(save_dir, 'checkpoint.pth')
    config_path = os.path.join(save_dir, 'main.yaml')
    res_path = os.path.join(save_dir, 'result.pth')
    pred_path = os.path.join(save_dir, 'predict.csv')
    return a_path, cp_path, config_path, res_path, pred_path


alpha_path, cp_path, config_path, _, _ = get_path(os.path.join(root_dir, 'run', args.output_dir))

config = yaml.safe_load(open(config_path, 'r'))
par = Struct(**config)

temp_csv_path = os.path.join(output_dir, 'temp_test_case.csv')

init_csv_dir = os.path.join(root_dir, par.split_dir, str(par.split_id))

val_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((1024, 1024)),
    T.ToTensor(),
])


def evaluate(test_loader, model, epoch):
    num_wsi_test = test_loader.dataset.num_wsi
    embedding_test = [[] for i in range(num_wsi_test)]
    label_test = [-1 for i in range(num_wsi_test)]
    bag_id_test = [[] for i in range(num_wsi_test)]

    correct = 0
    total = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, (img, wsi_id, bag_id, label, bag_path, bag_weight, index, patch_alpha) in enumerate(
                tqdm(test_loader, desc=f'val_{epoch}' if epoch is not None else 'test')):
            img = img.cuda()
            label = label.cuda()
            with autocast():
                y, embedding, alpha, _1, beta, _3 = model(img, patch_alpha=patch_alpha, tag=0)  # y: n*2 embedding : n * 512 alpha : n * 64
                loss = F.cross_entropy(y, label, reduction='mean')
            total_loss += loss.item()
            y = y.argmax(dim=1)
            total += y.size(0)
            correct += (y == label).sum().item()

            for ind in range(len(wsi_id)):
                embedding_test[wsi_id[ind]].append(embedding[ind].tolist())
                label_test[wsi_id[ind]] = label[ind].item()
                bag_id_test[wsi_id[ind]].append(bag_id[ind].item())

    acc = correct / total
    return acc, total_loss, embedding_test, label_test, bag_id_test


def evaluate2(test_loader2, net, epoch):
    net.eval()

    correct = 0
    total = 0
    total_loss = 0

    recall_ = torchmetrics.Recall(average='none', num_classes=2)
    precision_ = torchmetrics.Precision(average='none', num_classes=2)
    auc_ = torchmetrics.AUROC(pos_label=1)
    recall_ = recall_.cuda()
    precision_ = precision_.cuda()
    auc_ = auc_.cuda()

    pred_data = []
    with torch.no_grad():
        for i, (embedding, label, bag_id, wsi_name, index) in enumerate(test_loader2):
            embedding = embedding.cuda()
            label = label.cuda()

            y, alpha, _ = net(embedding, tag=1)

            loss = F.cross_entropy(y, label)
            total_loss += loss.item()
            p = y.argmax()
            total += label.size(0)  #
            correct += (p == label).sum().item()

            auc_.update(y.softmax(dim=1)[:, 1], label)
            pred_data.append([wsi_name[0], y.softmax(dim=1)[:, 1].item(), label.item()])
            recall_.update(p.unsqueeze(dim=0), label)
            precision_.update(p.unsqueeze(dim=0), label)

    pred = pd.DataFrame(columns=['wsi', 'p1', 'label'], data=pred_data)
    pred.to_csv(os.path.join(output_dir, 'predict.csv'), index=False)
    return correct / total, total_loss, precision_.compute()[1].item(), recall_.compute()[
        1].item(), auc_.compute().item(), recall_.compute()[0].item()


def get_dataloader2(embedding, label, bag_id, wsi_name):
    data2 = Data_embedding(data=(embedding, label, bag_id, wsi_name))
    dataloader2 = DataLoader(dataset=data2, batch_size=1, shuffle=True)
    return dataloader2


def test(model):
    patch_alpha = torch.load(alpha_path)['best_bag_alpha']
    test_data = MyDataset(temp_csv_path, transform=val_transforms, patch_alpha=patch_alpha)
    test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4, drop_last=True, shuffle=False, pin_memory=True)

    up_acc, loss_test, embedding_test, label_test, bag_id_test = evaluate(test_loader, model, None)
    test_loader2 = get_dataloader2(embedding_test, label_test, bag_id_test, test_loader.dataset.wsi_name)
    acc, loss_test, precision, recall, auc, sp = evaluate2(test_loader2, model, None)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    torch.save(dict(acc=acc, auc=auc, f1=f1, se=recall, sp=sp), os.path.join(output_dir, 'result.pth'))
    print(f"test_acc: {acc:.4f}  test_auc: {auc:.4f}   test_f1: {f1:.4f}  test_se: {recall:.4f} test_sp: {sp:.4f}")


def make_csv():
    a = torch.load(alpha_path)
    test0 = pd.read_csv(f'{init_csv_dir}/case_test.csv')
    tot = test0

    li_abspath = glob.glob(f'{wsi_dir}/*/*.svs')
    name_to_path = {i.split('/')[-1][:-4]: i for i in li_abspath}
    test_wsi = [name_to_path[i] for i in a['best_wsi_alpha'].keys()]

    test_wsi_bag = []
    for k, v in a['best_wsi_alpha'].items():
        wsi_i = name_to_path[k]
        for i in v[:tb]:
            test_wsi_bag.append(f'{wsi_i}_{i}')

    tot = tot[tot['wsi'].isin(test_wsi)]
    tot.reset_index(drop=True, inplace=True)
    for i in range(len(tot)):
        tot.loc[i, 'wsi_bag_id'] = tot.loc[i, 'wsi'] + '_' + str(tot.loc[i, 'bag_id'])

    tot = tot[tot['wsi_bag_id'].isin(test_wsi_bag)]
    tot = tot.iloc[:, [0, 1, 2, 3, 4]]  # csv type: use_loc
    tot.reset_index(drop=True, inplace=True)
    tot.to_csv(temp_csv_path, index=False)


if __name__ == '__main__':

    make_csv()
    net = MRAN()
    net.cuda()
    net.upstream = torch.nn.DataParallel(net.upstream, device_ids=range(torch.cuda.device_count()))

    cp = torch.load(cp_path)
    net.load_state_dict(cp['best_pth'])

    test(net)
    if os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)
