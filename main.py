import sys
import yaml

path_yaml = "./main.yaml"
par = yaml.safe_load(open(path_yaml, 'r'))
from utils.par import Struct

par = Struct(**par)
import os

root_dir = os.path.abspath(os.getcwd())  # xxx/MRAN
sys.path.append(root_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = par.CUDA_VISIBLE_DEVICES

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import subprocess
import torch.backends.cudnn as cudnn
import torchmetrics
import time

from torch.utils.data import DataLoader
from datasets.datasets import MyDataset, Data_embedding
from model.ResNet import MRAN
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR
from tqdm import tqdm
from collections import defaultdict
from utils.func import get_bag_weight, mask_cross_entropy, weight_cross_entropy, ramp_up

cudnn.benchmark = True  ##

output_dir = par.output_dir if par.output_dir else time.strftime('%Y-%m-%d%H', time.localtime(time.time()))
output_dir = os.path.join(root_dir, 'run', output_dir)
os.makedirs(output_dir, exist_ok=True)

subprocess.call(f'cp main.yaml {output_dir}', shell=True)

cp_path = os.path.join(output_dir, 'checkpoint.pth')
if not par.resume and os.path.exists(cp_path):
    os.remove(cp_path)

train_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((1024, 1024)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])
val_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((1024, 1024)),
    T.ToTensor(),
])

up_weight = defaultdict(lambda: 1.0)

wsi_alpha = {}
best_wsi_alpha = {}

bag_alpha = {}
bag_beta = {}
bag_label = {}

best_bag_alpha = {}
best_bag_bet = {}
best_bag_label = {}
best_acc_up = 0
best_acc_down = 0

final_recall = 0
final_precision = 0
final_sp = 0
final_f1 = 0
final_auc = 0


def train(train_loader, epoch, model, optimizer, scheduler=None):
    scaler = torch.cuda.amp.GradScaler()

    lossfunc = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight, reduction='none')
    lossfunc.cuda()

    num_wsi_train = train_loader.dataset.num_wsi
    temp_embedding = [[] for i in range(num_wsi_train)]
    temp_label = [-1 for i in range(num_wsi_train)]
    temp_bag_id = [[] for i in range(num_wsi_train)]

    total_loss = 0
    t0, t1, t2 = 0, 0, 0
    ramp_up1 = par.Lambda * ramp_up(epoch, par.num_epoch)
    ramp_up2 = par.Lambda ** 2 * ramp_up(epoch, par.num_epoch)

    cnt_one = 0
    model.train()

    for i, (img, wsi_id, bag_id, label, bag_path, bag_weight, index, _) in enumerate(tqdm(train_loader, desc=f'train_{epoch}')):

        img = img.cuda()
        label = label.cuda()
        cnt_one += (bag_weight == 1).sum().item()
        bag_weight = bag_weight.cuda()
        with autocast():
            y, embedding, alpha, y_alpha, beta, y_beta = model(img, tag=0)  # y: n*2 embedding : n * 512 _ : n*64
            loss0 = mask_cross_entropy(y, label, bag_weight, loss=lossfunc)
            w1 = bag_weight.unsqueeze(dim=1) * alpha

            loss1 = weight_cross_entropy(y_alpha.view(-1, 2), label.repeat(64, 1).T.contiguous().view(-1), w1.view(-1), loss=lossfunc)
            w2 = (bag_weight.unsqueeze(dim=1) * alpha).view(-1).unsqueeze(dim=1)
            w2 = (w2 * beta).view(-1)

            loss2 = weight_cross_entropy(y_beta.view(-1, 2), label.repeat(64 * 16, 1).T.contiguous().view(-1), w2, loss=lossfunc)
            loss = loss0 + ramp_up1 * loss1 + ramp_up2 * loss2
            t0 += loss0.item()
            t1 += ramp_up1 * loss1.item()
            t2 += ramp_up2 * loss2.item()

            total_loss += loss.item()

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        for ind in range(len(index)):
            temp_embedding[wsi_id[ind]].append(embedding[ind].tolist())
            temp_label[wsi_id[ind]] = label[ind].item()
            temp_bag_id[wsi_id[ind]].append(bag_id[ind].item())

    print(f"loss0: {t0:.4f}  loss1: {t1:.4f}   loss2: {t2:.4f}")

    return total_loss, temp_embedding, temp_label, temp_bag_id, cnt_one


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
        for i, (img, wsi_id, bag_id, label, bag_path, bag_weight, index, _) in \
                enumerate(tqdm(test_loader, desc=f'val_{epoch}' if epoch is not None else 'test')):
            img = img.cuda()
            label = label.cuda()
            with autocast():
                y, embedding, alpha, _, beta, __ = model(img, tag=0)  # y: n*2 embedding : n * 512 alpha : n * 64
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


def train2(train_loader, epoch, net, optimizer, scheduler=None):
    total_loss = 0
    t0 = 0
    t1 = 0

    lossfunc = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight, reduction='none')
    lossfunc.cuda()

    ramp_up3 = par.Lambda * ramp_up(epoch, par.num_epoch)
    up_weight.clear()
    net.train()
    for i, (embedding, label, bag_id, wsi_name, index) in enumerate(train_loader):
        embedding = embedding.cuda()
        label = label.cuda()

        y, alpha, y_alpha = net(embedding, tag=1)

        loss0 = F.cross_entropy(y, label) if not lossfunc else lossfunc(y, label)
        loss1 = weight_cross_entropy(y_alpha, label.repeat(alpha.shape[0]), alpha, loss=lossfunc)

        loss = loss0 + ramp_up3 * loss1
        t0 += loss0.item()
        t1 += ramp_up3 * loss1.item()

        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        temp = get_bag_weight(wsi_name[0], bag_id.squeeze(dim=0).tolist(), alpha.tolist())
        up_weight.update(temp)

    print(f"_loss0: {t0:.4f}  _loss1: {t1:.4f}")

    scheduler.step()

    return total_loss


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
            recall_.update(p.unsqueeze(dim=0), label)
            precision_.update(p.unsqueeze(dim=0), label)

    return correct / total, total_loss, precision_.compute()[1].item(), recall_.compute()[1].item(), \
           auc_.compute().item(), recall_.compute()[0].item()


def get_dataloader(up_weight=None):
    pt = os.path.join(root_dir, par.split_dir, str(par.split_id), 'case_train.csv')
    pe = os.path.join(root_dir, par.split_dir, str(par.split_id), 'case_val.csv')

    train_data = MyDataset(path=pt, transform=train_transforms, up_weight=up_weight)
    val_data = MyDataset(path=pe, transform=val_transforms, up_weight=up_weight)

    train_loader = DataLoader(dataset=train_data, batch_size=par.batch_size, num_workers=par.num_workers, drop_last=True, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=par.batch_size, num_workers=par.num_workers, drop_last=True, shuffle=False, pin_memory=True)

    return train_loader, val_loader


def get_dataloader2(embedding, label, bag_id, wsi_name):
    data2 = Data_embedding(data=(embedding, label, bag_id, wsi_name))
    dataloader2 = DataLoader(dataset=data2, batch_size=1, shuffle=True)
    return dataloader2


def downstream_task(train_loader2, val_loader2, net, optimizer2, scheduler2, epoch_up):
    id_up = f'{epoch_up:0{2}}'
    loss = train2(train_loader2, epoch_up, net, optimizer2, scheduler2)
    acc, loss_val, precision, recall, auc, sp = evaluate2(val_loader2, net, epoch=epoch_up)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    print(f'_epoch_{id_up}:  train loss: {loss:.4f}      val loss: {loss_val:.4f}      val acc: {acc:.6f} '
          f'auc: {auc:.6f}   f1: {f1:.6f}    se: {recall:.6f}       sp: {sp:.6f}')

    return acc, recall, precision, sp, f1, auc


def test(model):
    p_test = os.path.join(root_dir, par.split_dir, str(par.split_id), 'case_test.csv')

    test_data = MyDataset(p_test, transform=val_transforms)
    test_loader = DataLoader(dataset=test_data, batch_size=par.batch_size, num_workers=par.num_workers, drop_last=True, shuffle=False, pin_memory=True)

    up_acc, loss_test, embedding_test, label_test, bag_id_test = evaluate(test_loader, model, None)
    test_loader2 = get_dataloader2(embedding_test, label_test, bag_id_test, test_loader.dataset.wsi_name)
    acc, loss_test, precision, recall, auc, sp = evaluate2(test_loader2, model, None)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    print(f"test_acc: {acc:.4f}  test_auc: {auc:.4f}    test_f1: {f1:.4f}   test_precision: {precision:.4f}  test_se: {recall:.4f}  test_sp:{sp:.4f}")


def main():
    global up_weight
    train_loader, val_loader = get_dataloader(up_weight)
    net = MRAN()
    net.cuda()
    net.upstream = torch.nn.DataParallel(net.upstream, device_ids=range(torch.cuda.device_count()))

    optimizer1 = torch.optim.Adam(net.upstream.parameters(), lr=par.learning_rate1)
    scheduler1 = OneCycleLR(optimizer1, max_lr=par.learning_rate1, steps_per_epoch=len(train_loader), epochs=par.num_epoch)
    optimizer2 = torch.optim.Adam(net.downstream.parameters(), lr=par.learning_rate2)
    scheduler2 = MultiStepLR(optimizer2, milestones=[5], gamma=.5)

    global best_acc_up, best_acc_down, bag_alpha, final_auc, final_recall, final_f1, final_precision
    best_acc_down_epoch = -1

    start_epoch = 0
    if par.resume and os.path.exists(cp_path):
        cp = torch.load(cp_path)
        net.load_state_dict(cp['model_state_dict'])
        optimizer1.load_state_dict(cp['optimizer1_state_dict'])
        scheduler1.load_state_dict(cp['scheduler1_state_dict'])
        optimizer2.load_state_dict(cp['optimizer2_state_dict'])
        scheduler2.load_state_dict(cp['scheduler2_state_dict'])
        start_epoch = cp['epoch'] + 1
        up_weight = defaultdict(lambda: 1.0, cp['up_weight'])
        train_loader, val_loader = get_dataloader(up_weight)

    for epoch in range(start_epoch, par.num_epoch):

        loss, temp_embedding, temp_label, temp_bag_id, _ = train(train_loader, epoch, net, optimizer1, scheduler1)

        acc, loss_val, embedding_val, label_val, bag_id_val = evaluate(val_loader, net, epoch=epoch)
        if acc > best_acc_up:
            best_acc_up = acc

        print(f'epoch: {epoch} train loss: {loss:.2f}     val loss: {loss_val:.2f}     val acc: {acc:.6f}')

        train_loader2 = get_dataloader2(temp_embedding, temp_label, temp_bag_id, train_loader.dataset.wsi_name)
        val_loader2 = get_dataloader2(embedding_val, label_val, bag_id_val, val_loader.dataset.wsi_name)

        _acc, recall, precision, sp, f1, auc = downstream_task(train_loader2, val_loader2, net, optimizer2, scheduler2, epoch)

        if _acc >= best_acc_down:
            best_acc_down = _acc
            best_acc_down_epoch = epoch
            final_recall = recall
            final_precision = precision
            final_sp = sp
            final_f1 = f1
            final_auc = auc
            best_pth = net.state_dict()

        torch.save({'id': id, 'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer1_state_dict': optimizer1.state_dict(),
                    'scheduler1_state_dict': scheduler1.state_dict(),
                    'optimizer2_state_dict': optimizer2.state_dict(),
                    'scheduler2_state_dict': scheduler2.state_dict(),
                    'up_weight': dict(up_weight),
                    'best_pth': best_pth}, cp_path)

        train_loader, val_loader = get_dataloader(up_weight)

    print(f'best acc_up: {best_acc_up:.4f}     best acc_down: {best_acc_down:.4f}   final_f1: {final_f1:.4f}   '
          f'final_auc: {final_auc:.4f}    best_acc_epoch:{best_acc_down_epoch:.4f}')

    print(f'training completed.\n checkpoint is saved in output_dir: {output_dir}')


if __name__ == '__main__':
    main()
