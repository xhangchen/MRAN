import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ExtractEmbedding(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=2):
        super(ExtractEmbedding, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  #
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.num_classes = num_classes

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x' shape: n * 3 * 1024 *1024
        n = x.shape[0]
        out = rearrange(x, 'N C (ph h) (pw w) -> (N ph pw) C h w', ph=8, pw=8)  # (N*64)*3*128*128
        # replace rearrange
        # out = x.reshape((x.shape[0], 3, 1024, 8, 128))
        # out = out.reshape((out.shape[0], 3, 8, 128, 8, 128))
        # out = out.permute(0, 2, 4, 1, 3, 5)
        # out = out.reshape(-1, 3, 128, 128)

        out = F.relu(self.bn1(self.conv1(out)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # (N*64)*512*4*4

        out = out.mean(dim=(2, 3))

        out = out.reshape(n, -1, out.shape[-1])
        embedding = out.mean(dim=1).detach()
        out = self.linear(out).mean(dim=1)
        # out's shape: n * num_classes       embedding' shape: n * 512
        return out, embedding


class Attention(nn.Module):
    def __init__(self, keep_threshold=False):
        super(Attention, self).__init__()
        self.keep_threshold = keep_threshold

    def forward(self, x, w, b):
        # Process bag in one slide at a time
        # x = torch.squeeze(x, dim=0)
        gamma = x.shape[1]
        out_alpha = F.linear(x, w, b)

        out = torch.sqrt((out_alpha ** 2).sum(dim=2))
        alpha = out / out.sum(dim=1, keepdim=True).expand_as(out)
        if self.keep_threshold:
            # alpha = F.relu(alpha - .1 / float(gamma))
            alpha = F.relu(alpha - .1 / float(gamma))
        alpha = alpha / alpha.sum(dim=1, keepdim=True).expand_as(alpha)

        out = alpha.unsqueeze(dim=2).expand_as(x) * x
        out = out.sum(dim=1)
        out = F.linear(out, w, b)

        alpha = alpha.squeeze(dim=0).detach()
        # out's shape: 1* num_classes   alpha's shape: gamma    out_alpha: 1 * gamma * num_classes
        return out, alpha, out_alpha.squeeze(dim=0)


class Attention_sample(nn.Module):
    def __init__(self, ):
        super(Attention_sample, self).__init__()

    def forward(self, x, w, b):
        #  x: (n*64) x 512 x 16 x 16
        x = x.flatten(start_dim=2)
        x = x.permute(0, 2, 1)  # (n * 64) * 256 * 512
        y_beta = F.linear(x, w, b)
        gamma = x.shape[1]

        alpha = torch.sqrt((x ** 2).sum(dim=2))
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        # alpha = F.relu(alpha - .1 / float(gamma))
        alpha = F.relu(alpha - .1 / float(gamma))  #
        alpha = alpha / alpha.sum(dim=1, keepdim=True)

        out = alpha.unsqueeze(dim=2) * x
        out = out.sum(dim=1)

        return out, alpha, y_beta


class net_up_aa(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=2, keep_patch_threshold=True,
                 top_patch_num=None):
        super(net_up_aa, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.num_classes = num_classes
        self.attention_sample = Attention_sample()
        self.attention = Attention_up(keep_patch_threshold=keep_patch_threshold, top_patch_num=top_patch_num)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, patch_alpha):
        # x' shape: n * 3 * 1024 *1024
        n = x.shape[0]
        out = rearrange(x, 'N C (ph h) (pw w) -> (N ph pw) C h w', ph=8, pw=8)  # (N*64)*3*128*128
        # replace rearrange
        # out = x.reshape((x.shape[0], 3, 1024, 8, 128))
        # out = out.reshape((out.shape[0], 3, 8, 128, 8, 128))
        # out = out.permute(0, 2, 4, 1, 3, 5)
        # out = out.reshape(-1, 3, 128, 128)

        out = F.relu(self.bn1(self.conv1(out)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # (N*64)*512*4*4


        out, beta, y_beta = self.attention_sample(out, self.linear.weight, self.linear.bias)

        out = out.reshape(n, -1, out.shape[-1])

        out, embedding, alpha, out_alpha = self.attention(out, self.linear.weight, self.linear.bias, patch_alpha)
        # out's shape: n * num_classes       embedding's shape: n * 512     alpha' shape: 64
        return out, embedding, alpha, out_alpha, beta, y_beta


class Attention_up(nn.Module):
    def __init__(self, keep_patch_threshold=True, top_patch_num=None):
        super(Attention_up, self).__init__()
        self.keep_patch_threshold = keep_patch_threshold
        self.top_patch_num = top_patch_num

    def forward(self, x, w, b, patch_alpha):
        # Process bag in one slide at a time
        # x = torch.squeeze(x, dim=0)

        if not self.keep_patch_threshold:
            x_ = torch.empty((x.shape[0], self.top_patch_num, x.shape[2]), device=x.device)
            for i in range(x.shape[0]):
                x_[i] = x[i, patch_alpha[i][:self.top_patch_num]]
            x = x_
        gamma = x.shape[1]
        out_alpha = F.linear(x, w, b)

        out = torch.sqrt((out_alpha ** 2).sum(dim=2))
        alpha = out / out.sum(dim=1, keepdim=True).expand_as(out)
        if self.keep_patch_threshold:
            # alpha = F.relu(alpha - .1 / float(gamma))
            alpha = F.relu(alpha - .1 / float(gamma))  #
        alpha = alpha / alpha.sum(dim=1, keepdim=True).expand_as(alpha)

        out = alpha.unsqueeze(dim=2).expand_as(x) * x
        out = out.sum(dim=1)
        embedding = out.detach()
        out = F.linear(out, w, b)

        # alpha = alpha.squeeze(dim=0).detach()
        alpha = alpha.detach()
        # out's shape: n * num_classes   embedding's shape: n * 512     out_alpha's shape: n * 64 * num_classes
        return out, embedding, alpha, out_alpha


class net_up(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=2):
        super(net_up, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  #
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # todo
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.num_classes = num_classes
        self.attention = Attention_up()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x' shape: n * 3 * 1024 *1024
        n = x.shape[0]
        out = rearrange(x, 'N C (ph h) (pw w) -> (N ph pw) C h w', ph=8, pw=8)  # (N*64)*3*128*128
        # replace rearrange
        # out = x.reshape((x.shape[0], 3, 1024, 8, 128))
        # out = out.reshape((out.shape[0], 3, 8, 128, 8, 128))
        # out = out.permute(0, 2, 4, 1, 3, 5)
        # out = out.reshape(-1, 3, 128, 128)

        out = F.relu(self.bn1(self.conv1(out)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # (N*64)*512*16*16

        out = out.mean(dim=(2, 3))

        out = out.reshape(n, -1, out.shape[-1])

        out, embedding, alpha, out_alpha = self.attention(out, self.linear.weight, self.linear.bias)
        # out's shape: n * num_classes       embedding's shape: n * 512     alpha' shape: 64
        # out_alpha' shape* n * 64 * 2
        return out, embedding, alpha, out_alpha


class MIL(nn.Module):
    def __init__(self, dim_fectures=512, num_classes=2, num_hidden_unit=32, keep_threshold=True):
        super(MIL, self).__init__()
        self.linear0 = nn.Linear(dim_fectures, num_hidden_unit)
        self.linear1 = nn.Linear(num_hidden_unit, num_classes)
        self.attention = Attention(keep_threshold=keep_threshold)

    def forward(self, x):
        x = F.linear(x, self.linear0.weight, self.linear0.bias)
        return self.attention(x, self.linear1.weight, self.linear1.bias)


class MRAN(nn.Module):
    def __init__(self, num_classes=2, dim_fectures=512, num_hidden_unit=32, keep_bag_threshold=True,
                 keep_patch_threshold=True, top_patch_num=None):
        super(MRAN, self).__init__()
        self.upstream = net_up_aa(num_classes=num_classes, keep_patch_threshold=keep_patch_threshold,
                                  top_patch_num=top_patch_num)  # todo
        self.downstream = MIL(dim_fectures=dim_fectures, num_hidden_unit=num_hidden_unit,
                              keep_threshold=keep_bag_threshold)

    def forward(self, x, tag, patch_alpha=-1):
        if tag == 0:
            return self.upstream(x, patch_alpha=patch_alpha)
        else:
            return self.downstream(x)


class conv(nn.Module):
    def __init__(self, num_ceil=64):
        super(conv, self).__init__()
        ceil_edge = int(num_ceil ** .5)
        assert ceil_edge ** 2 == num_ceil and ceil_edge % 2 == 0 and ceil_edge <= 16

        cnt = int((16 - ceil_edge) / 2)
        li = [conv_unit() for i in range(cnt)]
        self.conv = nn.Sequential(*li)

    def forward(self, x):
        return self.conv(x)


class conv_unit(nn.Module):
    def __init__(self):
        super(conv_unit, self).__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(512)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out
