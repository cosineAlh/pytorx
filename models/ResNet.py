'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())
from python.torx.module.layer import crxb_Conv2d
from python.torx.module.layer import crxb_Linear

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise, enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma, enable_retention, retention_time, drift_coefficient, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = crxb_Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                 enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = crxb_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                 enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        #self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            #self.shortcut = nn.Sequential(
            #    crxb_Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
            #    #nn.BatchNorm2d(self.expansion*planes)
            #)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        #out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise, enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma, enable_retention, retention_time, drift_coefficient, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = crxb_Conv2d(in_planes, planes, kernel_size=1, bias=False,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                 enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = crxb_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                 enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = crxb_Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                 enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            #self.shortcut = nn.Sequential(
            #    crxb_Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
            #    #nn.BatchNorm2d(self.expansion*planes)
            #)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        #out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.conv2(out))
        #out = self.bn3(self.conv3(out))
        out = self.conv3(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise, enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma, enable_retention, retention_time, drift_coefficient, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = crxb_Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                 enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        
        #self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1, 
                                      crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                      enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.layer2 = self._make_layer(block, self.in_planes*2, num_blocks[1], stride=2, 
                                      crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                      enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.layer3 = self._make_layer(block, self.in_planes*4, num_blocks[2], stride=2, 
                                      crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                      enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.layer4 = self._make_layer(block, self.in_planes*8, num_blocks[3], stride=2, 
                                      crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                      enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.linear = nn.Linear(self.in_planes*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise, enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma, enable_retention, retention_time, drift_coefficient):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, 
                                crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                                enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise, enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma, enable_retention, retention_time, drift_coefficient):
    return ResNet(BasicBlock, [2,2,2,2], 
                crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)

def ResNet34(crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise, enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma, enable_retention, retention_time, drift_coefficient):
    return ResNet(BasicBlock, [3,4,6,3],
                crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)

def ResNet50(crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise, enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma, enable_retention, retention_time, drift_coefficient):
    return ResNet(Bottleneck, [3,4,6,3],
                crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)

def ResNet101(crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise, enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma, enable_retention, retention_time, drift_coefficient):
    return ResNet(Bottleneck, [3,4,23,3],
                crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)

def ResNet152(crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise, enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma, enable_retention, retention_time, drift_coefficient):
    return ResNet(Bottleneck, [3,8,36,3],
                crxb_size=crxb_size, scaler_dw=scaler_dw, gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp, enable_retention=enable_retention, retention_time=retention_time, drift_coefficient=drift_coefficient,
                enable_resistance_variance=enable_resistance_variance, resistance_variance_gamma=resistance_variance_gamma, enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF, enable_noise=enable_noise, ir_drop=ir_drop, device=device)