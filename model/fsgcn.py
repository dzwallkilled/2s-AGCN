import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.agcn import unit_gcn, unit_tcn, TCN_GCN_unit, conv_init, bn_init, conv_branch_init
from utils.msic import *
from model.ordinal_regressor import ORegressors


def fuse_logits(logits1, logits2, mode='sum', w=1):
    # mode: sum, weighted sum, max
    new_logits = []
    for d1, d2 in zip(logits1, logits2):
        if not isinstance(d1, list):
            if mode == 'sum':
                new_logits.append(d1 + d2)
            elif mode == 'wsum':
                new_logits.append(d1 + w * d2)
            elif mode == 'max':
                new_logits.append(torch.max(d1, d2))
            elif mode == 'mul':
                new_logits.append(torch.mul(d1, d2))
            else:
                raise ValueError(f"No such fusion mode {mode}")
        else:
            new_logits.append(fuse_logits(d1, d2, mode, w))
    return new_logits


# TODO: add pyramid tcn
class unit_ptcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pyramid_level=2):
        super(unit_ptcn, self).__init__()
        self.convs = nn.ModuleList()
        for i in range(pyramid_level):
            pad = int((kernel_size - 1) / 2)
            conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                                           stride=(stride, 1)),
                                 nn.BatchNorm2d(out_channels))
            self.convs.append(conv)
            kernel_size += 2
        self.pyramid_level = pyramid_level
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        out = None
        for i in range(self.pyramid_level):
            out = out + self.convs[i](x) if out is not None else self.convs[i](x)
        out /= self.pyramid_level

        return self.relu(out)


# TODO: add feature sharing layer
class unit_fsn(nn.Module):
    def __init__(self, in_channels1, out_channels1, in_channels2, out_channels2, coff_embedding=4):
        super(unit_fsn, self).__init__()
        assert in_channels1 == in_channels2 and out_channels1 == out_channels2
        self.in_c = in_channels1
        self.out_c = out_channels1
        self.inner_c = out_channels1 // coff_embedding

        self.theta = nn.Conv2d(in_channels1, self.inner_c, 1)
        self.phi = nn.Conv2d(in_channels2, self.inner_c, 1)
        self.gamma1 = nn.Conv2d(in_channels1, self.inner_c, 1)
        self.gamma2 = nn.Conv2d(in_channels1, self.inner_c, 1)
        self.conv1 = nn.Conv2d(self.inner_c, out_channels1, 1)
        self.conv2 = nn.Conv2d(self.inner_c, out_channels2, 1)

        self.bn1 = nn.BatchNorm2d(out_channels1)
        self.bn2 = nn.BatchNorm2d(out_channels2)
        self.soft = nn.Softmax(-2)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        if in_channels1 != out_channels1:
            self.identity1 = nn.Sequential(
                nn.Conv2d(in_channels1, out_channels1, 1),
                nn.BatchNorm2d(out_channels1)
            )
            self.identity2 = nn.Sequential(
                nn.Conv2d(in_channels2, out_channels2, 1),
                nn.BatchNorm2d(out_channels2)
            )
        else:
            self.identity1 = lambda x: x
            self.identity2 = lambda x: x

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn1, 1e-6)
        bn_init(self.bn2, 1e-6)

    def forward(self, x1, x2):
        N1, C1, T1, V1 = x1.size()
        N2, C2, T2, V2 = x2.size()
        theta_x = self.theta(x1).permute(0, 3, 1, 2).contiguous().view(N1, V1, self.inner_c * T1)
        phi_x = self.phi(x2).view(N2, self.inner_c * T2, V2)
        cor_matrix = self.soft(torch.matmul(theta_x, phi_x))
        gamma_x1 = self.gamma1(x1).view(N1, self.inner_c * T1, V1)
        gamma_x2 = self.gamma2(x2).view(N2, self.inner_c * T2, V2)
        out2 = self.bn1(self.conv1(torch.matmul(gamma_x1, cor_matrix).view(N2, self.inner_c, T2, V2)))
        out1 = self.bn2(self.conv2(torch.matmul(gamma_x2, cor_matrix.transpose(2, 1)).view(N1, self.inner_c, T1, V1)))

        out1 += self.identity1(x1)
        out1 = self.relu1(out1)
        out2 += self.identity2(x2)
        out2 = self.relu2(out2)
        return out1, out2


class PTCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, pyramid_level=2):
        super(PTCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.ptcn1 = unit_ptcn(out_channels, out_channels, stride=stride, pyramid_level=pyramid_level)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_ptcn(in_channels, out_channels, kernel_size=1, stride=stride,
                                      pyramid_level=pyramid_level)

    def forward(self, x):
        x = self.ptcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Model2Stream(nn.Module):
    def __init__(self, num_class=60, branch_args1=dict(), branch_args2=dict(), graph=None, graph_args=dict(),
                 use_fsn=False, use_ptcn=False, pyramid_level=2, fusion_mode='sum', fusion_weight=1):
        super(Model2Stream, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        self.A = self.graph.A

        self.pyramid_level = pyramid_level
        self.use_fsn = use_fsn
        self.use_ptcn = use_ptcn
        self.num_class = num_class
        self.fusion_mode = fusion_mode
        self.fusion_weight = fusion_weight

        self.data_bn1, self.layers1, self.final1, self.fs_layers = self._build_layers_one_stream(**branch_args1)
        self.data_bn2, self.layers2, self.final2, _ = self._build_layers_one_stream(**branch_args2)

    def _build_layers_one_stream(self, num_person, num_point, in_channels=2, out_channels=64):
        data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        layers = nn.ModuleList()
        fs_layers = nn.ModuleList()
        for i in range(10):
            if i == 0:
                out_channels = 64
                residual = False
            elif i in [4, 7]:
                out_channels *= 2
                residual = True
            else:
                residual = True
            layer = PTCN_GCN_unit(in_channels, out_channels, A=self.A,
                                  residual=residual,
                                  pyramid_level=self.pyramid_level) if self.use_ptcn \
                else TCN_GCN_unit(in_channels, out_channels, A=self.A,
                                  residual=residual)
            layers.append(layer)
            in_channels = out_channels
            if self.use_fsn and i in [4, 7]:
                fs_layers.append(unit_fsn(in_channels, out_channels, in_channels, out_channels))

        final = ORegressors(out_channels, self.num_class)
        bn_init(data_bn, 1)
        return data_bn, layers, final, fs_layers

    def forward(self, x1, x2, label):
        # stream 1
        N1, C1, T1, V1, M1 = x1.size()
        x1 = x1.permute(0, 4, 3, 1, 2).contiguous().view(N1, M1 * V1 * C1, T1)
        x1 = self.data_bn1(x1)
        x1 = x1.view(N1, M1, V1, C1, T1).permute(0, 1, 3, 4, 2).contiguous().view(N1 * M1, C1, T1, V1)

        # stream 2
        N2, C2, T2, V2, M2 = x2.size()
        x2 = x2.permute(0, 4, 3, 1, 2).contiguous().view(N2, M2 * V2 * C2, T2)
        x2 = self.data_bn2(x2)
        x2 = x2.view(N2, M2, V2, C2, T2).permute(0, 1, 3, 4, 2).contiguous().view(N2 * M2, C2, T2, V2)

        fs_i = 0
        for i in range(10):
            x1 = self.layers1[i](x1)
            x2 = self.layers2[i](x2)
            if self.use_fsn and i in [4, 7]:
                x1, x2 = self.fs_layers[fs_i](x1, x2)
                fs_i += 1

        # N*M,C,T,V
        c_new = x1.size(1)
        x1 = x1.view(N1, M1, c_new, -1)
        x1 = x1.mean(3).mean(1)

        # N*M,C,T,V
        c_new = x2.size(1)
        x2 = x2.view(N2, M2, c_new, -1)
        x2 = x2.mean(3).mean(1)

        out1, loss1, logits1 = self.final1(x1, label)
        out2, loss2, logits2 = self.final2(x2, label)
        if not self.training:
            logits3 = fuse_logits(logits1, logits2, self.fusion_mode, self.fusion_weight)
            risks = torch.stack([torch.stack(out).detach().max(dim=2)[1].sum(dim=0) + 1 for out in logits3])
            out3 = risks.transpose(1, 0)
            return out1, loss1, out2, loss2, out3, 0
        else:
            return out1, loss1, out2, loss2

def _test_unit_fsn():
    x1 = torch.rand((10, 32, 3, 32))
    x2 = torch.rand((10, 32, 3, 16))
    fsn = unit_fsn(32, 64, 32, 64)
    out1, out2 = fsn(x1, x2)
    return True


def _test_unit_ptcn():
    x = torch.rand((10, 100, 5, 32))
    ptcn = unit_ptcn(100, 100, 7, 1, pyramid_level=5)
    out = ptcn(x)
    return True


def _test_model():
    x = torch.rand((10, 2, 5, 32, 1))
    model = Model([3, 4, 5], graph='graph.h36m.Graph', in_channels=2)
    out = model(x)
    return True


def _test_model2s():
    x1 = torch.rand((10, 2, 5, 32, 1))
    x2 = torch.rand((10, 2, 5, 32, 1))
    branch_args1 = dict(num_point=32, num_person=1, in_channels=2, out_channels=64)
    branch_args2 = dict(num_point=32, num_person=1, in_channels=2, out_channels=64)
    model = Model2Stream([3, 4, 5], branch_args1, branch_args2,
                    graph='graph.h36m.Graph')
    out = model(x1, x2)
    return True

def _test_logits_fusion():
    data = torch.rand((10, 1024))
    model = ORegressors(1024, [2, 3, 5])
    risks1,_,logits1 = model(data)
    risks2,_,logits2 = model(data+1)
    new_logits = fuse_logits(logits1, logits2, 'sum', 1)
    new_logits2 = fuse_logits(logits1, logits2, 'wsum', 0)
    new_logits3 = fuse_logits(logits1, logits2, 'max')
    new_logits4 = fuse_logits(logits1, logits2, 'mul')
    d1 = torch.Tensor((1, 2, 3, 4,))
    d2 = torch.Tensor((4, 3, 2, 1))
    new_d = fuse_logits(d1, d2, 'mul')

    return new_logits


if __name__ == '__main__':
    # _test_unit_fsn() # pass
    # _test_unit_ptcn() # pass
    # _test_model()
    # _test_model2s()
    _test_logits_fusion()