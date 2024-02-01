import logging

import torch
from torch import nn


class SpatialBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, A, residual=False, edge_importance=True, adaptive=False, **kwargs):
        super(SpatialBasicBlock, self).__init__()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        self.conv = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if adaptive:
            self.A = nn.Parameter(A[:max_graph_distance+1], requires_grad=True)
        else:
            self.register_buffer('A', A[:max_graph_distance+1])
        self.edge = nn.Parameter(torch.ones_like(A[:max_graph_distance+1]),requires_grad=edge_importance)

    def forward(self, x):

        res_block = self.residual(x)

        x = self.conv(x, self.A*self.edge)
        x = self.bn(x)
        x = self.relu(x + res_block)

        return x


class TemporalBasicBlock(nn.Module):
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, **kwargs):
        super(TemporalBasicBlock, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels),
            )

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)

        return x


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance):
        super(SpatialGraphConv, self).__init__()

        # spatial class number (distance = 0 for class 0, distance = 1 for class 1, ...)
        self.s_kernel_size = max_graph_distance + 1

        # weights of different spatial classes
        self.gcn = nn.Conv2d(in_channels, out_channels*self.s_kernel_size, 1)

    def forward(self, x, A):

        # numbers in same class have same weight
        x = self.gcn(x)
        # print(x.size())
        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        # logging.info(f"SpatialGraphConv: {x.size()}")  # 32, 3, 64, 1, 17
        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()
        # logging.info(f"After einsum: {x.size()}")  # 32, 64, 1, 17
        return x


class ResGCNModule(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_block, temporal_block, A, initial=False, stride=1, kernel_size=(1, 2), **kwargs):
        super(ResGCNModule, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if initial:
            module_res, block_res = False, False
        elif spatial_block == 'Basic' and temporal_block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.scn = SpatialBasicBlock(in_channels, out_channels, max_graph_distance, A, block_res, **kwargs)
        self.tcn = TemporalBasicBlock(out_channels, temporal_window_size, stride, block_res, **kwargs)

    def forward(self, x):
        return self.tcn(self.scn(x), self.residual(x))


class ResGCNInputBranch(nn.Module):
    def __init__(self, structure, spatial_block, temporal_block, num_channel, A, **kwargs):
        super(ResGCNInputBranch, self).__init__()

        module_list = [ResGCNModule(num_channel, 64, 'Basic', 'Basic', A, initial=True, **kwargs)]
        module_list += [ResGCNModule(64, 64, 'Basic', 'Basic', A, initial=True, **kwargs) for _ in range(structure[0] - 1)]
        module_list += [ResGCNModule(64, 64, spatial_block, temporal_block, A, **kwargs) for _ in range(structure[1] - 1)]
        module_list += [ResGCNModule(64, 32, spatial_block, temporal_block, A, **kwargs)]

        self.bn = nn.BatchNorm2d(num_channel)
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):
        # print(x.size())  # [1, 1, 3, 1, 17, 2]
        N, C, T, V, M = x.size()
        # N, M, V, C = x.size()
        # x = x.permute(0, 3, 2, 1)  # N, C, V, M
        # x = x[:, :, None, :, :]
        # N, C, T, V, M = x.size()

        x = self.bn(x.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V))
        # x = self.bn(x.permute(0, 1, 3, 2).contiguous().view(N * M, C, V, 1))
        for layer in self.layers:
            x = layer(x)

        return x


def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #m.bias = None
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def zero_init_lastBN(modules):
    for m in modules:
        if isinstance(m, ResGCNModule):
            if hasattr(m.scn, 'bn_up'):
                nn.init.constant_(m.scn.bn_up.weight, 0)
