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

        # divide nodes into different classes
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

        # spatial graph convolution
        x = torch.einsum('nkctv,kvw->nctw', (x, A[:self.s_kernel_size])).contiguous()

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
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        spatial_block = U.import_class('src.model.TPGCN.blocks.Spatial_{}_Block'.format(spatial_block))
        temporal_block = U.import_class('src.model.TPGCN.blocks.Temporal_{}_Block'.format(temporal_block))
        if initial and 'adaptive' in kwargs:
            kwargs['adaptive'] == False
        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, A, block_res, **kwargs)
        self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res, **kwargs)

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

        N, C, T, V, M = x.size()
        x = self.bn(x.permute(0,4,1,2,3).contiguous().view(N*M, C, T, V))
        for layer in self.layers:
            x = layer(x)

        return x


class TPGCN(nn.Module):
    def __init__(self, module, structure, spatial_block, temporal_block, data_shape, num_class, A, **kwargs):
        super(TPGCN, self).__init__()

        num_input, num_channel, _, _, _ = data_shape

        # input branches
        self.input_branches = nn.ModuleList([
            ResGCNInputBranch(structure, spatial_block, temporal_block, num_channel, A, **kwargs)
            for _ in range(num_input)
        ])

        # main stream
        module_list = [module(32*num_input, 128, spatial_block, temporal_block, A, stride=2, **kwargs)]
        module_list += [module(128, 128, spatial_block, temporal_block, A, **kwargs) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, spatial_block, temporal_block, A, stride=2, **kwargs)]
        module_list += [module(256, 256, spatial_block, temporal_block, A, **kwargs) for _ in range(structure[3] - 1)]
        self.main_stream = nn.ModuleList(module_list)

        # output
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(256, num_class)

        # init parameters
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x):

        N, I, C, T, V, M = x.size()

        # input branches
        x_cat = []
        for i, branch in enumerate(self.input_branches):
            x_cat.append(branch(x[:,i,:,:,:,:]))
        x = torch.cat(x_cat, dim=1)

        # main stream
        for layer in self.main_stream:
            x = layer(x)

        # extract feature
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        # output
        x = self.global_pooling(x)
        x = x.view(N, M, -1).mean(dim=1)
        x = self.fcn(x)

        return x, feature


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
            if hasattr(m.tcn, 'bn_up'):
                nn.init.constant_(m.tcn.bn_up.weight, 0)