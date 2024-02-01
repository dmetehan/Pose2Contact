import logging

import torch
from torch import nn

from .base_gcn import ResGCNModule, ResGCNInputBranch, init_param, zero_init_lastBN


class DoubleTPGCN(nn.Module):
    def __init__(self, module, structure, spatial_block, temporal_block, data_shape, num_class, A, **kwargs):
        """

        :param module:
        :param structure:
        :param spatial_block:
        :param temporal_block:
        :param data_shape:
        :param num_class: For contact binary estimation this should correspond to the number of regions in one person
        :param A:
        :param kwargs:
        """
        self.num_class = num_class
        super(DoubleTPGCN, self).__init__()
        # logging.info(f"data_shape: {data_shape}")
        _, _, num_channel = data_shape
        num_input = 1

        # input branches
        self.input_branches = nn.ModuleList([
            ResGCNInputBranch(structure, spatial_block, temporal_block, num_channel, A, **kwargs)
            for _ in range(num_input)
        ])

        # adult stream
        module_list = [module(32*num_input, 128, spatial_block, temporal_block, A, stride=2, **kwargs)]
        module_list += [module(128, 128, spatial_block, temporal_block, A, **kwargs) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, spatial_block, temporal_block, A, stride=2, **kwargs)]
        module_list += [module(256, 256, spatial_block, temporal_block, A, **kwargs) for _ in range(structure[3] - 1)]
        self.adult_stream = nn.ModuleList(module_list)

        # child stream
        module_list = [module(32*num_input, 128, spatial_block, temporal_block, A, stride=2, **kwargs)]
        module_list += [module(128, 128, spatial_block, temporal_block, A, **kwargs) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, spatial_block, temporal_block, A, stride=2, **kwargs)]
        module_list += [module(256, 256, spatial_block, temporal_block, A, **kwargs) for _ in range(structure[3] - 1)]
        self.child_stream = nn.ModuleList(module_list)

        # output child
        self.child_gpooling = nn.AdaptiveAvgPool2d(1)
        self.child_fcn = nn.Linear(256, num_class)

        # output adult
        self.adult_gpooling = nn.AdaptiveAvgPool2d(1)
        self.adult_fcn = nn.Linear(256, num_class)

        # init parameters
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x):
        logging.debug(f"shape of the input is {x.shape}")
        # N, I, C, T, V, M = x.size()  # N:batch size, I:no of input types, C:no of channel, T:frames, V:no of joints, M:no of people
        N, M, V, C = x.size()
        x = x.permute(0, 3, 2, 1)  # N, C, V, M
        x = x[:, None, :, None, :, :]  # N, I, C, T, V, M
        N, I, C, T, V, M = x.size()
        # logging.info(f"N:{N}, I:{I}, C:{C}, T:{T}, V:{V}, M:{M}")

        # input branches
        x_cat = []
        for i, branch in enumerate(self.input_branches):
            x_cat.append(branch(x[:,i,:,:,:,:]))
        x = torch.cat(x_cat, dim=1)

        logging.debug(f"shape after the input branch is {x.shape}")

        xadult = x
        xchild = x
        # main stream
        for layer in self.adult_stream:
            xadult = layer(xadult)

        # main stream
        for layer in self.child_stream:
            xchild = layer(xchild)

        # extract feature
        _, C, T, V = xchild.size()
        child_feature = xchild.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        # output
        xchild = self.child_gpooling(xchild)
        xchild = xchild.view(N, M, -1).mean(dim=1)
        xchild = self.child_fcn(xchild)


        # extract feature
        _, C, T, V = xadult.size()
        adult_feature = xadult.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        # output
        xadult = self.adult_gpooling(xadult)
        xadult = xadult.view(N, M, -1).mean(dim=1)
        xadult = self.adult_fcn(xadult)

        logging.debug(f"shape of the xchild is {xchild.shape}")

        x = torch.bmm(xchild.view(-1, self.num_class, 1), xadult.view(-1, 1, self.num_class))
        x = torch.reshape(x, (-1, self.num_class*self.num_class))

        logging.debug(f"shape of the output is {x.shape}")

        feature = torch.cat([child_feature, adult_feature], dim=1)

        logging.debug(f"shape of the feature is {feature.shape}")

        return x, feature

    @staticmethod
    def create(_, block_structure, att_type, reduction='r1', **kwargs):
        structure = {'structure': [1, 2, 3, 3], 'spatial_block': 'Basic', 'temporal_block': 'Basic'}
        __reduction = {
            'r1': {'reduction': 1},
            'r2': {'reduction': 2},
            'r4': {'reduction': 4},
            'r8': {'reduction': 8},
        }
        kwargs.update({'module': ResGCNModule, 'attention': None})
        return DoubleTPGCN(**structure, **(__reduction[reduction]), **kwargs)
