import logging

import torch
from torch import nn

from .base_gcn import ResGCNModule, ResGCNInputBranch, init_param, zero_init_lastBN


class TPGCN(nn.Module):
    def __init__(self, module, structure, spatial_block, temporal_block, data_shape, num_class, A, **kwargs):
        super(TPGCN, self).__init__()
        logging.info(f"data_shape: {data_shape}")
        _, _, num_channel = data_shape
        num_input = 1

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

        # main stream
        for layer in self.main_stream:
            x = layer(x)

        logging.debug(f"shape after the main stream is {x.shape}")

        # extract feature
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        logging.debug(f"shape after extracting features is {x.shape}")

        # output
        x = self.global_pooling(x)
        x = x.view(N, M, -1).mean(dim=1)
        x = self.fcn(x)

        logging.debug(f"shape of the output is {x.shape}")

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
        return TPGCN(**structure, **(__reduction[reduction]), **kwargs)

