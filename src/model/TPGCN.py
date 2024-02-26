import logging

import torch
from torch import nn

from .base_gcn import ResGCNModule, ResGCNInputBranch, init_param, zero_init_lastBN


class TPGCN(nn.Module):
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
        super(TPGCN, self).__init__()
        # logging.info(f"data_shape: {data_shape}")
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
        module_list += [module(256, 512, spatial_block, temporal_block, A, **kwargs) for _ in range(structure[3] - 1)]
        self.main_stream = nn.ModuleList(module_list)

        # output child
        # self.child_gpooling = nn.AdaptiveAvgPool2d(1)
        # self.child_fcn21 = nn.Linear(256, 21)
        # self.child_fcn6 = nn.Linear(256, 6)
        # self.child_fcn21x21 = nn.Linear(256, 21)
        # self.child_fcn6x6 = nn.Linear(256, 6)
        #
        # # output adult
        # self.adult_gpooling = nn.AdaptiveAvgPool2d(1)
        # self.adult_fcn21 = nn.Linear(256, 21)
        # self.adult_fcn6 = nn.Linear(256, 6)
        # self.adult_fcn21x21 = nn.Linear(256, 21)
        # self.adult_fcn6x6 = nn.Linear(256, 6)

        # output together
        self.gpooling = nn.AdaptiveAvgPool2d(1)
        self.fcn42 = nn.Linear(512, 42)
        self.fcn12 = nn.Linear(512, 12)
        self.fcn21x21 = nn.Linear(512, 21*21)
        self.fcn6x6 = nn.Linear(512, 6*6)

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

        # extract feature
        _, C, T, V = x.size()
        features = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)

        # # output
        # xchild = self.child_gpooling(x)
        # xchild = xchild.view(N, M, -1).mean(dim=1)
        # xchild21 = self.child_fcn21(xchild)
        # xchild6 = self.child_fcn6(xchild)
        # xchild21x21 = self.child_fcn21x21(xchild)
        # xchild6x6 = self.child_fcn6x6(xchild)
        #
        # # output
        # xadult = self.adult_gpooling(x)
        # xadult = xadult.view(N, M, -1).mean(dim=1)
        # xadult21 = self.adult_fcn21(xadult)
        # xadult6 = self.adult_fcn6(xadult)
        # xadult21x21 = self.adult_fcn21x21(xadult)
        # xadult6x6 = self.adult_fcn6x6(xadult)

        # output
        x = self.gpooling(x)
        x = x.view(N, M, -1).mean(dim=1)
        x42 = self.fcn42(x)
        x12 = self.fcn12(x)
        x21x21 = self.fcn21x21(x)
        x6x6 = self.fcn6x6(x)

        # x21x21 = self.fcn21x21(x)
        # x42 = torch.cat((x21x21.view(-1, 21, 21).max(dim=1).values, x21x21.view(-1, 21, 21).max(dim=2).values), dim=1)
        # x12 = self.fcn12(x)
        # x6x6 = self.fcn6x6(x)

        # # xadult * xchild.T
        # x21x21 = torch.bmm(xadult21x21.view(-1, 21, 1), xchild21x21.view(-1, 1, 21))
        # x21x21 = torch.reshape(x21x21, (-1, 21*21))
        #
        # x6x6 = torch.bmm(xadult6x6.view(-1, 6, 1), xchild6x6.view(-1, 1, 6))
        # x6x6 = torch.reshape(x6x6, (-1, 6*6))
        #
        # x42 = torch.cat((xadult21, xchild21), dim=1)
        # x12 = torch.cat((xadult6, xchild6), dim=1)

        logging.debug(f"shape of the output is {x.shape}")
        logging.debug(f"shape of the feature is {features.shape}")

        return (x42, x12, x21x21, x6x6), features

    @staticmethod
    def create(_, block_structure, att_type, reduction='r1', **kwargs):
        # structure = {'structure': [1, 2, 3, 3], 'spatial_block': 'Basic', 'temporal_block': 'Basic'}
        structure = {'structure': [1, 2, 2, 2], 'spatial_block': 'Basic', 'temporal_block': 'Basic'}
        __reduction = {
            'r1': {'reduction': 1},
            'r2': {'reduction': 2},
            'r4': {'reduction': 4},
            'r8': {'reduction': 8},
        }
        kwargs.update({'module': ResGCNModule, 'attention': None})
        return TPGCN(**structure, **(__reduction[reduction]), **kwargs)
