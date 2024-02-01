import os, logging, numpy as np


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
class Graph:
    def __init__(self, dataset, graph, labeling, num_person_out=1, max_hop=10, dilation=1, normalize=True, threshold=0.2, **kwargs):
        self.dataset = dataset
        self.labeling = labeling
        self.graph = graph
        if labeling not in ['spatial', 'distance', 'zeros', 'ones', 'eye', 'pairwise0', 'pairwise1', 'geometric']:
            logging.info('')
            logging.error('Error: Do NOT exist this graph labeling: {}!'.format(self.labeling))
            raise ValueError()
        self.normalize = normalize
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_person_out = num_person_out
        self.threshold = threshold

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts, self.center = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset in ['flickr', 'youth']:
            num_node = 17
            neighbor_link = [(10, 8), (8, 6), (9, 7), (7, 5), (1, 13), (13, 11),
                             (16, 14), (14, 12), (11, 5), (12, 6), (5, 3), (6, 4),
                             (1, 0), (2, 0), (3, 1), (4, 2), (12, 11), (5, 6)]
            connect_joint = np.array([0, 0, 5, 6, 14, 0, 7, 12, 11, 0, 2, 5, 13, 0, 0, 6, 8])
            parts = [
                np.array([5, 7, 9]),  # left_arm
                np.array([6, 8, 10]),  # right_arm
                np.array([11, 13, 15]),  # left_leg
                np.array([12, 14, 16]),  # right_leg
                np.array([0, 1, 2, 3, 4])  # torso
            ]
            center = 0
        else:
            logging.info('')
            logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts, center

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        self.oA = A
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):

        if self.labeling == 'distance':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        elif self.labeling == 'spatial':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if hop_dis[j, i] == hop:
                            # if hop_dis[j, self.center] == np.inf or hop_dis[i, self.center] == np.inf:
                            #     continue
                            if hop_dis[j, self.center] == hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif hop_dis[j, self.center] > hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)

        elif self.labeling == 'zeros':
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))

        elif self.labeling == 'ones':
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.ones((len(valid_hop), self.num_node, self.num_node))
            for i in range(len(valid_hop)):
                A[i] = self._normalize_digraph(A[i])

        elif self.labeling == 'eye':

            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i in range(len(valid_hop)):
                A[i] = self._normalize_digraph(np.eye(self.num_node, self.num_node))

        elif self.labeling == 'pairwise0':
            # pairwise0: only pairwise inter-body link
            assert 'mutual' in self.graph

            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            v = self.num_node // 2
            for i in range(len(valid_hop)):
                A[i, v:, :v] = np.eye(v, v)
                A[i, :v, v:] = np.eye(v, v)
                A[i] = self._normalize_digraph(A[i])

        elif self.labeling == 'pairwise1':
            assert 'mutual' in self.graph
            v = self.num_node // 2
            self.edge += [(i, i + v) for i in range(v)]
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        elif self.labeling == 'geometric':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1

            geometric_matrix = np.load(os.path.join(os.getcwd(), 'src/dataset/a.npy'))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if geometric_matrix[i, j] > self.threshold:
                        adjacency[i, j] += geometric_matrix[i, j]
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        return A

    @staticmethod
    def _normalize_digraph(A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD
