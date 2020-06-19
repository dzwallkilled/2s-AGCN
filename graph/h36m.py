
from graph import tools

num_node = 32
self_link = [(i, i) for i in range(num_node)]

inward_ori_index = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                    (1, 7), (7, 8), (8, 9), (9, 10), (10, 11),
                    (1, 12), (12, 13),
                    (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (20, 23), (23, 24),
                    (13, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (28, 31), (31, 32)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)