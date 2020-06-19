import numpy as np
import os
import torch
import argparse

from torch.utils.data import Dataset
from tqdm import tqdm

from utils.msic import import_class
from data.h36m.definitions import CLASS_NUMS, CLASS_BIASES
from data_gen.h36m_gendata import gendata

pairs = {'h36m': ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
             (1, 7), (7, 8), (8, 9), (9, 10), (10, 11),
             (1, 12), (12, 13),
             (13, 14), (14, 15), (15, 16),
             (13, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (20, 23), (23, 24),
             (13, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (28, 31), (31, 32),
             (1, 1)  # to make it contain 32 pairs
             )}


# def decompose_scores_to_binary(scores, ranges):
#     """
#
#     :param scores: a list of ordinal scores
#     :param ranges: ranges for each element in scores, ranging from 1 to r (r in range)
#     :return:
#     """
#
#     outputs = [np.array([1 if (x+1) < rank else 0 for x in range(r - 1)]) for rank, r in zip(scores, ranges)]
#     return outputs

def decompose_scores_to_binary(risk, body_parts):
    """

    :param risk: a list of 19 risks
    :param body_parts: the risks of specific body parts to be calculated/transformed, from risk score to binaries
    :return:
    """
    # np.array() is important to preserve dimensions after Dataloader
    outputs = [np.array([1 if (x + 1) < risk[b] else 0 for x in range(CLASS_NUMS[b] - 1)]) for b in body_parts]
    return outputs


def assemble_logits_to_ranks(logits):
    """
    infer the ordinal scores from the output logits (softmax)
    :param logits:
    :return:
    """

    return 0


def ordinal_loss(logits, targets):
    return 0


def gen_bone_data(joint_data):
    # number sample, position, frame, joint number, body number
    N, P, F, J, B = joint_data.shape
    new_j = len(pairs['h36m'])
    bone_data = np.zeros((N, P, F, new_j, B), dtype=np.float32)
    for v1, v2 in tqdm(pairs['h36m']):
        v1 -= 1
        v2 -= 1
        bone_data[:, :, :, v2, :] = joint_data[:, :, :, v1, :] - joint_data[:, :, :, v2, :]
    return bone_data


class H36MSkel2D2Stream(Dataset):
    """
        loading both joint stream and bone stream
    """
    def __init__(self, body_parts, data_type='2stream', load_data_args=dict(), debug=False, phase='train'):
        train_data, train_label, val_data, val_label, test_data, test_label = gendata(**load_data_args)
        if phase == 'train':
            joint_data = train_data
            label = train_label
        elif phase == 'val':
            joint_data = val_data
            label = val_label
        elif phase == 'test':
            joint_data = test_data
            label = test_label
        else:
            raise ValueError(f"No such phase {phase}")

        self.joint_data = joint_data
        self.label = label
        self.pairs = pairs
        self.body_parts = body_parts
        self.data_type = data_type

        if debug:
            self.label = self.label[0:100]
            self.joint_data = self.joint_data[0:100]

        if data_type == 'bone' or data_type == '2stream':
            self.bone_data = gen_bone_data(joint_data)
        else:
            self.bone_data = None

    def get_mean_map(self):
        data = self.joint_data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.joint_data)

    def __getitem__(self, index):
        skel2d_joint = self.joint_data[index]
        risk = self.label[index] - CLASS_BIASES + 1
        target = decompose_scores_to_binary(risk, self.body_parts)

        if self.data_type == 'joint':
            return skel2d_joint, risk, target
        elif self.data_type == 'bone':
            skel2d_bone = self.bone_data[index]
            return skel2d_bone, risk, target
        elif self.data_type == '2stream':
            skel2d_bone = self.bone_data[index]
            return skel2d_joint, skel2d_bone, risk, target
        else:
            raise ValueError(f"No such data type {self.data_type}")


def test(data_path, index=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path:
    :param label_path:
    :param vid: the id of sample
    :param graph:
    :param is_3d: when vis NTU, set it True
    :return:
    '''
    import matplotlib.pyplot as plt

    dataloader = torch.utils.data.DataLoader(
        dataset=H36MSkel2D2Stream(body_parts=[1, 2, 3],
                                  load_data_args=dict(data_path=data_path, max_frame=3, viewpoint='front'),
                                  data_type='joint',
                                  debug=True,
                                  phase='train'),
        batch_size=10,
        shuffle=False,
        num_workers=1)

    if index is not None:
        data, _, label = dataloader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # fig.savefig('../graph/pics/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)

def _test_decompose_scores_to_binary():
    scores = [1, 2, 3]
    ranges = [2, 3, 4]
    output = decompose_scores_to_binary(scores, ranges)
    return True

if __name__ == '__main__':
    # _test_decompose_scores_to_binary()
    parser = argparse.ArgumentParser(description='H36M data converter.')
    parser.add_argument('--data_path', default='../data/h36m/data_all_ver2.npy')
    parser.add_argument('--viewpoint', default='front')
    parser.add_argument('--max_frame', type=int, default=9, choices=[3, 5, 7, 9, 11])

    args = parser.parse_args()

    graph = 'graph.h36m.Graph'
    os.makedirs('../graph/pics/skeleton_sequence/', exist_ok=True)
    test(args.data_path, 0, graph)