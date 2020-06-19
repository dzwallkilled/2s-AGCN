import argparse
from tqdm import tqdm
import sys
import numpy as np
import os

from utils.logger import get_logger

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization
from data.h36m.definitions import *

max_body_true = 1
max_body_kinect = 4
num_joint = 32

logger = get_logger()


def split_data(data, train_ref, val_ref, test_ref, viewpoint):
    """

    :param data: a pre-calculated numpy file, storing all the information of the dataset except the images
    :param train_ref: references of the train set
    :param val_ref: references of the val set
    :param test_ref: references of the test set
    :param viewpoint: 'front', 'side', 'back'
    :return: trainset, valset, testset
    """
    if isinstance(viewpoint, str):
        viewpoint = VIEWS[viewpoint]

    data_train = data[np.concatenate([train_ref + i * TOTAL_SAMPLES for i in range(4)], 0)]
    if viewpoint:
        data_train = [e for e in data_train if e['viewpoint'] == viewpoint]

    data_val = data[np.concatenate([val_ref + i * TOTAL_SAMPLES for i in range(4)], 0)]
    if viewpoint:
        data_val = [e for e in data_val if e['viewpoint'] == viewpoint]

    data_test = data[np.concatenate([test_ref + i * TOTAL_SAMPLES for i in range(4)], 0)]
    if viewpoint:
        data_test = [e for e in data_test if e['viewpoint'] == viewpoint]

    return data_train, data_val, data_test


def assemble_data(data, max_frame):
    out_data = np.zeros((len(data), 2, max_frame, num_joint, max_body_true), dtype=np.float32)
    for i, d in enumerate(tqdm(data)):
        skel2D_seq = d['skel2DPosSeq'] # totally 11 frames. [11, 64]
        frame, dim = skel2D_seq.shape
        assert dim == 64 and frame == 11, f"Something is wrong with data"
        # selecting neighbouring frames, total frames (including self) max_frame
        skel2D_seq = skel2D_seq[5-(max_frame-1)//2:5+(max_frame-1)//2+1, :]
        skel2D_seq = np.reshape(skel2D_seq, [max_frame, 32, 2, 1])  # [frame, joint, pos, body]
        skel2D_seq = np.transpose(skel2D_seq, [2, 0, 1, 3])
        out_data[i, :, :, :, :] = skel2D_seq

    return out_data


def parse_label(data):
    output_label = np.zeros([len(data), 19], dtype=np.int32)
    for i, d in enumerate(tqdm(data)):
        output_label[i, :] = d['scoresGT']
    return output_label


def gendata(data_path, out_path='', max_frame=3, viewpoint='front', seed=123, train_ratio=0.8, val_ratio=0.2):
    """

    :param data_path:
    :param out_path: a dir, if given, save data to out_path/
    :param max_frame:
    :param viewpoint:
    :param seed:
    :param train_ratio:
    :param val_ratio:
    :return: if out_path is '', return converted data, including training, val, and test.
    """
    # load data, which should be data_all_ver2.npy
    data_all = np.load(data_path, allow_pickle=True, encoding='latin1')
    total_samples = len(data_all) // 4
    assert total_samples == TOTAL_SAMPLES, f"Incorrect number of samples in data."

    np.random.seed(seed)
    references = np.random.permutation(total_samples)
    n = int(np.round(total_samples * train_ratio))
    vn = int(np.round(n * val_ratio))
    train_ref = references[:n]
    val_ref = references[:vn]
    test_ref = references[n:]

    logger.info(f'Train ({len(train_ref)})| ' + ' '.join(str(i) for i in train_ref[:10]) + ' ... ' + str(
        train_ref[-1]))
    logger.info(
        f'Val ({len(val_ref)})| ' + ' '.join(str(i) for i in val_ref[:10]) + ' ... ' + str(val_ref[-1]))
    logger.info(
        f'Test ({len(test_ref)})| ' + ' '.join(str(i) for i in test_ref[:10]) + ' ... ' + str(test_ref[-1]))

    train_data, val_data, test_data = split_data(data_all, train_ref, val_ref, test_ref, viewpoint)

    out_data_train = assemble_data(train_data, max_frame)
    out_label_train = parse_label(train_data)
    out_data_val = assemble_data(val_data, max_frame)
    out_label_val = parse_label(val_data)
    out_data_test = assemble_data(test_data, max_frame)
    out_label_test = parse_label(test_data)

    if out_path is not '':
        np.save(f"{out_path}/train_skel2d_{viewpoint}_{max_frame}.npy", out_data_train)
        logger.info(f"saved file: {out_path}/train_skel2d_{viewpoint}_{max_frame}.npy")

        np.save(f"{out_path}/val_skel2d_{viewpoint}_{max_frame}.npy", out_data_val)
        logger.info(f"saved file: {out_path}/val_skel2d_{viewpoint}_{max_frame}.npy")

        np.save(f"{out_path}/test_skel2d_{viewpoint}_{max_frame}.npy", out_data_test)
        logger.info(f"saved file: {out_path}/test_skel2d_{viewpoint}_{max_frame}.npy")
    else:
        return out_data_train, out_label_train, out_data_val, out_label_val, out_data_test, out_label_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='H36M data converter.')
    parser.add_argument('--data_path', default='../data/h36m/data_all_ver2.npy')
    parser.add_argument('--out_folder', default='../data/h36m/')
    parser.add_argument('--viewpoint', default='front')
    parser.add_argument('--max_frame', type=int, default=9, choices=[3, 5, 7, 9, 11])

    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)
    gendata(args.data_path,
            args.out_folder,
            args.max_frame,
            args.viewpoint)
    pass
