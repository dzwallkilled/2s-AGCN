from os import listdir, makedirs
from os.path import join
import h5py
import numpy as np
from tqdm import tqdm

def add_skel2DPosSeq_2_data_all():
    data = np.load('data_all.npy', allow_pickle=True, encoding="latin1")
    # open the mat file that contains 2d skeleton data of all frames
    # f is a dict, a key is a sample in f, with data contained in value.
    bar = tqdm(data)
    with h5py.File('data_2DPos.mat') as f:
        for d in data:
            sample_name = f"S{d['subject']:03d}A{d['action']:03d}SA{d['subaction']:03d}C{d['camera']:03d}"
            frame_num = d['fno'] - 1 # changed to 0-based index
            skel_2d_seq = f.get(sample_name)
            assert (d['skel2DPos'] == skel_2d_seq[frame_num, :]).all()
            if frame_num - 5 < 0:
                seq = [skel_2d_seq[0]]*5
                seq = np.concatenate([seq,skel_2d_seq[0:frame_num+6,:]])
            elif frame_num + 6 > len(skel_2d_seq):
                remained_frames = frame_num + 6 - len(skel_2d_seq) + 1
                seq = [skel_2d_seq[-1]] * remained_frames
                seq = np.concatenate([skel_2d_seq[frame_num-5:-1], seq])
            else:
                seq = skel_2d_seq[frame_num-5:frame_num+6, :]
            d['skel2DPosSeq'] = seq
            bar.update()
            pass

    np.save('data_all_ver2.npy', data)
    pass


if __name__ == '__main__':
    # add_skel2DPosSeq_2_data_all()
    data = np.load('data_all_ver2.npy', allow_pickle=True, encoding='latin1')
    pass

