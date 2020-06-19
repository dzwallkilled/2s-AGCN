import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

__all__ = ['decompose_target', 'assemble_logits_to_ordinal_ranks', 'Metrics']


# for ordinal regression, decompose target into num-1 binaries
def decompose_target(target, num):
    output = [np.array([1 if (x + 1) < target[b] else 0 for x in range(num - 1)]) for b in
              range(len(target))]
    return torch.from_numpy(np.array(output).transpose((1, 0)))


# def get_rank_deprecated(logits):
#     rank_p = []
#     for out in logits:
#         # max()[1] returns the index
#         predicts = torch.cat([d.cpu().max(dim=1)[1].view(-1, 1) for d in out], 1)
#         rank_p.append(predicts.sum(dim=1).long().numpy() + 1)
#
#     rank_p = np.array(rank_p)
#     return np.transpose(rank_p, [1, 0])

def assemble_logits_to_ordinal_ranks(logits):
    """rewrite the function in more efficient way"""
    rank_p = torch.stack([torch.stack(out).detach().max(dim=2)[1].sum(dim=0) + 1 for out in logits])
    rank_p = rank_p.transpose(1, 0)
    return rank_p.cpu().long().numpy()


class Metrics(object):
    def __init__(self, body_parts, num_classes):
        super(Metrics, self).__init__()
        self.predicts = []
        self.targets = []
        self.body_parts = body_parts
        self.labels = [range(1, n + 1) for n in num_classes]
        pass

    def reset(self):
        self.predicts = []
        self.targets = []

    def update1(self, logits, scores):
        predicts = assemble_logits_to_ordinal_ranks(logits)
        self.predicts.append(predicts)
        self.targets.append(scores.cpu().long().numpy())

    def update2(self, preds, scores):
        self.predicts.append(preds.cpu().long().numpy())
        self.targets.append(scores.cpu().long().numpy())

    def metrics(self):
        target_scores = np.concatenate(self.targets)  # shape: N x B
        predict_scores = np.concatenate(self.predicts)
        # with open('./scores_stn2_front_image.log', 'w') as f:
        #     for cnt, (t, p) in enumerate(zip(target_scores, predict_scores)):
        #         f.write(f'{cnt} {[t[b] for b in self.body_parts]} {p}\n')
        mae = []
        kappa = []
        accuracy = []
        for idx, body_idx in enumerate(self.body_parts):
            p = predict_scores[:, idx]
            t = target_scores[:, body_idx]
            mae.append(np.mean(np.abs(p - t)))
            kappa.append(cohen_kappa_score(p, t, labels=self.labels[idx]))
            accuracy.append(np.sum(p == t) / len(p) * 100.)

        return mae, kappa, accuracy

    def __len__(self):
        return len(np.concatenate(self.targets))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    # import sys
    # import os
    # from os.path import join
    #
    # sys.path.append(os.getcwd())
    # sys.path.append(join(os.getcwd(), '..'))
    # pass
    #
    # from dataloaders import make_dataloaders
    # from networks.child_cnn import Child
    # from networks.controller import Controller
    #
    # from dataloaders.datasets import CLASS_NUMS
    # from utils.config import get_args
    # from utils.loss import MultiTaskLoss
    #
    # args, _ = get_args()
    # # args.body_parts = [args.body_parts[0]]
    # device = 'cuda:0'
    # num_classes = [CLASS_NUMS[v] for v in args.body_parts]
    # criterion = MultiTaskLoss(num_classes=num_classes, body_parts=args.body_parts)
    # criterion = criterion.to(device)
    #
    # child = Child(args, num_classes).to(args.device)
    # controller = Controller(args).to(args.device)
    #
    # metrics = Metrics(args.body_parts, num_classes)
    #
    # train_data, val_data, *_ = make_dataloaders(args)
    # dag, *_ = controller()
    # with torch.no_grad():
    #     for _ in range(len(train_data)):
    #         images, scores, _ = train_data.get_batch()
    #         outputs = child(images, dag[0])
    #         metrics.update(logits=outputs, scores=scores)
    #
    # mae, kappa, accuracy = metrics.metrics()
    # metrics.reset()
    # print('train_data', mae, kappa, accuracy)
    #
    # dag, *_ = controller()
    # with torch.no_grad():
    #     for _ in range(len(val_data)):
    #         images, scores, _ = val_data.get_batch()
    #         outputs = child(images, dag[0])
    #         metrics.update(logits=outputs, scores=scores)
    #
    # mae, kappa, accuracy = metrics.metrics()
    # metrics.reset()
    # print('val_data', mae, kappa, accuracy)
    pass
