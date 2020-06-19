import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['Classifiers']

class Classifiers(nn.ModuleList):
    def __init__(self, in_dim, k=[3], body_parts=[]):
        super(Classifiers, self).__init__()
        self.k = k
        assert isinstance(k, list), "args k should be a list"
        self.body_parts = body_parts
        for ki in k:
            self.append(nn.Linear(in_dim, ki, bias=False))

    def forward(self, x, scores=None):
        output = [m(x) for m in self.children()]
        if scores is not None:
            scores = scores[:, self.body_parts]
            scores = torch.transpose(scores, 1, 0).long() - 1
            loss = torch.stack([F.cross_entropy(out, tar) for out, tar in zip(output, scores)]).sum()
        else:
            loss = 0

        risks = torch.stack([torch.max(out, dim=1)[1] + 1 for out in output]).transpose(1, 0).detach()
        return risks, loss


if __name__ == '__main__':
    from dataloaders import make_dataloaders
    from utils.config import get_args
    from utils.misc import get_available_device

    args, _ = get_args()
    args.device = get_available_device(0)
    _, _, _, test_loader, num_classes = make_dataloaders(args)

    data = torch.randn((args.test_batch_size, 1024))
    # targets = torch.FloatTensor([[2, 3, 4], [2, 3, 4]])
    _, scores, targets = test_loader.get_batch()
    del test_loader

    model = Classifiers(in_dim=1024, k=num_classes, body_parts=args.body_parts)
    output, loss = model(data, targets=scores)

    print('done')
