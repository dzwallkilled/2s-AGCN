import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['ORegressor', 'ORegressors']


def ordinal_loss(logits, targets):
    loss = torch.stack([F.cross_entropy(torch.stack(out, 2), tar) for out, tar in zip(logits, targets)])
    loss = torch.mul(loss, self.w).sum()


class ORegressor(nn.ModuleList):
    def __init__(self, in_dim, k=3):
        super(ORegressor, self).__init__()
        self.k = k
        for i in range(k - 1):
            self.append(nn.Linear(in_dim, 2, bias=False))
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, targets=None):
        output = [m(x) for m in self.children()]
        if targets is not None:
            loss = torch.stack([F.cross_entropy(out, tar) for out, tar in zip(output, targets)]).mean()
            return loss
        else:
            return output, 0


class ORegressors(nn.ModuleList):
    def __init__(self, in_dim, k=[3]):
        super(ORegressors, self).__init__()
        self.k = k
        assert isinstance(k, list), "args k should be a list"
        self.w = nn.Parameter(torch.Tensor([n - 1 for n in self.k]), requires_grad=False)
        for i in k:
            self.append(ORegressor(in_dim, k=i))

    def forward(self, x, targets=None):
        output = [m(x)[0] for m in super(ORegressors, self).children()]
        if targets is not None:
            loss = torch.stack([F.cross_entropy(torch.stack(out, 2), tar) for out, tar in zip(output, targets)])
            loss = torch.mul(loss, self.w).sum()
        else:
            loss = 0
        risks = torch.stack([torch.stack(out).detach().max(dim=2)[1].sum(dim=0) + 1 for out in output])
        risks = risks.transpose(1, 0)
        return risks, loss, output


if __name__ == '__main__':
    from utils.msic import get_available_device
    from data.h36m.definitions import JOINTS_OF_PARTS
    from feeders.h36m_dataset import decompose_scores_to_binary, assemble_logits_to_scores, ordinal_loss
    from utils.metrics import Metrics

    body_parts = list(JOINTS_OF_PARTS.keys())

    num_classes = [2, 3, 4]
    batch_size = 2
    data = torch.randn((2, 128))
    tscores = torch.FloatTensor([[1, 2, 3], [2, 3, 4]])
    tlogits = decompose_scores_to_binary(risk=tscores, body_parts=body_parts)

    model = ORegressors(in_dim=128, k=num_classes)
    loss = model(data, targets=tlogits)

    print('done')
