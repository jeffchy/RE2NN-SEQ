import torch
from torch import nn


class PriorityLayer(nn.Module):
    def __init__(self, C, priority_mat=None, priority_bias=None):
        super(PriorityLayer, self).__init__()

        if priority_mat is None:
            self.priority_mat = nn.Parameter(torch.eye(C).float(), requires_grad=False)
            self.priority_bias = nn.Parameter(torch.zeros(C).float(), requires_grad=False)
        else:
            base = torch.eye(C).float()
            origin = torch.from_numpy(priority_mat).float()
            origin_C = origin.shape[0]
            base[:origin_C, :origin_C] = origin
            self.priority_mat = nn.Parameter(base, requires_grad=False)
            self.priority_bias = nn.Parameter(torch.zeros(C).float(), requires_grad=False)

    def forward(self, scores):
        """
        :param scores: B x L x C
        :return: new scores: B x L x C
        """
        if len(scores.shape) > 2:
            out = torch.einsum('blc,cd->bld', scores, self.priority_mat)
            out = out + self.priority_bias
        else:
            out = torch.einsum('bc,cd->bd', scores, self.priority_mat)
            out = out + self.priority_bias
        return out