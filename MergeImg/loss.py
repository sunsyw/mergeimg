import torch.nn as nn
import torch.nn.functional as F
import torch


"""
forward中的self.loss 在调用forward才会实例化

"""
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a*b, c*d)
    # [ab, cd]
    G = torch.mm(features, features.t())
    # [ab, cd] * [cd, ab]
    return G.div(a * b * c * d)
    # [ab, cd] * [cd, ab] / abcd


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


