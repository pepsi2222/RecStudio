from operator import neg
import torch
import torch.nn.functional as F


class FullScoreLoss(torch.nn.Module):
    r"""Calculate loss with positive scores and scores on all items.

    The loss need user's perference scores on positive items(ground truth) and all other items.
    However, due to the item numbers are very huge in real-world datasets, calculating scores on all items
    may be very time-consuming. So the loss is seldom used in large-scale dataset.
    """

    def forward(self, label, pos_score, all_score):
        r"""
        """
        pass


class PairwiseLoss(torch.nn.Module):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        pass


class PointwiseLoss(torch.nn.Module):
    def forward(self, label, pos_score):
        raise NotImplementedError(f'{type(self).__name__} is an abstrat class, \
            this method would not be implemented')

class L1Loss(PointwiseLoss):
    def forward(self, label, pos_score):
        if label.dim() > 1:
            return torch.mean(torch.mean(torch.absolute(label - pos_score), dim=-1))
        else:
            return torch.mean(torch.absolute(label - pos_score))

class SquareLoss(PointwiseLoss):
    def forward(self, label, pos_score):
        if label.dim() > 1:
            return torch.mean(torch.mean(torch.square(label - pos_score), dim=-1))
        else:
            return torch.mean(torch.square(label - pos_score))

# class AccuracyLoss(PointwiseLoss):
#     def forward(self, label, pos_score):
#         if label.dim() > 1:
#             return torch.mean(torch.mean(label == pos_score, dim=-1))
#         else:
#             return torch.mean(label == pos_score)

class dCorLoss(PointwiseLoss):
    def forward(self, label, pos_score):
        if label.dim() > 1:
            pass
        else:
            a = torch.norm(label[:,None] - label, p = 2, dim = 2)
            b = torch.norm(pos_score[:,None] - pos_score, p = 2, dim = 2)

            A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
            B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean() 

            n = label.size(0)

            dcov2_ab = (A * B).sum()/float(n * n)
            dcov2_aa = (A * A).sum()/float(n * n)
            dcov2_bb = (B * B).sum()/float(n * n)
            dcor = -torch.sqrt(dcov2_ab)/torch.sqrt(torch.sqrt(dcov2_aa) * torch.sqrt(dcov2_bb))
            return dcor

class SoftmaxLoss(FullScoreLoss):
    def forward(self, label, pos_score, all_score):
        if all_score.dim() > pos_score.dim():
            return torch.mean(torch.logsumexp(all_score, dim=-1) - pos_score)
        else:
            output = torch.logsumexp(all_score, dim=-1, keepdim=True) - pos_score
            notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
            output = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
            return torch.mean(output)


class BPRLoss(PairwiseLoss):
    def __init__(self, dns=False):
        super().__init__()
        self.dns = dns

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        if not self.dns:
            loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            return -torch.mean((loss * weight).sum(-1))
        else:
            loss = -torch.mean(
                F.logsigmoid(pos_score - torch.max(neg_score, dim=-1)))
            return loss


class Top1Loss(BPRLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        if not self.dns:
            loss = torch.sigmoid(neg_score - pos_score.view(*pos_score.shape, 1))
            loss += torch.sigmoid(neg_score ** 2)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            return torch.mean((loss * weight).sum(-1))
        else:
            max_neg_score = torch.max(neg_score, dim=-1)
            loss = torch.sigmoid(max_neg_score-pos_score)
            loss = loss + torch.sigmoid(max_neg_score ** 2)
        return loss


class SampledSoftmaxLoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        if new_pos.dim() < new_neg.dim():
            new_pos.unsqueeze_(-1)
        new_neg = torch.cat([new_pos, new_neg], dim=-1)
        output = torch.logsumexp(new_neg, dim=-1, keepdim=True) - new_pos
        notpadnum = torch.logical_not(torch.isinf(new_pos)).float().sum(-1)
        output = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
        return torch.mean(output)


class WeightedBPRLoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
        weight = F.softmax(neg_score - log_neg_prob, -1)
        return -torch.mean((loss * weight).sum(-1))


class BinaryCrossEntropyLoss(PairwiseLoss):
    def __init__(self, dns=False):
        super().__init__()
        self.dns = dns

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        # pos_score: B | B x L | B x L
        # neg_score: B x neg | B x L x neg | B x neg
        assert ((pos_score.dim() == neg_score.dim()-1) and (pos_score.shape ==
                neg_score.shape[:-1])) or (pos_score.dim() == neg_score.dim())
        if not self.dns:
            weight = self._cal_weight(neg_score, log_neg_prob)
            notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum()
            output = torch.nan_to_num(F.logsigmoid(pos_score), nan=0.0).sum() / notpadnum
            neg_score = F.softplus(neg_score) * weight
            neg_score_sum = neg_score.sum(-1)
            if pos_score.dim() == neg_score.dim()-1:
                padding_mask = torch.isinf(pos_score)
                neg_score_sum.masked_fill_(padding_mask, 0.0)
            return -output + torch.mean(neg_score_sum)
        else:
            return torch.mean(-F.logsigmoid(pos_score) + F.softplus(torch.max(neg_score, dim=-1)))

    def _cal_weight(self, neg_score, log_neg_prob):
        return torch.ones_like(neg_score) / neg_score.size(-1)


class WeightedBinaryCrossEntropyLoss(BinaryCrossEntropyLoss):
    def _cal_weight(self, neg_score, log_neg_prob):
        return F.softmax(neg_score - log_neg_prob, -1)


class HingeLoss(PairwiseLoss):
    def __init__(self, margin=2, num_items=None):
        super().__init__()
        self.margin = margin
        self.n_items = num_items

    def forward(self, label, pos_score, log_pos_prob, neg_score, neg_prob):
        loss = torch.maximum(torch.max(neg_score, dim=-1).values - pos_score +
                             self.margin, torch.tensor([0]).type_as(pos_score))
        if self.n_items is not None:
            impostors = neg_score - pos_score.view(-1, 1) + self.margin > 0
            rank = torch.mean(impostors, -1) * self.n_items
            return torch.mean(loss * torch.log(rank + 1))
        else:
            return torch.mean(loss)


class InfoNCELoss(SampledSoftmaxLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        return super().forward(label, pos_score, torch.zeros_like(pos_score),
                               neg_score, torch.zeros_like(neg_score))


class NCELoss(PairwiseLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        loss = F.logsigmoid(new_pos) + (new_neg - F.softplus(new_neg)).sum(1)
        return -loss.mean()


class CCLLoss(PairwiseLoss):
    def __init__(self, margin=0.8, neg_weight=0.3) -> None:
        super().__init__()
        self.margin = margin
        self.neg_weight = neg_weight

    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        # pos_score: [B,] or [B, N]
        # neg_score: [B, num_neg] or [B, N, num_neg]
        pos_score = torch.sigmoid(pos_score)
        neg_score = torch.sigmoid(neg_score)
        neg_score_mean = torch.mean(torch.relu(neg_score - self.margin), dim=-1)  # [B] or [B,N]
        notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum()
        loss = (1 - pos_score) + self.neg_weight * neg_score_mean
        loss = torch.nan_to_num(loss, posinf=0.0)
        return loss.sum() / notpadnum


def l2_reg_loss_fn(*args):
    loss = 0.
    for emb in args:
        loss = loss + torch.mean(torch.sum(emb * emb, dim=-1)) # [B, D] -> [B] -> []
    return loss 

class MaskBPRLoss(BPRLoss):   
    def forward(self, mask, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        if not self.dns:
            loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)
            weight = F.softmax(torch.ones_like(neg_score), -1)
            return -torch.mean(mask * (loss * weight).sum(-1))
        else:
            loss = -torch.mean(mask *
                F.logsigmoid(pos_score - torch.max(neg_score, dim=-1)))
            return loss