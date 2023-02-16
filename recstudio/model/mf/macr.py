import torch
from recstudio.model.mf.bpr import BPR
from recstudio.model import basemodel, scorer

r"""
MACR
######

Paper Reference:
    Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System (KDD'21)
    https://doi.org/10.1145/3447548.3467289
"""

class MACR(BPR):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('MACR')
        # parent_parser.add_argument("--backbone", type=str, default='BPR', help='backbone recommender Y_k')
        # parent_parser.add_argument("--user_module", type=str, default='PMF', help='user module Y_u')
        # parent_parser.add_argument("--item_module", type=str, default='PMF', help='item module Y_i')
        parent_parser.add_argument("--c", type=float, default=40.0, help='reference status')
        parent_parser.add_argument("--alpha", type=float, default=1e-3, help='weight of user loss')
        parent_parser.add_argument("--beta", type=float, default=1e-3, help='weight of item loss')
        return parent_parser        
    
    def _get_score_func(self):
        class MACRScorer(scorer.InnerProductScorer):
            def __init__(self, c, embed_dim):
                super().__init__()
                self.c = c
                self.eval = False
                self.user_module = torch.nn.Parameter(torch.randn(embed_dim, 1))
                self.item_module = torch.nn.Parameter(torch.randn(embed_dim, 1))
            def forward(self, query, items):
                yk = super().forward(query, items)
                yu = torch.sigmoid(query @ self.user_module)
                yi = torch.sigmoid(items @ self.item_module)
                yk_ = yk if not self.eval else yk - self.c
                if query.size(0) == items.size(0):
                    if query.dim() < items.dim():
                        yui = torch.sigmoid(yk_ * yu * yi.squeeze(-1))
                    else:
                        yui = torch.sigmoid(yk_ * yu.squeeze(-1) * yi.squeeze(-1))
                else:
                    yui = yk_ * (yu @ yi.transpose(0, 1))
                    
                if self.eval == False:
                    return yu, yi, yui
                else:
                    return yui
        return MACRScorer(self.config['c'], self.embed_dim)
    
    def _get_loss_func(self):
        class MACRLoss(torch.nn.BCELoss):
            def __init__(self, alpha, beta):
                super().__init__(reduction='mean')
                self.alpha = alpha
                self.beta = beta
            def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
                pos_yu, pos_yi, pos_yui = pos_score
                neg_yu, neg_yi, neg_yui = neg_score
                loss_o = super().forward(pos_yui, torch.ones_like(pos_yui)) + super().forward(neg_yui, -torch.ones_like(neg_yui))
                loss_u = super().forward(pos_yu, torch.ones_like(pos_yu)) + super().forward(neg_yu, -torch.ones_like(neg_yu))
                loss_i = super().forward(pos_yi, torch.ones_like(pos_yi)) + super().forward(neg_yi, -torch.ones_like(neg_yi))
                return loss_o + self.alpha * loss_u + self.beta * loss_i
        return MACRLoss(self.config['alpha'], self.config['beta'])
    
    def training_step(self, batch):
        self.score_func.eval = False
        return super().training_step(batch)
    
    def _test_step(self, batch, metric, cutoffs):
        self.score_func.eval = True
        return super()._test_step(batch, metric, cutoffs)