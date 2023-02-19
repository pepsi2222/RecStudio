import torch
from recstudio.ann import sampler
from recstudio.model import basemodel, scorer
from recstudio.model.module import MLPModule
from recstudio.model.basemodel import DebiasedRetriever

r"""
MACR
######

Paper Reference:
    Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System (KDD'21)
    https://doi.org/10.1145/3447548.3467289
"""

class MACR(DebiasedRetriever):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('MACR')
        parent_parser.add_argument("--c", type=float, default=40.0, help='reference status')
        parent_parser.add_argument("--alpha", type=float, default=1e-3, help='weight of user loss')
        parent_parser.add_argument("--beta", type=float, default=1e-3, help='weight of item loss')
        return parent_parser        
    
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.backbone['matching'].loss_fn = None

    def _get_score_func(self):
        # For only topk() function
        class MACRScorer(scorer.InnerProductScorer):
            def __init__(self, c, user_module, item_module):
                super().__init__()
                self.c = c
                self.user_module = user_module  # shared
                self.item_module = item_module  # shared
            def forward(self, query, items):
                yk = super().forward(query, items)
                yu = self.user_module(query)
                yi = self.item_module(items)
                yui = (yk - self.c) * torch.outer(yu, yi)
                return yui
        
        # Flexible but Dangerous, 
        # maybe can examine 
        # whether the str is `torch.[a-zA-z.]+([a-zA-z.,]+)` like, and
        # whether the outermost `(` and `)` match.
        # self.user_module = eval(self.config['user_module'])
        # self.item_module = eval(self.config['item_module'])
        assert self.config['user_module']['mlp_layers'][-1] == 1
        assert self.config['item_module']['mlp_layers'][-1] == 1
        self.user_module = MLPModule(**self.config['user_module'])
        self.item_module = MLPModule(**self.config['item_module'])
        assert 'sigmoid' in str(self.user_module[-1]).lower(), \
            'sigmoid' in str(self.item_module[-1]).lower()
        return MACRScorer(self.config['c'], self.user_module, self.item_module)
        
    
    def _get_loss_func(self):
        class BCELoss(torch.nn.Module):
            def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
                return -torch.mean(torch.log(pos_score) + torch.log(1 - neg_score)) 
        return BCELoss()
    
    def _get_final_loss(self, propensity, loss: dict, output: dict):
        pos_yui = output['matching']['score']['pos_score'] * \
                          output['user_module']['score']['pos_score'] * \
                          output['item_module']['score']['pos_score']
        neg_yui = output['matching']['score']['neg_score'] * \
                        torch.outer(
                            output['user_module']['score']['neg_score'], 
                            output['item_module']['score']['neg_score'])
        loss_click = self.loss_fn(
                        pos_score=pos_yui,
                        neg_score=neg_yui,
                        label=None,
                        log_pos_prob=None,
                        log_neg_prob=None)
        score_u = self.user_module(output['matching']['query'])
        loss_u = self.loss_fn(
                    pos_score=score_u,
                    neg_score=score_u,
                    label=None,
                    log_pos_prob=None,
                    log_neg_prob=None)
        loss_i = self.loss_fn(
                    pos_score=self.item_module(output['matching']['item']),
                    neg_score=self.item_module(output['matching']['neg_item']),
                    label=None,
                    log_pos_prob=None,
                    log_neg_prob=None)
        return loss_click + self.config['alpha'] * loss_u + self.config['beta'] * loss_i

    def _get_sampler(self, train_data):
        return sampler.MaskedUniformSampler(train_data.num_items)