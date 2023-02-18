import torch
import numpy as np
from recstudio.ann.sampler import Sampler
from recstudio.model.mf.bpr import BPR
from recstudio.model import loss_func
from recstudio.model.basemodel import DebiasedRetriever

r"""
DICE
#########

Paper Reference:
    Disentangling User Interest and Conformity for Recommendation with Causal Embedding (WWW'21)
    https://doi.org/10.1145/3442381.3449788
"""

class DICE(DebiasedRetriever):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('DICE')
        parent_parser.add_argument("--discrepancy", type=str, default='l1', help='discrepency loss function')
        parent_parser.add_argument("--int_weight", type=float, default=0.1, help='weight for interest term in the loss function')
        parent_parser.add_argument("--pop_weight", type=float, default=0.1, help='weight for popularity term in the loss function')
        parent_parser.add_argument("--dis_penalty", type=float, default=0.01, help='discrepency penalty')
        parent_parser.add_argument("--margin_up", type=float, default=40.0, help='margin for negative but more popular sampling')
        parent_parser.add_argument("--margin_down", type=float, default=40.0, help='margin for negative and less popular sampling')
        parent_parser.add_argument("--pool", type=int, default=40, help='pool for negative sampling')
        parent_parser.add_argument("--adaptive", type=bool, default=True, help='adapt hyper-parameters or not')
        parent_parser.add_argument("--margin_decay", type=float, default=0.9, help='decay of margin')
        parent_parser.add_argument("--loss_decay", type=float, default=0.9, help='decay of loss')
        return parent_parser  
    
    def _get_loss_func(self):
        return loss_func.BPRLoss()    

    def _get_final_loss(propensity, loss : dict, output : dict):
        query_int = output['interest']['query']
        query_con = output['conformity']['query']
        pos_item_int = output['interest']['item']
        pos_item_con = output['conformity']['item']
        neg_item_int = output['interest']['neg_item']
        neg_item_con = output['conformity']['neg_item']
        item_int = torch.vstack((pos_item_int, 
                    neg_item_int.view(-1, pos_item_int.shape[1])))
        item_con = torch.vstack((pos_item_con, 
                    neg_item_con.view(-1, pos_item_con.shape[1])))
        loss_dis = self.discrepancy(query_int, query_con) + \
                            self.discrepancy(item_int, item_con)

        query = torch.vstack((query_int, query_con))
        pos_item = torch.vstack((pos_item_int, pos_item_con))
        neg_item = torch.vstack((neg_item_int, neg_item_con))
        pos_click_score = self.score_func(query, pos_item)
        neg_click_score = self.score_func(query, neg_item)
        loss_click = self.loss_fn(
            pos_score=pos_click_score, neg_score=neg_click_score, 
            label=None, log_pos_prob=None, log_neg_prob=None)

        loss_int = torch.mean(output['mask'] * loss['interest'])
        loss_con = torch.mean(~output['mask'] * loss['conformity']) + \
                torch.mean(output['mask'] * self.backbone['conformity'].loss_fn(
                    pos_score=output['conformity']['score']['neg_score'], 
                    neg_score=output['conformity']['score']['pos_score'],
                    label=None, log_pos_prob=None, log_neg_prob=None
                ))

        return self.int_weight * loss_int + self.pop_weight * loss_pop + \
                        loss_click - self.dis_pen * loss_dis
    
    def _adapt(self, current_epoch):
        if not hasattr(self, 'last_epoch'):
            self.last_epoch = 0
            self.int_weight = self.config['int_weight']
            self.con_weight = self.config['con_weight']
        if current_epoch > self.last_epoch:
            self.last_epoch = current_epoch
            self.int_weight = self.int_weight * self.config['loss_decay']
            self.pop_weight = self.pop_weight * self.config['loss_decay']
            self.sampler._adapt()

    def training_step(self, batch, nepoch):
        self._adapt(nepoch)
        return super().training_step(batch, nepoch)
        
    def _get_sampler(self, train_data):
        class PopularSamplerWithMargin(Sampler):
            def __init__(self, num_items, popularity, pool, margin_up, margin_down, margin_decay):
                super().__init__(num_items, scorer_fn=None)
                self.pop = popularity
                self.pool = pool
                self.margin_up = margin_up
                self.margin_down = margin_down
                self.margin_decay = margin_decay           
            def _adapt(self):
                self.margin_up = self.margin_decay * self.margin_up
                self.margin_down = self.margin_decay * self.margin_down
            def forward(self, query, num_neg, pos_items=None, user_hist=None):
                # user_hist(torch.Tensor): [num_user(or batch_size),max_hist_seq_len]
                # query: B x D, pos_item: B
                num_queries = np.prod(query.shape[:-1])
                shape = query.shape[:-1]
                device = device = user_hist.device
                with torch.no_grad():
                    neg_items = torch.full((num_queries, num_neg), -1, device=device)  # padding with zero
                    neg_items = neg_items.reshape(*shape, -1)  # B x L x Neg || B x Neg
                    mask = torch.full(neg_items.shape, False, device=device)

                    for u, hist in enumerate(user_hist):
                        pos_item = pos_items[u]

                        pop_items = torch.nonzero(self.pop > self.pop[pos_item] + self.margin_up).to(device)
                        pop_items = pop_items[torch.logical_not(torch.isin(pop_items, hist))]
                        num_pop_items = pop_items.shape[0]

                        unpop_items = torch.nonzero(self.pop < self.pop[pos_item] - self.margin_down).to(device)    
                        unpop_items = unpop_items[torch.logical_not(torch.isin(unpop_items, hist))]
                        num_unpop_items = unpop_items.shape[0]

                        if num_pop_items < self.pool:
                            idx = torch.randint(num_unpop_items, (num_neg,))
                            neg_items[u] = unpop_items[idx]
                            mask[u] = False
                        elif num_unpop_items < self.pool:
                            idx = torch.randint(num_pop_items, (num_neg,))
                            neg_items[u] = pop_items[idx]
                            mask[u] = True
                        else:
                            idx = torch.randint(num_pop_items, (num_neg//2,))
                            neg_items[u][:num_neg//2] = pop_items[idx]
                            mask[u][:num_neg//2] = True
                            
                            idx = torch.randint(num_unpop_items, (num_neg - num_neg//2,))
                            neg_items[u][num_neg//2:] = unpop_items[idx]
                            mask[u][num_neg//2:] = False

                    neg_prob = self.compute_item_p(None, neg_items)
                    if pos_items is not None:
                        pos_prob = self.compute_item_p(None, pos_items)
                        return pos_prob, neg_items, (mask, neg_prob)
                    else:
                        return neg_items, (mask, neg_prob)
                                                        
            def compute_item_p(self, query, pos_items):
                return torch.zeros_like(pos_items)

        return PopularSamplerWithMargin(train_data.num_items, train_data.item_freq, self.config['pool'],
                                        self.config['margin_up'], self.config['margin_down'], 
                                        self.config.get('margin_decay', None))
