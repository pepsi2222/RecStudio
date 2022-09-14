import torch
import numpy as np
from recstudio.ann.sampler import Sampler
from recstudio.model.mf.bpr import BPR
from recstudio.model import basemodel, loss_func
import time

class DICE(BPR):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('DICE')
        parent_parser.add_argument("--dis_loss", type=str, default='l1', help='discrepency loss function')
        parent_parser.add_argument("--int_weight", type=float, default=0.1, help='weight for interest term in the loss function')
        parent_parser.add_argument("--pop_weight", type=float, default=0.1, help='weight for popularity term in the loss function')
        parent_parser.add_argument("--dis_pen", type=float, default=0.01, help='discrepency penalty')
        parent_parser.add_argument("--margin_up", type=float, default=40.0, help='margin for negative but more popular sampling')
        parent_parser.add_argument("--margin_down", type=float, default=40.0, help='margin for negative and less popular sampling')
        parent_parser.add_argument("--pool", type=int, default=40, help='pool for negative sampling')
        parent_parser.add_argument("--adaptive", type=bool, default=True, help='adapt hyper-parameters or not')
        parent_parser.add_argument("--margin_decay", type=float, default=0.9, help='decay of margin')
        parent_parser.add_argument("--loss_decay", type=float, default=0.9, help='decay of loss')
        return parent_parser             

    def _get_query_encoder(self, train_data):
        int = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        pop = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        class DICEQueryEncoder(torch.nn.Module):
            def __init__(self, int, pop):
                super().__init__()
                self.int = int
                self.pop = pop
            def forward(self, batch):
                return torch.cat((self.int(batch), self.pop(batch)), dim=-1)
        return DICEQueryEncoder(int, pop)

    def _get_item_encoder(self, train_data):
        int = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        pop = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        class DICEItemEncoder(torch.nn.Module):
            def __init__(self, int, pop):
                super().__init__()
                self.int = int
                self.pop = pop
            def forward(self, batch):
                return torch.cat((self.int(batch), self.pop(batch)), dim=-1)
        return DICEItemEncoder(int, pop)

    def _get_item_vector(self):
        return torch.hstack((self.item_encoder.int.weight[1:], self.item_encoder.pop.weight[1:]))
    
    def _get_loss_func(self):
        class DICELoss(torch.nn.Module):
            def __init__(self, int_weight, pop_weight, loss_decay, dis_pen, dis_criterion):
                super().__init__()
                self.int_weight = int_weight
                self.pop_weight = pop_weight
                self.loss_decay = loss_decay
                self.dis_pen = dis_pen
                self.bprloss = loss_func.BPRLoss()
                self.maskbprloss = loss_func.MaskBPRLoss() 
                self.dis_criterion = dis_criterion
            def adapt(self):
                self.int_weight = self.int_weight * self.loss_decay
                self.pop_weight = self.pop_weight * self.loss_decay
            def forward(self, mask, pos_int_score, pos_pop_score, pos_click_score,
                                    neg_int_score, neg_pop_score, neg_click_score,
                                    query_int, query_pop, items_int, items_pop):
                loss_int = self.maskbprloss(mask=mask, pos_score=pos_int_score, neg_score=neg_int_score,
                                            label=None, log_pos_prob=None, log_neg_prob=None)                  
                loss_pop = self.maskbprloss(mask=mask, pos_score=neg_pop_score, neg_score=pos_pop_score,
                                            label=None, log_pos_prob=None, log_neg_prob=None) + \
                        self.maskbprloss(mask=~mask, pos_score=pos_pop_score, neg_score=neg_pop_score,
                                        label=None, log_pos_prob=None, log_neg_prob=None)
                loss_click = self.bprloss(pos_score=pos_click_score, neg_score=neg_click_score, 
                                        label=None, log_pos_prob=None, log_neg_prob=None)
                dis_loss = self.dis_criterion(query_int, query_pop) + self.dis_criterion(items_int, items_pop)
                return self.int_weight * loss_int + self.pop_weight * loss_pop + \
                        loss_click - self.dis_pen * dis_loss
        return DICELoss(self.config['int_weight'], self.config['pop_weight'], self.config['loss_decay'],
                        self.config['dis_pen'], self._get_discrepancy_criterion())        

    def _get_discrepancy_criterion(self):
        if self.config['dis_loss'].lower() == 'l1':
            return loss_func.L1Loss()
        elif self.config['dis_loss'].lower() == 'l2':
            return loss_func.SquareLoss()
        elif self.config['dis_loss'].lower() == 'dcor':
            return loss_func.dCorLoss()
    
    def adapt(self, current_epoch):
        if not hasattr(self, 'last_epoch'):
            self.last_epoch = 0
        if current_epoch > self.last_epoch:
            self.last_epoch = current_epoch
            self.loss_fn.adapt()
            self.sampler.adapt()

    def training_step(self, batch, nepoch):
        self.adapt(nepoch)
        output = self.forward(batch)
        mask, score, query, items = output['mask'], output['score'], output['query'], output['items']
        loss_value = self.loss_fn(mask, **score, **query, **items)
        return loss_value

    def forward(self, batch):
        output = {}
        pos_items = self._get_item_feat(batch)
        pos_item_vec = self.item_encoder(pos_items)
        if self.neg_count is None:
            raise ValueError("`negative_count` value is required when "
                             "`sampler` is not none.")

        (_, neg_item_idx, _), query = self.sampling(batch=batch, num_neg=self.neg_count,
                                                                            excluding_hist=self.config.get('excluding_hist', False),
                                                                            method=self.config.get('sampling_method', 'none'), return_query=True)
        query_int, query_pop = query.chunk(2, -1)
        neg_item_idx, mask = neg_item_idx
       
        pos_item_int, pos_item_pop = pos_item_vec.chunk(2, -1)
        pos_int_score = self.score_func(query_int, pos_item_int)
        pos_pop_score = self.score_func(query_pop, pos_item_pop)
        pos_click_score = self.score_func(query, pos_item_vec)

        if batch[self.fiid].dim() > 1:
            pos_int_score[batch[self.fiid] == 0] = -float('inf')
            pos_pop_score[batch[self.fiid] == 0] = -float('inf')
            pos_click_score[batch[self.fiid] == 0] = -float('inf')

        neg_items = self._get_item_feat(neg_item_idx)
        neg_item_vec = self.item_encoder(neg_items)

        neg_item_int, neg_item_pop = neg_item_vec.chunk(2, -1)
        neg_int_score = self.score_func(query_int, neg_item_int)
        neg_pop_score = self.score_func(query_pop, neg_item_pop)
        neg_click_score = self.score_func(query, neg_item_vec)

        output['mask'] = mask
        output['score'] = {'pos_int_score': pos_int_score, 'pos_pop_score': pos_pop_score, 'pos_click_score': pos_click_score,
                           'neg_int_score': neg_int_score, 'neg_pop_score': neg_pop_score, 'neg_click_score': neg_click_score}
        output['query'] = {'query_int': query.chunk(2, -1)[0], 'query_pop': query.chunk(2, -1)[1]}
        items = torch.vstack((pos_item_vec, neg_item_vec.view(-1, 2*self.embed_dim)))
        output['items'] = {'items_int': items.chunk(2, -1)[0], 'items_pop': items.chunk(2, -1)[1]}
        return output
        
    def _get_sampler(self, train_data):
        class PopularSamplerWithMargin(Sampler):
            def __init__(self, num_items, popularity, pool, margin_up, margin_down, margin_decay):
                super().__init__(num_items, scorer_fn=None)
                self.pop = popularity
                self.pool = pool
                self.margin_up = margin_up
                self.margin_down = margin_down
                self.margin_decay = margin_decay           
            def adapt(self):
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
                            for cnt in range(num_neg):
                                idx = torch.randint(num_unpop_items, (1,))
                                neg_items[u][cnt] = unpop_items[idx]
                                mask[u][cnt] = False
                        elif num_unpop_items < self.pool:
                            for cnt in range(num_neg):
                                idx = torch.randint(num_pop_items, (1,))
                                neg_items[u][cnt] = pop_items[idx]
                                mask[u][cnt] = True
                        else:
                            for cnt in range(num_neg):
                                if torch.rand(1) < 0.5:
                                    idx = torch.randint(num_pop_items, (1,))
                                    neg_items[u][cnt] = pop_items[idx]
                                    mask[u][cnt] = True
                                else:
                                    idx = torch.randint(num_unpop_items, (1,))
                                    neg_items[u][cnt] = unpop_items[idx]
                                    mask[u][cnt] = False

                    neg_prob = self.compute_item_p(None, neg_items)
                    if pos_items is not None:
                        pos_prob = self.compute_item_p(None, pos_items)
                        return pos_prob, (neg_items, mask), neg_prob
                    else:
                        return (neg_items, mask), neg_prob
                                                        
            def compute_item_p(self, query, pos_items):
                return torch.zeros_like(pos_items)

        return PopularSamplerWithMargin(train_data.num_items, train_data.item_freq, self.config['pool'],
                                        self.config['margin_up'], self.config['margin_down'], 
                                        self.config.get('margin_decay', None))
