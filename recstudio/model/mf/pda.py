import inspect
import torch
import torch.nn.functional as F
from recstudio.model.mf.bpr import BPR
from recstudio.model import basemodel, scorer, loss_func

class PDA(BPR):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('PDA')
        parent_parser.add_argument("--algo", type=str, default='PDG', help='algo of PDA')
        parent_parser.add_argument("--gamma", type=float, default=0.02, help='gamma for PDA')
        return parent_parser        

    def _get_item_encoder(self, train_data):
        item_emb = super()._get_item_encoder(train_data)
        if self.config['algo'].upper()  == 'PDG' or 'PDGA':
            pop = (train_data.item_freq + 1) / (torch.sum(train_data.item_freq) + train_data.num_items)
            pop = (pop - pop.min()) / (pop.max() - pop.min())
            pop = pop.unsqueeze(-1)
        elif self.config['algo'].upper()  == 'PD' or 'PDA':
            NotImplementedError(f"{self.config['algo'] } is not implemented.")

        class PDAItemEncoder(torch.nn.Module):
            def __init__(self, item_emb, pop):
                super().__init__()
                self.item_emb = item_emb
                self.register_buffer('pop', pop)
            def forward(self, batch):
                items = self.item_emb(batch)
                pop = self.pop[batch]
                return torch.cat((items, pop), dim=-1)

        return PDAItemEncoder(item_emb, pop)

    def _get_item_vector(self):
        return torch.hstack((self.item_encoder.item_emb.weight[1:], self.item_encoder.pop[1:]))

    def _get_score_func(self):  
        class PDAScorer(scorer.InnerProductScorer):
            def __init__(self, algo, gamma):
                super().__init__()
                self.algo = algo.upper()
                self.gamma = gamma
            def forward(self, query, items):
                items, pop = items.split([items.shape[-1]-1, 1], dim=-1)
                f = super().forward(query, items)
                elu_ = F.elu(f) + 1
                if self.algo == 'PD' or 'PDG':
                    return elu_
                elif self.algo == 'PDA' or 'PDGA':
                    return pop ** self.gamma * elu_

        return PDAScorer(self.config['algo'], self.config['gamma']) 
    
    def forward(self, batch, full_score, return_query=False, return_item=False, return_neg_item=False, return_neg_id=False):
        output = {}
        pos_items = self._get_item_feat(batch)
        pos_item_vec = self.item_encoder(pos_items)
        if self.sampler is not None:
            if self.neg_count is None:
                raise ValueError("`negative_count` value is required when "
                                 "`sampler` is not none.")

            (log_pos_prob, neg_item_idx, log_neg_prob), query = self.sampling(batch=batch, num_neg=self.neg_count,
                                                                              excluding_hist=self.config.get('excluding_hist', False),
                                                                              method=self.config.get('sampling_method', 'none'), return_query=True)
            pos_score = self.score_func(query, pos_item_vec)
            pos_score = pos_item_vec.split([pos_item_vec.shape[-1]-1, 1], dim=-1)[1] ** self.config['gamma'] * pos_score #
            if batch[self.fiid].dim() > 1:
                pos_score[batch[self.fiid] == 0] = -float('inf')

            neg_items = self._get_item_feat(neg_item_idx)
            neg_item_vec = self.item_encoder(neg_items)
            neg_score = self.score_func(query, neg_item_vec)
            neg_score = neg_item_vec.split([neg_item_vec.shape[-1]-1, 1], dim=-1)[1] ** self.config['gamma'] * neg_score #
            output['score'] = {'pos_score': pos_score, 'log_pos_prob': log_pos_prob,
                               'neg_score': neg_score, 'log_neg_prob': log_neg_prob}

            if return_neg_item:
                output['neg_item'] = neg_item_vec[0]                        #
            if return_neg_id:
                output['neg_id'] = neg_item_idx

            # data_augmentation
            if self.training and hasattr(self, 'data_augmentation'):
                data_augmentation_args = {"batch": batch}
                if 'query' in inspect.getargspec(self.data_augmentation).args:
                        data_augmentation_args['query'] = query
                output.update(self.data_augmentation(**data_augmentation_args))
        
        if return_query:
            output['query'] = query
        if return_item:
            output['item'] = pos_item_vec.split([pos_item_vec.shape[-1]-1, 1], dim=-1)[0]   #                           #
        return output