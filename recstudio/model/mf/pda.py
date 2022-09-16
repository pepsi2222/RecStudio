import torch
import torch.nn.functional as F
from recstudio.model.mf.bpr import BPR
from recstudio.model import basemodel, scorer

r"""
PDA
#########

Paper Reference:
    Causal Intervention for Leveraging Popularity Bias in Recommendation (SIGIR'21)
    https://doi.org/10.1145/3404835.3462875
"""

class PDA(BPR):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('PDA')
        parent_parser.add_argument("--algo", type=str, default='PDG', help='algo of PDA')
        parent_parser.add_argument("--gamma", type=float, default=0.02, help='gamma for PDA')
        return parent_parser        

    def _get_item_encoder(self, train_data):
        if self.config['algo'].upper()  == 'PDG' or 'PDGA':
            pop = (train_data.item_freq + 1) / (torch.sum(train_data.item_freq) + train_data.num_items)
            pop = (pop - pop.min()) / (pop.max() - pop.min())
            pop = pop.unsqueeze(-1)
        elif self.config['algo'].upper()  == 'PD' or 'PDA':
            NotImplementedError(f"{self.config['algo'] } is not implemented.")

        class PDAItemEncoder(torch.nn.Embedding):
            def __init__(self, pop, num_embeddings, embedding_dim, padding_idx=None):
                super().__init__(num_embeddings, embedding_dim, padding_idx)
                self.register_buffer('pop', pop)
            def forward(self, batch):
                items = super().forward(batch)
                pop = self.pop[batch]
                return torch.cat((items, pop), dim = -1)

        return PDAItemEncoder(pop, train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_item_vector(self):
        return torch.hstack((self.item_encoder.weight[1:], self.item_encoder.pop[1:]))

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

    def training_step(self, batch):
        output = self.forward(batch, False, return_item=True, return_neg_item=True)
        score = output['score']
        if self.config['algo'] == 'PD' or 'PDG':
            score['pos_score'] = output['item'].split([output['item'].shape[-1]-1, 1], dim=-1)[1] ** self.config['gamma'] * \
                                    score['pos_score']
            score['neg_score'] = output['neg_item'].split([output['neg_item'].shape[-1]-1, 1], dim=-1)[1] ** self.config['gamma'] * \
                                    score['neg_score']
        score['label'] = batch[self.frating]
        loss_value = self.loss_fn(**score)
        return loss_value