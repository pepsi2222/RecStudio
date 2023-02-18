import torch
from recstudio.model import basemodel, loss_func
from recstudio.model.mf import BPR
import torch.nn.functional as F

r"""
UBPR
######

Paper Reference:
    Unbiased Pairwise Learning from Biased Implicit Feedback (ICTIR'20)
    https://doi.org/10.1145/3409256.3409812
"""

class UBPR(BPR):
    '''
    Code as a retriever and the result is the same with BPR.
    It may be coded as a ranker so as to correctly treat some samples, which is positive, with BPRLoss.
    '''

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('UBPR')
        parent_parser.add_argument("--eta", type=float, default=0.5, help='adjust propensities')
        return parent_parser
    
    def _init_model(self, train_data):
        super()._init_model(train_data)
        # self.user_item_matrix = train_data.get_graph(0, form='csr')[0]  
        pop = (train_data.item_freq / torch.max(train_data.item_freq)) ** self.config['eta']
        self.register_buffer('pop', pop.unsqueeze(-1))
        self.rating_threshold = train_data.config.get('ranker_rating_threshold', 0)
    
    def _get_loss_func(self):
        class UBPRLoss(torch.nn.Module):
            def forward(self, weight, pos_score, log_pos_prob, neg_score, log_neg_prob):
                loss = F.logsigmoid(pos_score.view(*pos_score.shape, 1) - neg_score)   
                return -torch.mean((loss * F.softmax(weight.expand_as(neg_score), dim=-1)).sum(-1))
                ###
        return UBPRLoss()
    
    def training_step(self, batch):
        output = self.forward(batch, False, return_neg_id=True)
        score = output['score']
        
        pop = self.pop[batch[self.fiid]]
        weight = 1 / (pop + 1e-7)
        # pop_i = self.pop[batch[self.fiid]]
        # pop_j = self.pop[output['neg_id']]
        # label_j = self.user_item_matrix[]
        # weight = 1 / (pop + 1e-7) * (1 - label_j / pop_j)
        
        loss_value = self.loss_fn(weight, **score)
        return loss_value     