import torch.nn.functional as F
from recstudio.model import basemodel, scorer, loss_func
from recstudio.model.basemodel import DebiasedRetriever
from recstudio.model.module.propensity import Popularity

r"""
PDA
#########

Paper Reference:
    Causal Intervention for Leveraging Popularity Bias in Recommendation (SIGIR'21)
    https://doi.org/10.1145/3404835.3462875
"""

class PDA(DebiasedRetriever):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('PDA')
        parent_parser.add_argument("--algo", type=str, default='PD', help='algo of PDA')
        parent_parser.add_argument("--popularity", type=str, default='global', help='global or local')
        parent_parser.add_argument("--gamma", type=float, default=0.02, help='gamma for PDA')
        return parent_parser        

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.backbone['PDA'].loss_fn = None

    def _get_propensity(self, train_data):
        if self.config['popularity'].lower() == 'global':
            self.propensity = Popularity()
            self.propensity.fit(train_data)
        elif self.config['popularity'].lower() == 'local':
            raise ValueError(f"Local popularity is not implemented.")

    def _get_score_func(self):  
        class PDAEvalScorer(scorer.InnerProductScorer):
            def __init__(self, algo, gamma, pop):
                super().__init__()
                self.algo = algo
                self.gamma = gamma
                self.pop = pop
            def forward(self, query, items):
                f = super().forward(query, items)
                elu_ = F.elu(f) + 1
                if self.algo == 'PD':
                    return elu_
                elif self.algo == 'PDA':
                    return self.pop ** self.gamma * elu_

        return PDAEvalScorer(self.config['algo'], 
                             self.config['gamma'],
                             self.propensity.pop) 

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

    def _get_loss_func(self):
        return loss_func.BPRLoss()
    
    def _get_final_loss(self, loss: dict, output: dict, batch : dict):
        pos_weight = self.propensity(batch[self.fiid]) ** self.config['gamma']
        neg_weight = self.propensity(output['PDA']['neg_id']) ** self.config['gamma']
        score = output['PDA']['score']
        score['pos_score'] = pos_weight * score['pos_score']
        score['neg_score'] = neg_weight * score['neg_score']
        score['label'] = batch[self.frating]
        loss = self.loss_fn(**score)
        return loss