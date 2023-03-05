import torch
from recstudio.model import basemodel
from recstudio.model.basemodel import DebiasedRetriever
from recstudio.model.module.propensity import Popularity

r"""
RelMF
######

Paper Reference:
    Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback (WSDM'20)
    https://doi.org/10.1145/3336191.3371783
"""

class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        return super().forward(pos_score, label)

class RelMF(DebiasedRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('RelMF')
        parent_parser.add_argument("--eta", type=float, default=0.5, help='adjust propensities')
        return parent_parser
    
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.backbone['RelMF'].loss_fn = None
        # self.rating_threshold = train_data.config.get('ranker_rating_threshold', 0)

    def _get_propensity(self, train_data):
        self.propensity = Popularity(self.config['eta'])
        self.propensity.fit(train_data)

    def _get_loss_func(self):
        return BCEWithLogitsLoss()
    
    def _get_final_loss(self, loss : dict, output : dict, batch : dict):
        pop = self.propensity(batch[self.fiid])
        score = output['RelMF']['score']
        label = batch[self.frating]
        # label = (batch[self.frating] > self.rating_threshold).float()
        score['label'] = label / (pop + 1e-7) + (1 - label) * (1 - label / (pop + 1e-7))
        loss = self.loss_fn(**score)
        return loss