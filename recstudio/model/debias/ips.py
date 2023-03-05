import torch
from recstudio.model import basemodel
from recstudio.model.basemodel import DebiasedRetriever
from recstudio.model.module.propensity import Popularity

r"""
IPS
######

Paper Reference:
    Recommendations as treatments: debiasing learning and evaluation (ICML'16)
    https://dl.acm.org/doi/10.5555/3045390.3045567
"""

class IPS(DebiasedRetriever):

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
        self.propensity = Popularity()
        self.propensity.fit(train_data)
    
    def _get_final_loss(self, loss : dict, output : dict, batch : dict):
        unreweighted_loss = loss['IPS']
        weight = 1 / (self.propensity(batch[self.fiid]) + 1e-7)
        reweighted_loss = torch.mean(weight * unreweighted_loss)
        return reweighted_loss