import torch
from recstudio.model import basemodel
from recstudio.model.basemodel import DebiasedRetriever
from recstudio.model.module.propensity import Popularity

r"""
UBPR
######

Paper Reference:
    Unbiased Pairwise Learning from Biased Implicit Feedback (ICTIR'20)
    https://doi.org/10.1145/3409256.3409812
"""

class UBPR(DebiasedRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('UBPR')
        parent_parser.add_argument("--eta", type=float, default=0.5, help='adjust propensities')
        return parent_parser

    def _get_propensity(self, train_data):
        self.propensity = Popularity(self.config['eta'])
        self.propensity.fit(train_data)
    
    def _get_final_loss(self, loss: dict, output: dict, batch: dict):
        unreweighted_loss = loss['UBPR']
        weight_i = 1 / (self.propensity(batch[self.fiid]) + 1e-7)
        weight_j = 1
        reweighted_loss = torch.mean(weight_i * weight_j * unreweighted_loss)
        if self.config['in_batch_sampling_count'] is None:
            return reweighted_loss
        else:
            bs = batch['frating'].shape[0]
            assert self.config['in_batch_sampling_count'] < bs
            in_batch_sample_item_idx = torch.multinomial(
                torch.ones((bs, bs), device=self.device) - \
                    torch.eye(bs, device=self.device),
                self.config['in_batch_sampling_count'])
            
            in_weight_j = 1 - 1 / (self.propensity(
                                    batch[self.fiid][in_batch_sample_item_idx]
                                    ) + 1e-7)
            score = output['UBPR']['score']
            score['neg_score'] = score['pos_score'][in_batch_sample_item_idx]
            score['label'] = batch[self.frating]
            in_batch_loss = torch.mean(weight_i * in_weight_j * \
                            self.backbone['UBPR'].loss_fn(**score))
            return reweighted_loss + in_batch_loss