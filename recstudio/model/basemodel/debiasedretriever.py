from . import BaseRetriever
from .. import loss_func, scorer
from loss_func import FullScoreLoss
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from recstudio.utils import get_model
from types import MethodType

class DebiasedRetriever(BaseRetriever):
    def __init__(self, config: Dict = None, **kwargs):
        super().__init__(self, config, kwargs)
        self.backbone = torch.nn.ModuleList()

        if 'propensity' in kwargs:
            assert isinstance(kwargs['propensity'], torch.nn.Module), \
                "propensity must be torch.nn.Module"
            self.propensity = kwargs['propensity']
        else:
            self.propensity = None

        if 'discrepancy' in kwargs:
            assert isinstance(kwargs['discrepancy_criterion'], torch.nn.Module), \
                "discrepancy_criterion must be torch.nn.Module"
            self.discrepancy = kwargs['discrepancy_criterion']
        else:
            self.discrepancy = self._get_discrepancy()

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.propensity = self._get_propensity(train_data) if not self.propensity else self.propensity
        self._add_backbone(train_data)

    def _get_query_encoder(self, train_data):
        return None

    def _get_item_encoder(self, train_data):
        return None

    def _get_sampler(self, train_data):
        return None

    # def _get_score_func(self):
    #     # For only topk() function
    #     return scorer.InnerProductScorer()

    def _get_propensity(self, train_data):
        return None

    def _get_discrepancy(self):
        if 'discrepancy' not in self.config.keys():
            return None
        elif self.config['discrepancy'].lower() == 'l1':
            return loss_func.L1Loss()
        elif self.config['discrepancy'].lower() == 'l2':
            # Whether l2 normalization matters?
            return loss_func.SquareLoss()
        elif self.config['discrepancy'].lower() == 'dcor':
            return loss_func.dCorLoss()
        elif self.config['discrepancy'].lower() == 'cos':
            return scorer.CosineScorer()
        else:
            raise ValueError(f"{self.config['discrepancy']} "
                            "is unsupportable.")

    def _add_backbone(self, train_data):
        for name in self.config['backbone'].keys():
            if name in self.backbone.keys():
                raise ValueError(f'Backbone name {name} appears more than one time.')
            # TODO: support BaseRanker type models
            model_class, model_conf = get_model(self.config['backbone'][name]['model'])
            # TODO: another way to pass model config / parameters
            backbone = model_class(model_conf)
            backbone._init_model(train_data)
            self.backbone[name] = backbone

    def _get_masked_batch(self, backbone_name, batch):
        return batch

    def forward(self, batch, 
                return_query=False, return_item=False, 
                return_neg_item=False, return_neg_id=False):
        query, neg_item_idx, log_pos_prob, log_neg_prob = None, None, None, None
        if self.config['co_sampling']:
            if self.sampler is not None:
                if self.neg_count is None:
                    raise ValueError("`negative_count` value is required when "
                                    "`sampler` is not none.")
                (log_pos_prob, neg_item_idx, log_neg_prob), query = self.sampling(batch=batch, num_neg=self.neg_count,
                                                                                excluding_hist=self.config.get('excluding_hist', False),
                                                                                method=self.config.get('sampling_method', 'none'), return_query=True)
        output = {}
        for name, backbone in self.backbone.items():
            batch = self._get_masked_batch(name, batch)
            output[name] = self.backbone.forward(
                batch, 
                isinstance(backbone.loss_fn, FullScoreLoss),
                query=query, 
                neg_item_idx=neg_item_idx,
                log_pos_prob=log_pos_prob,
                log_neg_prob=log_neg_prob,
                return_query=True, 
                retuen_item=True,
                return_neg_item=True,
                return_neg_id=True)
        return output

    def training_step(self, batch):
        if self.propensity is not None:
            propensity = self.propensity(batch)
        else:
            propensity = None

        output = self.forward(batch)
        loss = {}
        for name, backbone in self.backbone.items():
            score = output[name]['score']
            score['label'] = batch[self.frating]
            if backbone.loss_fn is not None:
                loss[name] = backbone.loss_fn(
                    reduction=self.config['backbone'][name]['loss_reduction'],
                    **score)
        loss_value = self._get_final_loss(propensity, loss, output)
        return loss_value

    def _get_final_loss(self, propensity : Tensor, loss : dict, output : dict):
        pass
    
    def _get_item_vector(self):
        item_vector = {}
        for name, backbone in self.backbone.items():
            item_vector[name] = backbone._get_item_vector()
        item_vector = self._concat_item_vector(item_vector)
        return item_vector
    
    def _concat_item_vector(item_vector : dict):
        return torch.hstack([v for _, v in item_vector.items()])

    def _concat_query_vector(query_vector : dict):
        return torch.hstack([v for _, v in query_vector.items()])
    
    def topk(self, batch, k, user_h=None, return_query=False):
        query = {}
        for name, backbone in self.backbone.items():
            query[name] = backbone.query_encoder(backbone._get_query_feat(batch))
        query = self._concat_query_vector(query)
        return super().topk(batch, k, user_h, return_query, query)