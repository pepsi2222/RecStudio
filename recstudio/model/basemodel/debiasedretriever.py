from . import BaseRetriever
from .. import loss_func, scorer
from loss_func import FullScoreLoss
from recstudio.utils import get_model
import torch
from typing import Dict, List, Optional, Tuple, Union

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
        self._add_backbone(train_data)
        super()._init_model(train_data)
        self.propensity = self._get_propensity(train_data) if not self.propensity else self.propensity
    
    def _get_sampler(self, train_data):
        return None

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
        output = self.forward(batch)
        loss = {}
        for name, backbone in self.backbone.items():
            score = output[name]['score']
            score['label'] = batch[self.frating]
            if backbone.loss_fn is not None:
                loss[name] = backbone.loss_fn(
                    reduction=self.config['backbone'][name]['loss_reduction'],
                    **score)
        loss_value = self._get_final_loss(loss, output, batch)
        return loss_value

    def _get_final_loss(self, loss : Dict, output : Dict, batch : Dict):
        pass
    
    """Below is all for evaluation."""
    def _get_query_encoder(self, train_data):
        return DebiasedQueryEncoder(self.backbone)

    def _get_item_encoder(self, train_data):
        return DebiasedItemEncoder(self.backbone)
    
    def _get_query_feat(self, data):
        query_feat = {}
        for k, v in self.backbone.items():
            query_feat[k] = v._get_query_feat(data)
        return query_feat 
      
    def _get_item_feat(self, data):
        item_feat = {}
        for k, v in self.backbone.items():
            item_feat[k] = v._get_item_feat(data)
        return item_feat
    
    def _get_item_vector(self):
        item_vector = {}
        for name, backbone in self.backbone.items():
            item_vector[name] = backbone._get_item_vector()
        item_vector = torch.hstack([v for _, v in item_vector.items()])
        return item_vector
    
class DebiasedQueryEncoder(torch.nn.Module):
    def __init__(self, backbone, 
                 func=lambda d: torch.hstack([v for _, v in d.items()])):
        """func(function): decide how to get the query vector"""
        super().__init__()
        self.func = func
        self.query_encoders = {}
        for k, v in backbone.items():
            self.query_encoders[k] = v.query_encoder
    def forward(self, input):
        """input (dict): {backbone name: corresponding query feat}"""
        query = {}
        for k, v in self.query_encoders.items():
            query[k] = v(input[k])
        query = self.func(query)
        return query
    
class DebiasedItemEncoder(torch.nn.Module):
    def __init__(self, backbone,
                 func=lambda d: torch.hstack([v for _, v in d.items()])):
        """choice(str): one of backbone names or `all`"""
        super().__init__()
        self.func = func
        self.item_encoders = {}
        for k, v in backbone.items():
            self.item_encoders[k] = v.item_encoder
    def forward(self, input):
        """input (dict): {backbone name: corresponding item feat}"""
        item = {}
        for k, v in self.item_encoders.items():
            item[k] = v(input[k])
        item = self.func(item)
        return item