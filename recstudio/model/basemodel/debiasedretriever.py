from . import BaseRetriever
from .. import loss_func
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

    def _get_propensity(self, train_data)
        return None

    def _get_discrepancy(self):
        if 'discrepancy' not in self.config.keys():
            return None
        elif self.config['discrepancy'].lower() == 'l1':
            return loss_func.L1Loss()
        elif self.config['discrepancy'].lower() == 'l2':
            return loss_func.SquareLoss()
        elif self.config['discrepancy'].lower() == 'dcor':
            return loss_func.dCorLoss()

    def _add_backbone(self, train_data):
        for name in self.config['backbone']:
            if name in self.backbone.keys():
                raise ValueError(f'Backbone name {name} appears more than one time.')
            model_class, model_conf = get_model(self.config['backbone'][name])
            backbone = model_class(model_conf)
            backbone._init_model(train_data)
            if self.config['co_sampling']:
                # 如果多个backbone共同采样，一个step只采一次：
                # 不用backbone的samling方法，用debias框架自定义的_sample方法
                # 能不能把baseretriever的forward里面的sampling放到外面，
                # 然后把query等参数传给forward
                # 采样的时候 debias框架先采样，再分别传给不同的backbone
                backbone.forward = MethodType(..., backbone)
            self.backbone[name] = backbone

    def _sample(
        self,
        batch,
        neg: int = 1,
        excluding_hist: bool = False,
        return_query: bool = True
    ):


    def forward(self, batch, return_query=False, return_item=False, return_neg_item=False, return_neg_id=False):
        output = {}
        for name, backbone in self.backbone.items():
            output[name] = self.backbone.forward(
                batch, 
                isinstance(backbone.loss_fn, FullScoreLoss), 
                return_query, 
                retuen_item,
                return_neg_item,
                return_neg_id)
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
            loss[name] = backbone.loss_fn(
                reduction=self.config['name']['loss_reduction'],
                **score)
        loss_value = self._get_final_loss(propensity, loss, output)
        return loss_value

    def _get_final_loss(propensity : Tensor, loss : dict, output : dict):
        pass
    