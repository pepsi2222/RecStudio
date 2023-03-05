import torch
from recstudio.model.basemodel import DebiasedRetriever
from recstudio.model import basemodel, scorer
import copy

r"""
CausE
#########

Paper Reference:
    Causal Embeddings for Recommendation (RecSys'18)
    https://dl.acm.org/doi/10.1145/3240323.3240360

    Here is the `prod-C` setting.
"""
class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, label, pos_score, log_pos_prob, neg_score, log_neg_prob):
        return super().forward(pos_score, label)

class CausE(DebiasedRetriever):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('CausE')
        parent_parser.add_argument("--dis_penalty", type=float, default=1.0, help='discrepancy penalty')
        parent_parser.add_argument("--discrepancy", type=str, default='l1', help='discrepancy between the two product representations')
        parent_parser.add_argument("--data_mode", type=str, default='reg', help='setups of splitting dataset ')
        return parent_parser     

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.backbone['control'].loss_fn = BCEWithLogitsLoss() 
        self.backbone['treatment'].loss_fn = BCEWithLogitsLoss()
        self.backbone['treatment'].query_encoder = self.backbone['control'].query_encoder

    def _concat_item_vector(item_vector : dict):
        return item_vector['control']

    def _concat_query_vector(query_vector : dict):
        return query_vector['control']
    
    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_final_loss(self, loss : dict, output : dict, batch : dict):
        item_c = output['control']['item']
        item_t = output['treatment']['control']
        loss_dis = self.discrepancy(item_c, item_t)
        return loss['control'] + loss['treatment'] + \
                self.config['dis_penalty'] * loss_dis
    
    def _get_masked_batch(self, backbone_name, batch):
        masked_batch = copy.deepcopy(batch)
        control = (masked_batch['Loader'] == 0) # larger dataset
        if backbone_name == 'control':
            for k, v in masked_batch.items():
                masked_batch[k] = v[control]
        elif backbone_name == 'treatment':
            for k, v in masked_batch.items():
                masked_batch[k] = v[~control]
        
    def _get_train_loaders(self, train_data, ddp=False):
        #TODO: get two datasets from Sc and St.
        train_data_c = train_data_t = None
        loader_c = train_data_c.train_loader(
                    batch_size = self.config['backbone']['control']['batch_size'],
                    shuffle = True,
                    num_workers = self.config['num_workers'],
                    drop_last = False, ddp=ddp)
        loader_t = train_data_t.train_loader(
                    batch_size = self.config['backbone']['treatment']['batch_size'],
                    shuffle = True,
                    num_workers = self.config['num_workers'],
                    drop_last = False, ddp=ddp)
        return [loader_c, loader_t]

    def current_epoch_trainloaders(self, nepoch):
        combine = False
        concat = True
        return self.trainloaders, combine, concat