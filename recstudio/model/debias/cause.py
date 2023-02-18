import torch
from recstudio.model.mf.pmf import PMF
from recstudio.model import basemodel, loss_func, scorer

r"""
CausE
#########

Paper Reference:
    Causal Embeddings for Recommendation (RecSys'18)
    https://dl.acm.org/doi/10.1145/3240323.3240360
"""

class CausE(PMF):               
        
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('CausE')
        parent_parser.add_argument("--method", type=str, default='prod-c', help='setups of incorporating the exploration data')
        parent_parser.add_argument("--dis_pen", type=float, default=1.0, help='discrepency penalty')
        parent_parser.add_argument("--dis_loss", type=str, default='l1', help='discrepency between the two product representations')
        parent_parser.add_argument("--split_mode", type=str, default='skew', help='setups of splitting dataset ')
        return parent_parser             

    def _get_item_encoder(self, train_data):
        if self.config['method'].lower() == 'avg':
            class CausEItemEncoder(torch.nn.Embedding):
                def forward(self, batch):
                    c_items_emb = super().forward(batch)
                    t_items_emb = self.weight[-1].expand_as(c_items_emb)
                    return torch.cat((c_items_emb, t_items_emb), dim = -1)
            return CausEItemEncoder(train_data.num_items + 1, self.embed_dim, padding_idx=0)
        elif self.config['method'].lower() == 'prod-c' or 'prod-t':
            return torch.nn.Embedding(train_data.num_items, 2 * self.embed_dim, padding_idx=0)
        else:
            ValueError(f"{self.config['method']} is invalid.")

    def _get_item_vector(self):
        if self.config['method'].lower() == 'avg':
            c_items_emb = self.item_encoder.weight[1:-1]
            return torch.hstack((c_items_emb, self.item_encoder.weight[-1:].expand_as(c_items_emb)))
        elif self.config['method'].lower() == 'prod-c' or 'prod-t':
            return self.item_encoder.weight[1:]
    
    def _get_score_func(self):
        class CausEScorer(scorer.InnerProductScorer):
            def __init__(self, method):
                super().__init__()
                self.eval = False
                self.method = method.lower()
            def forward(self, query, items):
                c_items, t_items = items.chunk(2, -1)
                if self.eval == False:
                    return super().forward(query, c_items), super().forward(query, t_items)
                elif self.method == 'avg' or 'prod-c':
                    return super().forward(query, c_items)
                elif self.method == 'prod-t':
                    return super().forward(query, t_items)
        return CausEScorer(self.config['method'])

    def _get_loss_func(self):
        class CausELoss(torch.nn.BCEWithLogitsLoss):
            def __init__(self, dis_pen, dis_criterion):
                super().__init__(reduction='mean')
                self.dis_pen = dis_pen
                self.dis_criterion = dis_criterion
            def forward(self, items, label, pos_score, treatment): 
                pos_score = (torch.vstack(pos_score) * torch.vstack((1-treatment, treatment))).sum(0)
                pred_loss = super().forward(pos_score, label)
                c_items, t_items = items.chunk(2, -1)
                dis_loss = self.dis_pen * self.dis_criterion(c_items, t_items.detach())
                return pred_loss + self.dis_pen * dis_loss
                    
        return CausELoss(self.config['dis_pen'], self._get_discrepancy_criterion())        

    def _get_discrepancy_criterion(self):
        # Whether l2 normalization matters?
        if self.config['dis_loss'].lower() == 'l1':
            return loss_func.L1Loss()
        elif self.config['dis_loss'].lower() == 'l2':
            return loss_func.SquareLoss()
        elif self.config['dis_loss'].lower() == 'cos':
            return scorer.CosineScorer()
        else:
            NotImplementedError(f"{self.config['dis_loss']} is unsupportable.")

    def training_step(self, batch):
        self.score_func.eval = False
        output = self.forward(batch, False, return_item=True)
        score, items = output['score'], output['item']
        score['label'] = batch[self.frating]
        score['treatment'] = batch['Loader']
        loss_value = self.loss_fn(items, **score)
        return loss_value

    def _test_step(self, batch, metric, cutoffs):
        self.score_func.eval = True
        return super()._test_step(batch, metric, cutoffs)

    def _get_sampler(self, train_data):
        return None
        
    def _get_train_loaders(self, train_data):
        #TODO: get two loaders from Sc and St.
        pass

    def current_epoch_trainloaders(self, nepoch):
        combine = False
        concat = True
        return self.trainloaders, combine, concat