import torch
from recstudio.model import basemodel, scorer
from recstudio.model.module import propensity
from recstudio.data.advance_dataset import ALSDataset

r"""
ExpoMF
#########

Paper Reference:
    Causal Inference for Recommendation
    http://www.its.caltech.edu/~fehardt/UAI2016WS/papers/Liang.pdf
"""

class IPW(basemodel.BaseRetriever):
    
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('IPW')
        parent_parser.add_argument("--lambda_theta", type=float, default=1e-5, help='lambda_theta for IPW')
        parent_parser.add_argument("--lambda_beta", type=float, default=1e-5, help='lambda_beta for IPW')
        parent_parser.add_argument("--init_range", type=float, default=0.01, help='init std for PMF')
        parent_parser.add_argument("--propensity_estimation", type=str, default='popularity', help='estimation for propensities')
        parent_parser.add_argument("--prediction", type=str, default='conditional', help='prediction method')
        return parent_parser                         
        
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.user_item_matrix = train_data.get_graph(0, form='csr')[0]  
    
    def _init_parameter(self):
        super()._init_parameter()
        self.query_encoder.weight.requires_grad = False
        self.item_encoder.weight.requires_grad = False

    @staticmethod
    def _get_dataset_class():
        return ALSDataset

    def _get_query_encoder(self, train_data):
        if self.config['prediction'].lower() == 'conditional':
            return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        elif self.config['prediction'].lower() == 'marginal':
            class IPWQueryEncoder(torch.nn.Embedding):
                def forward(self, batch):
                    query = super().forward(batch)
                    return query, batch
            return IPWQueryEncoder(train_data.num_users, self.embed_dim, padding_idx=0)
        else:
            raise ValueError(f"{self.config['prediction']} is illegal.")

    def _get_item_encoder(self, train_data):
        if self.config['prediction'].lower() == 'conditional':
            return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        elif self.config['prediction'].lower() == 'marginal':
            class IPWItemEncoder(torch.nn.Embedding):
                def forward(self, batch):
                    items = super().forward(batch)
                    return torch.cat((items, batch.unsqueeze(-1)), dim=-1)
            return IPWItemEncoder(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_item_vector(self):
        if self.config['prediction'].lower() == 'conditional':
            return self.item_encoder.weight[1:]
        elif self.config['prediction'].lower() == 'marginal':
            num_items = (self.item_encoder.num_embeddings).unsqueeze(-1)
            return torch.hstack((self.item_encoder.weight[1:], torch.arange(1, num_items)))

    def _get_score_func(self):       
        if self.config['prediction'].lower() == 'conditional':
            return scorer.InnerProductScorer()
        elif self.config['prediction'].lower() == 'marginal':
            class IPWScorer(scorer.InnerProductScorer):
                def __init__(self, propensity):
                    super().__init__()
                    self.eval = False
                    self.propensity = propensity
                def forward(self, query, items):
                    query, uid = query
                    items, iid = items.split([items.shape[-1]-1, 1], dim=-1)
                    if self.eval == False:
                        return super().forward(query, items)
                    else:
                        p = self.propensity(uid, iid)
                        return p * super().forward(query, items)
            return IPWScorer(self.p_model)
   
    def _get_train_loaders(self, train_data):
        loader = train_data.train_loader(
            batch_size = self.config['batch_size'],
            shuffle = True, 
            num_workers = self.config['num_workers'], 
            drop_last = False)
        loader_T = train_data.transpose().train_loader(
            batch_size = self.config['batch_size'],
            shuffle = True, 
            num_workers = self.config['num_workers'], 
            drop_last = False)
        return [loader, loader_T]     

    def current_epoch_trainloaders(self, nepoch):
        return self.trainloaders[nepoch % len(self.trainloaders)], False, False       

    def training_step(self, batch):
        """
        Update latent user/item factors
        """
        label = (batch[self.frating] > 0).float()
        if self.config['prediction'].lower() == 'marginal':
            self.score_func.eval = False
        query_emb = self.query_encoder(self._get_query_feat(batch))
        item_emb = self.item_encoder(self._get_item_feat(batch))
        p_bs = self.p_model(batch)
        weight_bs = 1 / (p_bs + 1e-7).detach()

        if batch[self.fuid].dim() == 1:
            batch.update({self.fiid: torch.arange(self.item_encoder.num_embeddings).expand(self.config['batch_size'], self.item_encoder.num_embeddings)})
            p = self.p_model(batch)
            weight = 1 / (p + 1e-7).detach()

            R = self.user_item_matrix            
            for i, uid in enumerate(batch[self.fuid]):
                y, iids = R.data[R.indptr[uid] : R.indptr[uid + 1]], R.indices[R.indptr[uid] : R.indptr[uid + 1]]
                y = self._to_device(torch.from_numpy(y), self.device)

                weight_o = weight[i][iids]
                A = (weight_o * self.item_encoder.weight[iids].transpose(0, 1)) @ \
                    self.item_encoder.weight[iids] + \
                    self.config['lambda_theta'] * torch.eye(self.embed_dim, device=self.device)     # D x D                              
                B = ((weight_o * y) @ self.item_encoder.weight[iids]).unsqueeze(-1)                # D x 1
                self.query_encoder.weight[uid] = torch.linalg.solve(A, B).squeeze(-1)

            pos_score = self.score_func(query_emb, item_emb)
            reg = self.config['lambda_theta'] * (query_emb**2).sum(-1) + \
                    self.config['lambda_beta'] * (item_emb**2).sum([-2, -1])
        else:
            R = self.user_item_matrix.T
            for i, iid in enumerate(batch[self.fiid]):
                batch.update({self.fuid: torch.arange(self.query_encoder.num_embeddings).expand(batch[self.fiid].shape[0], self.query_encoder.num_embeddings)})
                p = self.p_model(batch)
                weight = 1 / (p + 1e-7).detach()

                y, uids = R.data[R.indptr[iid] : R.indptr[iid + 1]], R.indices[R.indptr[iid] : R.indptr[iid + 1]]
                y = self._to_device(torch.from_numpy(y), self.device)

                weight_o = weight[i][uids]
                A = (weight_o * self.query_encoder.weight[uids].transpose(0, 1)) @ \
                    self.query_encoder.weight[uids] + \
                    self.config['lambda_beta'] * torch.eye(self.embed_dim, device=self.device)      # D x D
                B = ((weight_o * y) @ self.query_encoder.weight[uids]).unsqueeze(-1)               # D x 1
                self.item_encoder.weight[iid] = torch.linalg.solve(A, B).squeeze(-1)    
        
            pos_score = self.score_func(item_emb, query_emb)
            reg = self.config['lambda_theta'] * (query_emb**2).sum([-2, -1]) + \
                    self.config['lambda_beta'] * (item_emb**2).sum(-1)
        loss = (weight_bs * (label - pos_score)**2).sum(-1) + reg
        return {'loss': loss} 

    def _test_step(self, batch, metric, cutoffs):
        if self.config['prediction'].lower() == 'marginal':
            self.score_func.eval = True
        return super()._test_step(batch, metric, cutoffs)   

    def fit(self, train_data, val_data = None, run_mode='light', config = None, **kwargs):
        self.p_model = propensity.get_propensity(self.config) 
        self.p_model.fit(train_data)
        super().fit(train_data, val_data, run_mode, config, **kwargs)      