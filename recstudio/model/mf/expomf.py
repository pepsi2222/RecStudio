import torch
import numpy as np
from recstudio.model import basemodel, scorer, loss_func
from recstudio.data.dataset import MFDataset
from recstudio.data.advance_dataset import ALSDataset

r"""
ExpoMF
#########

Paper Reference:
    Modeling User Exposure in Recommendation (WWW'16)
    https://dl.acm.org/doi/10.1145/2872427.2883090
"""

class ExpoMF(basemodel.BaseRetriever):
    
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('ExpoMF')
        parent_parser.add_argument("--lambda_y", type=float, default=1.0, help='lambda_y for ExpoMF')
        parent_parser.add_argument("--lambda_theta", type=float, default=1e-5, help='lambda_theta for ExpoMF')
        parent_parser.add_argument("--lambda_beta", type=float, default=1e-5, help='lambda_beta for ExpoMF')
        parent_parser.add_argument("--init_range", type=float, default=0.01, help='init std for PMF')
        parent_parser.add_argument("--init_mu", type=float, default=0.01, help='init mu for ExpoMF')
        parent_parser.add_argument("--with_exposure_covariates", type=bool, default=False, help='if there are exposure covariates or not')
        parent_parser.add_argument("--alpha1", type=float, default=1.0, help='alpha1 for ExpoMF')
        parent_parser.add_argument("--alpha2", type=float, default=1.0, help='alpha2 for ExpoMF')
        parent_parser.add_argument("--mu_init_range", type=float, default=0.01, help='init std for mu model in ExpoMF')
        return parent_parser                         
        
    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.user_item_matrix = train_data.get_graph(0, form='csr')[0]  
        self.register_buffer('a', torch.ones(train_data.num_users, train_data.num_items))
        
        if not self.config['with_exposure_covariates']:
            self.register_buffer('mu', self.config['init_mu'] * torch.ones(train_data.num_items))
        else:
            class mu_model(basemodel.BaseRetriever):

                @staticmethod
                def _get_dataset_class():
                    return MFDataset

                def _init_parameter(self):
                    super()._init_parameter()
                    # TODO: How to use init.py when different parts of a module have differnt 'init_method'?          
                    torch.nn.init.constant_(self.query_encoder.bias.weight, np.log(self.config['init_mu'] / (1 - self.config['init_mu'])))
                    self.item_encoder.weight.requires_grad = False
                
                def _get_item_encoder(self, train_data):
                    # the content of document i obtained through natural language processing 
                    # (e.g., word embeddings, latent Dirichlet allocation), 
                    # or the position of venue i obtained by 
                    # first clustering all the venues in the data set 
                    # then finding the expected assignment to L clusters for each venue.

                    # x_i
                    NotImplementedError('ExpoMF with exposure covariates should override this method.')                                              

                def _get_query_encoder(self, train_data):
                    # \psi, \psi_bias
                    # psi = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
                    # psi_bias = torch.nn.Embedding(train_data.num_users, 1, padding_idx=0)
                    # class muQueryEncoder(torch.nn.Module):
                    #     def __init__(self, psi, psi_bias):
                    #         super().__init__()
                    #         self.psi = psi
                    #         self.psi_bias = psi_bias
                    #     def forward(self, batch):
                    #         return self.psi(batch), self.psi_bias(batch)
                    # return muQueryEncoder(psi, psi_bias)
                    class muQueryEncoder(torch.nn.Embedding):
                        def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
                            super().__init__(num_embeddings, embedding_dim, padding_idx)
                            self.bias = torch.nn.Embedding(num_embeddings, 1, padding_idx)
                        def forward(self, batch):
                            return super().forward(batch), self.bias(batch)
                    return muQueryEncoder(train_data.num_users, self.embed_dim, padding_idx=0)

                def _get_score_func(self):
                    class muScorer(scorer.InnerProductScorer):
                        def forward(self, query, items):
                            query, query_bias = query
                            return torch.sigmoid(super().forward(query, items) + query_bias)
                    return muScorer
                   
                def _get_loss_func(self):
                    return loss_func.SquareLoss()

            mu_model_conf = self.config
            for k, v in mu_model_conf.items():
                if k.startswith('mu_model_'):
                    mu_model_conf.pop(k)
                    mu_model_conf.update({k[9:]: v})
            self.mu_model = mu_model(mu_model_conf)

            mu_dataset_class = self.mu_model._get_dataset_class()
            # take the last batch as validation
            self.mu_model.datasets = mu_dataset_class(name=train_data.name).build(
                split_ratio=self.mu_model.config['batch_size'], **self.mu_model.config) 
    
    def _init_parameter(self):
        super()._init_parameter()
        self.query_encoder.weight.requires_grad = False
        self.item_encoder.weight.requires_grad = False

    @staticmethod
    def _get_dataset_class():
        return ALSDataset

    def _get_query_encoder(self, train_data):
        if not self.config['with_exposure_covariates']:
            return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        else:
            class ExpoMFQueryEncoder(torch.nn.Embedding):
                def forward(self, batch):
                    query = super().forward(batch)
                    return query, batch
            return ExpoMFQueryEncoder(train_data.num_users, self.embed_dim, padding_idx=0)

    def _get_item_encoder(self, train_data):
        if not self.config['with_exposure_covariates']:
            return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        else:
            class ExpoMFItemEncoder(torch.nn.Embedding):
                def forward(self, batch):
                    items = super().forward(batch)
                    return torch.cat((items, batch.unsqueeze(-1)), dim=-1)
            return ExpoMFItemEncoder(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_item_vector(self):
        if not self.config['with_exposure_covariates']:
            return self.item_encoder.weight[1:]
        else:
            num_items = (self.item_encoder.weight.shape[0]).unsqueeze(-1)
            return torch.hstack((self.item_encoder.weight[1:], torch.arange(1, num_items)))

    def _get_score_func(self):       
        if not self.config['with_exposure_covariates']:
            return scorer.InnerProductScorer()
        else:
            class ExpoMFScorer(scorer.InnerProductScorer):
                def __init__(self, mu_model):
                    super().__init__()
                    self.mu_model = mu_model
                def forward(self, query, items):
                    query, uid = query
                    items, iid = items.split([items.shape[-1]-1, 1], dim=-1)
                    mu = self.mu_model.score_func(
                        self.mu_model.query_encoder(uid),
                        self.mu_model.item_encoder(iid))
                    return mu * super().forward(query, items)
            return ExpoMFScorer(self.mu_model)
        
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
        return self.trainloaders[nepoch % len(self.trainloaders)], False       

    def training_epoch(self, nepoch):
        super().training_epoch(nepoch)       
        if not self.config['with_exposure_covariates']:
            # update \mu_{i}; prior on \mu_{i} ~ Beta(alpha1, alpha2)
            self.mu = (self.config['alpha1'] + torch.sum(self.a, dim=0) - 1) / \
                    (self.config['alpha1'] + self.config['alpha2'] + self.a.shape[0] - 2)
        else:
            # update user exposure factors \psi and \psi_bias with mini-batch SGD      
            self.mu_model.fit(*self.mu_model.datasets[:2])                       
        return torch.tensor(0.)

    def training_step(self, batch):
        a = self._expectation(batch)
        self._maximization(batch, a)
    
    def _expectation(self, batch):
        """
        Compute the posterior of exposure latent variables a_{ui}
        """
        if self.config['with_exposure_covariates']:
            mu = self.mu_model.score_func(self.mu_model.query_encoder(self.mu_model._get_query_feat(batch)),
                                        self.mu_model.item_encoder(self.mu_model._get_item_feat(batch)))
        elif batch[self.fuid].dim() == 1:
            mu = self.mu
            P_y0_given_a1 = np.sqrt(self.config['lambda_y'] / 2 * torch.pi) * \
                            torch.exp(-self.config['lambda_y'] * 
                                (self.query_encoder(self._get_query_feat(batch)) @  # B x D
                                self.item_encoder.weight.transpose(0, 1))           # D x num_items
                                **2 / 2)                                            # -> B x num_items
        else: 
            mu = self.mu[batch[self.fiid]].unsqueeze(-1)
            P_y0_given_a1 = np.sqrt(self.config['lambda_y'] / 2 * torch.pi) * \
                            torch.exp(-self.config['lambda_y'] * 
                                (self.item_encoder(self._get_item_feat(batch)) @  # B x D
                                self.query_encoder.weight.transpose(0, 1))          # D x num_users
                                **2 / 2)                                            # -> B x num_users
                              
        a = (P_y0_given_a1 + 1e-8) / (P_y0_given_a1 + 1e-8 + (1 - mu) / mu)        
        for i, j in batch[self.frating].nonzero():
            a[i, j] = torch.tensor(1.)

        # update self.a
        if batch[self.fuid].dim() == 1:    
            for i, uid in enumerate(batch[self.fuid]):
                for j, iid  in enumerate(batch[self.fiid][i]):
                    self.a[uid, iid] = a[i, j]
        return a     
    
    def _maximization(self, batch, a):
        """
        Update latent factors theta and beta
        """
        if batch[self.fuid].dim() == 1:
            R = self.user_item_matrix
            for i, uid in enumerate(batch[self.fuid]):
                A = self.config['lambda_y'] * \
                    (a[i] * self.item_encoder.weight.transpose(0, 1)) @ \
                    self.item_encoder.weight + \
                    self.config['lambda_theta'] * torch.eye(self.embed_dim, device=self.device)     # D x D
                y, iids = R.data[R.indptr[uid] : R.indptr[uid + 1]], R.indices[R.indptr[uid] : R.indptr[uid + 1]]
                y = self._to_device(torch.from_numpy(y), self.device)
                B = self.config['lambda_y'] * ((a[i][iids] * y) @ \
                    self.item_encoder.weight[iids]).unsqueeze(-1)                                   # D x 1
                self.query_encoder.weight[uid] = torch.linalg.solve(A, B).squeeze(-1)
        else:
            R = self.user_item_matrix.T
            for i, iid in enumerate(batch[self.fiid]):
                A = self.config['lambda_y'] * \
                    (a[i] * self.query_encoder.weight.transpose(0, 1)) @ \
                    self.query_encoder.weight + \
                    self.config['lambda_beta'] * torch.eye(self.embed_dim, device=self.device)      # D x D
                y, uids = R.data[R.indptr[iid] : R.indptr[iid + 1]], R.indices[R.indptr[iid] : R.indptr[iid + 1]]
                y = self._to_device(torch.from_numpy(y), self.device)
                B = self.config['lambda_y'] * ((a[i][uids] * y) @ \
                    self.query_encoder.weight[uids]).unsqueeze(-1)                                  # D x 1
                self.item_encoder.weight[iid] = torch.linalg.solve(A, B).squeeze(-1)              