import torch
import numpy as np
from recstudio.model import basemodel, scorer, loss_func
from recstudio.data.dataset import MFDataset
from recstudio.data.advance_dataset import ALSDataset


class ExpoMF(basemodel.BaseRetriever):
    
    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('ExpoMF')
        parent_parser.add_argument("--lambda_y", type=float, default=1.0, help='lambda_y for ExpoMF')
        parent_parser.add_argument("--lambda_theta", type=float, default=1e-5, help='lambda_theta for ExpoMF')
        parent_parser.add_argument("--lambda_beta", type=float, default=1e-5, help='lambda_beta for ExpoMF')
        parent_parser.add_argument("--query_std", type=float, default=0.01, help='query std for PMF')
        parent_parser.add_argument("--item_std", type=float, default=0.01, help='item std for PMF')
        parent_parser.add_argument("--init_mu", type=float, default=0.01, help='init mu for ExpoMF')
        parent_parser.add_argument("--with_exposure_covariates", type=bool, default=False, help='if there are exposure covariates or not')
        parent_parser.add_argument("--alpha1", type=float, default=1.0, help='alpha1 for ExpoMF')
        parent_parser.add_argument("--alpha2", type=float, default=1.0, help='alpha2 for ExpoMF')
        parent_parser.add_argument("--init_std", type=float, default=0.01, help='init std for ExpoMF')
        return parent_parser                         
        
    def _init_model(self, train_data):
        super()._init_model(train_data)   
        self.a = torch.ones(train_data.num_users, train_data.num_items)
        if not self.config['with_exposure_covariates']:
            self.mu = self.config['init_mu'] * torch.ones(train_data.num_items)
        else:
            class mu_model(basemodel.BaseRetriever):

                @staticmethod
                def _get_dataset_class():
                    return MFDataset

                def _init_parameter(self):
                    super()._init_parameter()
                    torch.nn.init.normal_(self.query_encoder.psi.weight, mean=0, std=self.config['init_std'])
                    torch.nn.init.constant_(self.query_encoder.psi_bias.weight, np.log(self.config['init_mu'] / (1 - self.config['init_mu'])))
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
                    psi = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
                    psi_bias = torch.nn.Embedding(train_data.num_users, 1)
                    class muQueryEncoder(torch.nn.Module):
                        def __init__(self, psi, psi_bias):
                            super().__init__()
                            self.psi = psi
                            self.psi_bias = psi_bias
                        def forward(self, batch):
                            return self.psi(batch), self.psi_bias(batch)
                    return muQueryEncoder(psi, psi_bias)

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
        if not self.config['with_exposure_covariates']:
            torch.nn.init.normal_(self.query_encoder.weight, mean=0, std=self.config['query_std'])
            torch.nn.init.normal_(self.item_encoder.weight, mean=0, std=self.config['item_std'])
            self.query_encoder.weight.requires_grad = False
            self.item_encoder.weight.requires_grad = False      
        else:
            torch.nn.init.normal_(self.query_encoder.query_emb.weight, mean=0, std=self.config['query_std'])
            torch.nn.init.normal_(self.item_encoder.item_emb.weight, mean=0, std=self.config['item_std'])
            self.query_encoder.query_emb.weight.requires_grad = False
            self.item_encoder.item_emb.weight.requires_grad = False

    @staticmethod
    def _get_dataset_class():
        return ALSDataset

    def _get_query_encoder(self, train_data):   #由于这个方法是em更新不是sgd，所以别的地方不用改query？
        query_emb = torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
        if not self.config['with_exposure_covariates']:
            return query_emb
        else:
            class ExpoMFQueryEncoder(torch.nn.Module):
                def __init__(self, query_emb):
                    super().__init__()
                    self.query_emb = query_emb
                def forward(self, batch):
                    query = self.query_emb(batch)
                    return query, batch
            return ExpoMFQueryEncoder(query_emb)

    def _get_item_encoder(self, train_data):
        item_emb = torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)
        if not self.config['with_exposure_covariates']:
            return item_emb
        else:
            class ExpoMFItemEncoder(torch.nn.Module):
                def __init__(self, item_emb):
                    super().__init__()
                    self.item_emb = item_emb
                def forward(self, batch):
                    items = self.item_emb(batch)
                    return items, batch
            return ExpoMFItemEncoder(item_emb)

    def _get_item_vector(self):
        num_items = self.item_encoder.item_emb.weight.shape[0]
        return self.item_encoder.item_emb.weight[1:], torch.arange(1, num_items)

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
                    items, iid = items
                    mu = self.mu_model.score_func(
                        self.mu_model.query_encoder(uid),
                        self.mu_model.item_encoder(iid)
                    )
                    return mu * super().forward(query, items)
            return ExpoMFScorer(self.mu_model)
        
    def _get_train_loaders(self, train_data):
        loader = train_data.loader(
            batch_size = self.config['batch_size'],
            shuffle = True, 
            num_workers = self.config['num_workers'], 
            drop_last = False)
        loader_T = train_data.transpose().loader(
            batch_size = self.config['batch_size'],
            shuffle = True, 
            num_workers = self.config['num_workers'], 
            drop_last = False)
        return [loader, loader_T]            

    def training_epoch(self, nepoch):
        trn_dataloaders, _ = self.current_epoch_trainloaders(nepoch)
        i=0
        for loader in trn_dataloaders:                       
            for batch in loader:
                # if batch[self.fuid].dim() == 1:
                #     break
                # data to device
                batch = self._to_device(batch, self.device)               
                # update latent user/item factors
                a = self._expectation(batch)
                self._maximization(batch, a)
                # i=i+1
                # print(i,a.all())
        
        if not self.config['with_exposure_covariates']:
            # update \mu_{i}; prior on \mu_{i} ~ Beta(alpha1, alpha2)
            self.mu = (self.config['alpha1'] + torch.sum(self.a, dim=0) - 1) / \
                    (self.config['alpha1'] + self.config['alpha2'] + self.a.shape[0] - 2)
        else:
            # update user exposure factors \psi and \psi_bias with mini-batch SGD      
            self.mu_model.fit(*self.mu_model.datasets[:2])            
                                             
        return torch.tensor(0.)
    
    def _expectation(self, batch):
        """
        Compute the posterior of exposure latent variables a_{ui}
        """
        if self.config['with_exposure_covariates']:
            mu = torch.sigmoid(
                self.mu_model.query_encoder[self.mu_model._get_query_feat(batch)] * \
                self.mu_model.item_encoder[self.mu_model._get_item_feat(batch)] + \
                self.mu_model.psi_bias[batch[self.fuid]])
        else:
            mu = self.mu[batch[self.fiid]]
            if batch[self.fuid].dim() == 1:
                P_y0_given_a1 = np.sqrt(self.config['lambda_y'] / 2 * torch.pi) * \
                        torch.exp(-self.config['lambda_y'] * \
                        torch.bmm(self.item_encoder(self._get_item_feat(batch)),                  # B x N x D
                                self.query_encoder(self._get_query_feat(batch)).unsqueeze(-1))    # B x D -> B x D x 1
                                .squeeze(-1)    
                                **2 / 2)                              
            else:
                mu = mu.expand(batch[self.fuid].shape[1], mu.shape[0]).transpose(0, 1)
                P_y0_given_a1 = np.sqrt(self.config['lambda_y'] / 2 * torch.pi) * \
                        torch.exp(-self.config['lambda_y'] * \
                        torch.bmm(self.query_encoder(self._get_query_feat(batch)),              # B x N x D
                                self.item_encoder(self._get_item_feat(batch)).unsqueeze(-1))    # B x D -> B x D x 1
                                .squeeze(-1)    
                                **2 / 2)
                        
        mu = self._to_device(mu, self.device)        
        a = (P_y0_given_a1 + 1e-8) / \
            (P_y0_given_a1 + 1e-8 + (1 - mu) / mu)        
        nnz = batch[self.frating].nonzero() 
        for idx in nnz:
            i, j = idx[0], idx[1]
            a[i, j] = torch.tensor(1.)

        # update self.a
        if batch[self.fuid].dim() == 1:    
            for i, uid in enumerate(batch[self.fuid]):
                for j, iid  in enumerate(batch[self.fiid][i]):
                    self.a[uid, iid] = a[i, j]
        else:
            for i, iid in enumerate(batch[self.fiid]):
                for j, uid  in enumerate(batch[self.fuid][i]):
                    self.a[uid, iid] = a[i, j]
        return a                                                                            # B x N         
    
    def _maximization(self, batch, a):
        """
        Update latent factors theta and beta
        """
        if batch[self.fuid].dim() == 1:
            beta_betaT = torch.zeros(batch[self.frating].shape[0], batch[self.frating].shape[1], 
                                    self.embed_dim, self.embed_dim, device=self.device)    # B x N x D x D
            item_embeddings = self.item_encoder(self._get_item_feat(batch))                        
            for i in range(batch[self.frating].shape[0]):
                for j in range(batch[self.frating].shape[1]):
                    beta_betaT[i, j] = item_embeddings[i, j].view(-1, 1) @ \
                        item_embeddings[i, j].view(1, -1) 
            A = self.config['lambda_y'] * torch.sum(
                a.unsqueeze(-1).unsqueeze(-1) * beta_betaT
                , dim = 1) + \
                self.config['lambda_theta'] * torch.eye(self.embed_dim, device=self.device)
            B = self.config['lambda_y'] * torch.sum(                                        # B x D
                (a * batch[self.frating]).unsqueeze(-1) *                                   # B x N -> B x N x 1
                self.item_encoder(self._get_item_feat(batch)), dim = 1)                     # B x N x D
            self.query_encoder.weight[batch[self.fuid]] = torch.linalg.solve(A, B)
        else:
            theta_thetaT = torch.zeros(batch[self.frating].shape[0], batch[self.frating].shape[1], 
                                        self.embed_dim, self.embed_dim, device=self.device) # B x N x D x D
            user_embeddings = self.query_encoder(self._get_query_feat(batch))                             
            for i in range(batch[self.frating].shape[0]):
                for j in range(batch[self.frating].shape[1]):
                    theta_thetaT[i, j] = user_embeddings[i, j].view(-1, 1) @ \
                        user_embeddings[i, j].view(1, -1) 
            A = self.config['lambda_y'] * torch.sum(
                a.unsqueeze(-1).unsqueeze(-1) * theta_thetaT
                , dim = 1) + \
                self.config['lambda_beta'] * torch.eye(self.embed_dim, device=self.device)
            B = self.config['lambda_y'] * torch.sum(                                        # B x D
                (a * batch[self.frating]).unsqueeze(-1) *                                   # B x N -> B x N x 1
                self.query_encoder(self._get_query_feat(batch)), dim = 1)                   # B x N x D 
            self.item_encoder.weight[batch[self.fiid]] = torch.linalg.solve(A, B)              