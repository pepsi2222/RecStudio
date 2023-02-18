import torch
from recstudio.model import basemodel, scorer
from recstudio.data import dataset

r"""
RelMF
######

Paper Reference:
    Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback (WSDM'20)
    https://doi.org/10.1145/3336191.3371783
"""

class RelMF(basemodel.BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('RelMF')
        parent_parser.add_argument("--eta", type=float, default=0.5, help='adjust propensities')
        return parent_parser
    
    def _init_model(self, train_data):
        super()._init_model(train_data)
        pop = (train_data.item_freq / torch.max(train_data.item_freq)) ** self.config['eta']
        self.register_buffer('pop', pop)
        self.rating_threshold = train_data.config.get('ranker_rating_threshold', 0)

    @staticmethod
    def _get_dataset_class():
        return dataset.MFDataset
    
    def _get_query_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)
    
    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        return torch.nn.BCEWithLogitsLoss()
    
    def training_step(self, batch):
        output = self.forward(batch, False)
        score = output['score']
        
        label = (batch[self.frating] > self.rating_threshold).float()
        pop = self.pop[batch[self.fiid]]
        score['label'] = label / (pop + 1e-7) + (1 - label) * (1 - label / (pop + 1e-7))
        
        loss_value = self.loss_fn(input=score['pos_score'], target=score['label'])
        return loss_value
    
    def _get_sampler(self, train_data):
        return None