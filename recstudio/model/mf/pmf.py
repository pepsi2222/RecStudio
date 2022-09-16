import torch
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model import basemodel, scorer, loss_func

class PMF(basemodel.BaseRetriever):

    def add_model_specific_args(parent_parser):
        parent_parser = basemodel.Recommender.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('PMF')
        parent_parser.add_argument("--init_range", type=float, default=0.1, help='std for PMF')
        return parent_parser

    @staticmethod
    def _get_dataset_class():
        return dataset.MFDataset

    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def _get_query_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_users, self.embed_dim, padding_idx=0)

    def _get_score_func(self):
        return scorer.InnerProductScorer()

    def _get_loss_func(self):
        return loss_func.SquareLoss()
