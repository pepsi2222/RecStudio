from . import BaseRetriever
from typing import Dict, List, Optional, Tuple, Union
import torch

class DebiasedModel(BaseRetriever):
    def __init__(self, config: Dict = None, **kwargs):
        super().__init__(self, config, kwargs)

        if 'query_post_processing_module' in kwargs:
            assert isinstance(kwargs['query_post_processing_module'], torch.nn.Module), \
                "query_post_processing_module must be torch.nn.Module"
            self.query_post_processing_module = kwargs['query_post_processing_module']
        else:
            self.query_post_processing_module = self._get_query_post_processing_module()
            
        if 'item_post_processing_module' in kwargs:
            assert isinstance(kwargs['item_post_processing_module'], torch.nn.Module), \
                "item_post_processing_module must be torch.nn.Module"
            self.item_post_processing_module = kwargs['item_post_processing_module']
        else:
            self.item_post_processing_module = self._get_item_post_processing_module()

        if 'propensity' in kwargs:
            assert isinstance(kwargs['propensity'], torch.nn.Module), \
                "propensity must be torch.nn.Module"
            self.propensity = kwargs['propensity']
        else:
            self.propensity = None

    def _init_model(self, train_data):
        super()._init_model(train_data)
        self.propensity = self._get_propensity(train_data) if not self.propensity else self.propensity

    def _get_query_post_processing_module(self):
        return None

    def _get_item_post_processing_module(self):
        return None

    def _get_propensity(self, train_data)
        return None
    
    def get_scores(query, pos_item_vec, neg_item_vec=None, item_vectors=None):
        score = {'pos_score': self.score_func(query, pos_item_vec)}
        if neg_item_vec is not None:
            score['neg_score'] = self.score_func(query, neg_item_vec)
        if item_vectors is not None:
            score['all_score'] = self.score_func(query, item_vectors)
        return score

    def forward(self, batch, full_score, return_query=False, return_item=False, return_neg_item=False, return_neg_id=False):
        # query_vec, pos_item_vec, neg_item_vec,
        output = {}
        pos_items = self._get_item_feat(batch)
        pos_item_vec = self.item_encoder(pos_items)

        if self.item_post_processing_module is not None:
            post_processed_pos_item_vec = self.item_post_processing_module(pos_item_vec)
        else:
            post_processed_pos_item_vec = pos_item_vec

        if self.sampler is not None:
            if self.neg_count is None:
                raise ValueError("`negative_count` value is required when "
                                 "`sampler` is not none.")

            (log_pos_prob, neg_item_idx, log_neg_prob), query = self.sampling(batch=batch, num_neg=self.neg_count,
                                                                              excluding_hist=self.config.get('excluding_hist', False),
                                                                              method=self.config.get('sampling_method', 'none'), return_query=True)
            if self.query_post_processing_module is not None:
                post_processed_query_vec = self.query_post_processing_module(query)
            else:
                post_processed_query_vec = query

            # pos_score = self.score_func(query, pos_item_vec)

            if batch[self.fiid].dim() > 1:
                pos_score[batch[self.fiid] == 0] = -float('inf')  # padding

            neg_items = self._get_item_feat(neg_item_idx)
            neg_item_vec = self.item_encoder(neg_items)

            if self.item_post_processing_module is not None:
                post_processed_neg_item_vec = self.item_post_processing_module(neg_item_vec)
            else:
                post_processed_neg_item_vec = neg_item_vec

            # neg_score = self.score_func(query, neg_item_vec)

            # output['score'] = {'pos_score': pos_score, 'log_pos_prob': log_pos_prob,
            #                    'neg_score': neg_score, 'log_neg_prob': log_neg_prob}
            output['score'] = {'log_pos_prob': log_pos_prob, 'log_neg_prob': log_neg_prob}
            output['score'].update(
                self.get_scores(
                    query=post_processed_query_vec,
                    pos_item_vec=post_processed_pos_item_vec,
                    neg_item_vec=post_processed_neg_item_vec
                ))

            if return_neg_item:
                output['neg_item'] = neg_item_vec
            if return_neg_id:
                output['neg_id'] = neg_item_idx
        else:
            query = self.query_encoder(self._get_query_feat(batch))

            if self.query_post_processing_module is not None:
                post_processed_query_vec = self.query_post_processing_module(query)
            else:
                post_processed_query_vec = query

            # pos_score = self.score_func(query, pos_item_vec)
            if batch[self.fiid].dim() > 1:
                pos_score[batch[self.fiid] == 0] = -float('inf')  # padding
            # output['score'] = {'pos_score': pos_score}
            
            if full_score:
                item_vectors = self._get_item_vector()

                if self.item_post_processing_module is not None:
                    post_processed_item_vectors = self.item_post_processing_module(item_vectors)
                else:
                    post_processed_item_vectors = item_vectors
                # all_item_scores = self.score_func(query, item_vectors)
                # output['score']['all_score'] = all_item_scores
            else:
                post_processed_item_vectors = None

            output['score'] = self.get_scores(
                query=post_processed_query_vec,
                pos_item_vec=post_processed_pos_item_vec,
                item_vectors=post_processed_item_vectors
            )

        if return_query:
            output['query'] = query
        if return_item:
            output['item'] = pos_item_vec
        return output

    def training_step(self, batch):
        output = self.forward(batch, full_score=False)
        score = output['score']
        score['label'] = batch[self.frating]
        if self.propensity is not None:
            score['propensity'] = self.propensity(batch)
        loss_value = self.loss_fn(**score)
        return loss_value

    