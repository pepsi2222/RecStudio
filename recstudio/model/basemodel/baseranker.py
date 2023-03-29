from collections import defaultdict
import inspect
import torch
import recstudio.eval as eval
from .recommender import Recommender
from typing import Dict, List, Optional, Tuple, Union


class BaseRanker(Recommender):

    # For the training of Ranker, we consider the following two cases:
    #    1. The dataset is exposure dataset, which contains both exposed and clicked interactions and
    #       exposed and unclicked interactions. For example, the CTR task aims to model the click rate
    #       for users. In this case, negative samples commonly come from the dataset, sampling technique
    #       is not used.
    #    2. The dataset is click dataset, which contains only exposed and click interactions. Due to lack
    #       of negatives, usually negative sampling technique is used. We support negative sampling by two
    #       ways. The first way is to combine a retriever with the ranker, where the retriever is used as a negative
    #       sampler. The second way is to sampling negatives in dataset, which aims to make the dataset looks like
    #       the dataset in case 1. In addition, the negative samples are flatten as a point-wise way like in case 1
    #       but not the pair-wise.
    #
    # For the other explaination of the dataset, we can regard the dataset as the following two cases:
    #   1. Point-wise dataset that contains positive and negative samples. To be stressed is that the negative samples
    #      could be exposed but unclicked interactions (true negatives) or sampled negatives (sampled negatives), which
    #      is dicided by the original dataset. For example, Gowalla dataset contains clicked interactions only, so we
    #      need to sample to get negatives to train ranker like FM. But for the rating dataset like MovieLens, we can
    #      get negatives in the dataset by setting `ranker_rating_threshold` in the configuration file. In the case, the
    #      loss function is usually choosed as point-wise BCE loss (not the BinaryCrossEntropyLoss in our loss_func).
    #      In addition, topk-based metrics are not supported in the case, usually metrics like AUC, logloss are used.
    #   2. Point-wise dataset that contains positive. This kind of dataset is usually used for training retriever. There
    #      is only positives in dataset so we need to sample negatives in the training procedure. However, there is no
    #      sampler in ranker model, which makes it necessary to combine a retriever. By combining a retriever, not only
    #      negatives samples could be got, but also the topk function is supported by a two-stage way (candidates are
    #      generated by the retriever and then be reranked). In this case, usually the pair-wise loss function are used,
    #      like SampledSoftmax, BinaryCrossEntropyLoss, et al. In addition, topk metrics are supported in the case, while
    #      non-topk-based metrics is not supported.
    #

    def _init_model(self, train_data, drop_unused_field=True):
        super()._init_model(train_data, drop_unused_field)
        self.fiid = train_data.fiid
        self.fuid = train_data.fuid
        if self.retriever is None:
            self.retriever = self._get_retriever(train_data)
        if self.retriever is None:
            self.logger.warning('No retriever is used, topk metrics is not supported.')

    def _set_data_field(self, data):
        # token_field = set([k for k, v in data.field2type.items() if v=='token'])
        # token_field.add(data.frating)
        # data.use_field = token_field
        data.use_field = data.field2type.keys()

    def _get_retriever(self, train_data):
        return None

    def _generate_multi_item_batch(self, batch, item_ids):
        num_item = item_ids.size(-1)
        item_ids = item_ids.view(-1)
        items = self._get_item_feat(item_ids)
        if isinstance(items, torch.Tensor): # only id
            items = {self.fiid: items}  # convert to dict
        multi_item_batch = {}
        #
        for k, v in batch.items():
            if (k not in items):
                multi_item_batch[k] = v.unsqueeze(1) \
                                       .expand(-1, num_item, *tuple([-1 for i in range(len(v.shape)-1)]))
                multi_item_batch[k] = multi_item_batch[k].reshape(-1, *(v.shape[1:]))
            else:
                multi_item_batch[k] = items[k]
        return multi_item_batch

    def forward(self, batch):
        # calculate scores
        if self.retriever is None:
            output = self.score(batch)
            return {'pos_score': output['score'], 'label': batch[self.frating]}, output
        else:
            # only positive samples in batch
            assert self.neg_count is not None, 'expecting neg_count is not None.'
            pos_output = self.score(batch)
            pos_prob, neg_item_idx, neg_prob = self.retriever.sampling(
                batch, self.neg_count, method=self.config['train']['sampling_method'])
            # 
            neg_output = self.score_multi(batch, neg_item_idx) # neg_item_idx: [batch_size, neg], score: [batch_size * neg]
            return {'pos_score': pos_output['score'], 'log_pos_prob': pos_prob, 'neg_score': neg_output['score'].view(-1, self.neg_count),
                    'log_neg_prob': neg_prob, 'label': batch[self.frating]}, (pos_output, neg_output)

    def score_multi(self, batch, item_ids=None):
        # score for a batch or a batch with different items for each context
        if item_ids is not None:
            neg_batch = self._generate_multi_item_batch(batch, item_ids)
            num_neg = item_ids.size(-1) 
            return self.score(neg_batch)
        else:
            return self.score(batch)

    def score(self, batch):
        # score for a single interaction.
        # should be overload for each ranker model.
        pass

    def build_index(self):
        raise NotImplementedError("build_index for ranker not implemented now.")

    def training_step(self, batch):
        y_h, output = self.forward(batch)
        loss = self.loss_fn(**y_h) # + loss using output
        return loss

    def validation_step(self, batch):
        eval_metric = self.config['eval']['val_metrics']
        if self.config['eval']['cutoff'] is not None:
            cutoffs = self.config['eval']['cutoff'] if isinstance(self.config['eval']['cutoff'], List) \
                else [self.config['eval']['cutoff']]
        else:
            cutoffs = None
        return self._test_step(batch, eval_metric, cutoffs)

    def test_step(self, batch):
        eval_metric = self.config['eval']['test_metrics']
        if self.config['eval']['cutoff'] is not None:
            cutoffs = self.config['eval']['cutoff'] if isinstance(self.config['eval']['cutoff'], List) \
                else [self.config['eval']['cutoff']]
        else:
            cutoffs = None
        return self._test_step(batch, eval_metric, cutoffs)

    def topk(self, batch, topk, user_hist=None, return_candidates=False):
        retriever = self.retriever
        if (retriever is None):
            raise NotImplementedError("`topk` function not supported for ranker without retriever.")
        else:
            score_re, topk_items_re = retriever.topk(batch, retriever.config['eval']['topk'], user_hist)
            score = self.score_multi(batch, topk_items_re)['score'].view(topk_items_re.shape[0], -1) # [batch, topk]
            assert topk <= retriever.config['eval']['topk'], '`topk` of ranker must be smaller than the retriever.'
            scores, _idx = torch.topk(score, topk, dim=-1)
            topk_items = torch.gather(topk_items_re, -1, _idx)
            if return_candidates:
                return score, topk_items, score_re, topk_items_re
            else:
                return score, topk_items

    def _test_step(self, batch, metric, cutoffs=None):
        rank_m = eval.get_rank_metrics(metric)
        pred_m = eval.get_pred_metrics(metric)
        bs = batch[self.frating].size(0)
        if len(rank_m) > 0:
            assert cutoffs is not None, 'expected cutoffs for topk ranking metrics.'
        # TODO: discuss in which cases pred_metrics should be calculated. According to whether there are neg labels in dataset?
        # When there are neg labels in dataset, should rank_metrics be considered?
        if self.retriever is None:
            result, _ = self.forward(batch)
            global_m = eval.get_global_metrics(metric)
            metrics = {}
            for n, f in pred_m:
                if not n in global_m:
                    if len(inspect.signature(f).parameters) > 2:
                        metrics[n] = f(result['pos_score'], result['label'], 
                                       self.config['eval']['binarized_prob_thres'])
                    else:
                        metrics[n] = f(result['pos_score'], result['label'])
            if len(global_m) > 0:
                # gather scores and labels for global metrics like AUC.
                metrics['score'] = result['pos_score'].detach()
                metrics['label'] = result['label']
        else:
            # pair-wise, support topk-based metrics, like [NDCG, Recall, Precision, MRR, MAP, MR, et al.]
            # The case is suitable for the scene where there are only positives in dataset.
            topk = self.config['eval']['topk']
            score, topk_items = self.topk(batch, topk, batch['user_hist'])
            if batch[self.fiid].dim() > 1:
                target, _ = batch[self.fiid].sort()
                idx_ = torch.searchsorted(target, topk_items)
                idx_[idx_ == target.size(1)] = target.size(1) - 1
                label = torch.gather(target, 1, idx_) == topk_items
                pos_rating = batch[self.frating]
            else:
                label = batch[self.fiid].view(-1, 1) == topk_items
                pos_rating = batch[self.frating].view(-1, 1)
            metrics = {f"{name}@{cutoff}": func(label, pos_rating, cutoff) for cutoff in cutoffs for name, func in rank_m}
        return metrics, bs

    def _test_epoch_end(self, outputs):
        metric_list, bs = zip(*outputs)
        bs = torch.tensor(bs)
        out = defaultdict(list)
        for o in metric_list:
            for k, v in o.items():
                out[k].append(v)
        if 'score' in out:
            # gather scores and labels for global metrics
            scores = torch.cat(out['score'], dim=0)
            labels = torch.cat(out['label'], dim=0)
            del out['score']
            del out['label']
        for k, v in out.items():
            metric = torch.tensor(v)
            out[k] = (metric * bs).sum() / bs.sum()
        #
        # calculate global metrics like AUC.
        global_m = eval.get_global_metrics(out)
        if len(global_m) > 0:
            for m, f in global_m:
                out[m] = f(scores, labels)
        return out
