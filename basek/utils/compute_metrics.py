from collections import defaultdict

from basek.utils.imports import numpy as np

from concurrent.futures import ThreadPoolExecutor

import time


class ComputeMetrics():

    def __init__(self, match_points, outfile='./metrics'):
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.match_points = sorted(list(set(match_points)))
        self.max_match_point = self.match_points[-1]
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.count = 0
        self.out = open(outfile, 'a')

    def reset(self):
        self.out.write('=' * 52 + f'  test epoch-{self.count:4d} finished  ' + '=' * 52 + '\n')
        self.out.write('#' * 132 + '\n')
        self.out.flush()
        self.metrics = defaultdict(lambda: defaultdict(list))

    def add_one_batch(self, I, ground_truth_iid_seq, ground_truth_seq_len, hist_iid_seq, hist_seq_len):
        batch_size = len(I)
        match_points = self.match_points
        max_match_point = self.max_match_point

        hits_matrix = np.zeros(shape=(batch_size, max_match_point), dtype=np.int64)
        novelty_hits_matrix = np.zeros(shape=(batch_size, max_match_point), dtype=np.int64)
        mrrs = np.zeros(shape=(batch_size, max_match_point))
        groud_truth_count = np.zeros(shape=(batch_size), dtype=np.int64)
        idcgs = np.tile(
            np.cumsum([1.0 / np.log2(i + 2) for i in range(max_match_point)]).reshape(1, -1),
            (batch_size, 1)
        )
        ndcgs = np.zeros(shape=(batch_size, max_match_point))

        drop_hits_matrix = hits_matrix.copy()
        drop_mrrs = mrrs.copy()
        drop_groud_truth_count = groud_truth_count.copy()
        drop_idcgs = idcgs.copy()
        drop_ndcgs = ndcgs.copy()
        drop_non_zeros = []

        for per in range(batch_size):
            per_pred = I[per]
            per_len_ground_truth = np.squeeze(ground_truth_seq_len[per])
            per_len_hist = np.squeeze(hist_seq_len[per])
            groud_truth_count[per] = per_len_ground_truth

            per_hits = hits_matrix[per]
            per_idcgs = idcgs[per]
            per_mrrs = mrrs[per]

            per_ground_truth_iid_seq = ground_truth_iid_seq[per][:per_len_ground_truth]
            per_ground_truth_iid_set = set(per_ground_truth_iid_seq)
            per_idcgs_cut_point = min(per_len_ground_truth, max_match_point)
            per_hist_iid_seq = hist_iid_seq[per][:per_len_hist]
            per_hist_iid_set = set(per_hist_iid_seq)

            per_drop_ground_truth_iid_seq = list(filter(lambda x: x not in per_hist_iid_set, per_ground_truth_iid_seq))
            per_drop_ground_truth_iid_seq = np.array(per_drop_ground_truth_iid_seq)
            if per_drop_ground_truth_iid_seq.shape[0] != 0:
                drop_non_zeros.append(per)

            pre_hits_idx, per_drop_hits_idx, pre_novelty_hits_idx = [], [], []
            for pred_point, pred in enumerate(per_pred):
                if pred not in per_hist_iid_set:
                    pre_novelty_hits_idx.append(pred_point)
                if pred in per_ground_truth_iid_set:
                    pre_hits_idx.append(pred_point)
                    if pred not in per_hist_iid_set:
                        per_drop_hits_idx.append(pred_point)

            if len(pre_hits_idx) != 0:
                per_hits[pre_hits_idx] = 1
                first_match_point = pre_hits_idx[0]
                per_mrrs[first_match_point:] = 1.0 / (first_match_point + 1)
                per_idcgs[per_idcgs_cut_point + 1:] = per_idcgs[per_idcgs_cut_point]
                per_dgs = np.zeros_like(per_idcgs)
                per_dgs[pre_hits_idx] = 1.0 / np.log2(np.array(pre_hits_idx) + 2)
                per_dcgs = np.cumsum(per_dgs)
                ndcgs[per] = per_dcgs / (per_idcgs + 1e-16)

            per_drop_hits = drop_hits_matrix[per]
            per_drop_idcgs = drop_idcgs[per]
            per_drop_mrrs = drop_mrrs[per]
            per_drop_len_ground_truth = per_drop_ground_truth_iid_seq.shape[0]
            drop_groud_truth_count[per] = per_drop_len_ground_truth
            per_drop_idcgs_cut_point = min(per_drop_len_ground_truth, max_match_point)

            if len(per_drop_hits_idx) != 0:
                per_drop_hits[per_drop_hits_idx] = 1
                first_match_point = per_drop_hits_idx[0]
                per_drop_mrrs[first_match_point:] = 1.0 / (first_match_point + 1)
                per_drop_idcgs[per_drop_idcgs_cut_point + 1:] = per_drop_idcgs[per_drop_idcgs_cut_point]
                per_drop_dgs = np.zeros_like(per_drop_idcgs)
                per_drop_dgs[per_drop_hits_idx] = 1.0 / np.log2(np.array(per_drop_hits_idx) + 2)
                per_drop_dcgs = np.cumsum(per_drop_dgs)
                drop_ndcgs[per] = per_drop_dcgs / (per_drop_idcgs / 1e-16)

            per_novelty_hits = novelty_hits_matrix[per]
            per_novelty_hits[pre_novelty_hits_idx] = 1

        groud_truth_count = groud_truth_count.reshape(-1, 1) + 1e-16
        cumsum_hits_matrix = np.cumsum(hits_matrix, axis=-1)
        recalls = cumsum_hits_matrix / groud_truth_count
        precisions = cumsum_hits_matrix / np.arange(1, max_match_point + 1).reshape(1, -1)
        f1_scores = (2 * recalls * precisions) / (recalls + precisions + 1e-16)
        maps = np.cumsum(np.where(hits_matrix, precisions, np.zeros_like(precisions)), axis=-1) / groud_truth_count

        drop_groud_truth_count = drop_groud_truth_count.reshape(-1, 1) + 1e-16
        drop_cumsum_hits_matrix = np.cumsum(drop_hits_matrix, axis=-1)
        drop_recalls = drop_cumsum_hits_matrix / drop_groud_truth_count
        drop_precisions = drop_cumsum_hits_matrix / np.arange(1, max_match_point + 1).reshape(1, -1)
        drop_f1_scores = (2 * drop_recalls * drop_precisions) / (drop_recalls + drop_precisions + 1e-16)
        drop_maps = \
            np.cumsum(np.where(drop_hits_matrix, drop_precisions, np.zeros_like(drop_precisions)), axis=-1) \
            / drop_groud_truth_count

        novelties = np.cumsum(novelty_hits_matrix, axis=-1) / np.arange(1, max_match_point + 1).reshape(1, -1)

        batch_metrics = defaultdict(dict)
        for match_point in match_points:
            batch_metrics['drop_Recall'][match_point] = drop_recalls[drop_non_zeros, match_point - 1]
            batch_metrics['drop_Precision'][match_point] = drop_precisions[drop_non_zeros, match_point - 1]
            batch_metrics['drop_F1_Score'][match_point] = drop_f1_scores[drop_non_zeros, match_point - 1]
            batch_metrics['drop_NDCG'][match_point] = drop_ndcgs[drop_non_zeros, match_point - 1]
            batch_metrics['drop_MAP'][match_point] = drop_maps[drop_non_zeros, match_point - 1]
            batch_metrics['drop_MRR'][match_point] = drop_mrrs[drop_non_zeros, match_point - 1]
            batch_metrics['Novelty'][match_point] = novelties[:, match_point - 1]
            batch_metrics['MRR'][match_point] = mrrs[:, match_point - 1]
            batch_metrics['MAP'][match_point] = maps[:, match_point - 1]
            batch_metrics['NDCG'][match_point] = ndcgs[:, match_point - 1]
            batch_metrics['F1_Score'][match_point] = f1_scores[:, match_point - 1]
            batch_metrics['Precision'][match_point] = precisions[:, match_point - 1]
            batch_metrics['Recall'][match_point] = recalls[:, match_point - 1]

        for per_metric, per_batch_metric_values in batch_metrics.items():
            for match_point, per_batch_metric_value in per_batch_metric_values.items():
                self.metrics[per_metric][match_point].append(per_batch_metric_value)

    def print_metrics(self):
        aggregated_metrics = defaultdict(dict)
        for per_metric, per_metric_values in self.metrics.items():
            for math_point, per_metric_value in per_metric_values.items():
                aggregated_metrics[per_metric][math_point] = \
                    np.mean(np.concatenate(per_metric_value, axis=0))
        print('\n' + '#' * 132)
        for per_metric, per_aggregated_metric_values in aggregated_metrics.items():
            header = '=' * 52 + f'    {per_metric}    ' + '=' * 52
            print(header)
            self.out.write(header + '\n')
            per_metric_str = ''
            for math_point, per_aggregated_metric_value in per_aggregated_metric_values.items():
                per_metric_str = per_metric_str + '-' + \
                    f' @{math_point:3d}: {per_aggregated_metric_value:.10f} ' + '-'
                if len(per_metric_str) > 120:
                    print(per_metric_str)
                    self.out.write(per_metric_str + '\n')
                    per_metric_str = ''
            if per_metric_str:
                print(per_metric_str)
                self.out.write(per_metric_str + '\n')
        self.out.flush()
        self.reset()
