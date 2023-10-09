import argparse
import sys
from typing import List

import numpy as np
from sklearn import metrics

from result_visualization_utils import util


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # parse arguments
    parser = argparse.ArgumentParser(prog='print-metrics')
    parser.add_argument('--evaluation-data-paths', type=str, required=True, nargs='+',
                        help='Path to evaluation data files as produced by classification-with-embeddings')
    parser.add_argument('--method-names', type=str, required=False, nargs='*',
                        help='Names of evaluated methods (in same order as corresponding evaluation data files)')

    parsed_args = vars(parser.parse_args(argv[1:]))

    evaluation_data_paths = parsed_args['evaluation_data_paths']

    # parse scores and true values
    scores, true_values = util.parse_scores_and_true_values(evaluation_data_paths)

    for idx in range(len(evaluation_data_paths)):
        print_metrics(parsed_args['method_names'][idx], true_values[idx], scores[idx])
        print('###')
        print_metrics_ci(parsed_args['method_names'][idx], true_values[idx], scores[idx])
        print('###')


def print_metrics(method: str, true_values: list, scores: np.ndarray):
    """Print metrics for evaluated method.

    :param method: name of evaluated method
    :param true_values: ground truth values
    :param scores: scores for classes (probabilities)
    """

    auc = compute_auc(true_values, scores)
    auprc = compute_auprc(true_values, scores)
    rp60 = compute_rp(true_values, scores, 60)

    print('Metrics for method \'{}\':'.format(method))
    print('AUC: {}'.format(round(auc, 3)))
    print('AUPRC: {}'.format(round(auprc, 3)))
    print('RP80: {}'.format(round(rp60, 3)))


def print_metrics_ci(method: str, true_values: list, scores: np.ndarray, n_iterations: int = 1000):
    """Print metrics for evaluated method with confidence intervals
    computed using bootstrapping of results.

    :param method: name of evaluated method
    :param true_values: ground truth values
    :param scores: scores for classes (probabilities)
    :param n_iterations: number of iterations to perform
    """
    auc_vals = []
    auprc_vals = []
    rp60_vals = []

    rng = np.random.RandomState(seed=42)
    idxs = np.arange(len(true_values))

    for iteration_idx in range(n_iterations):
        sample_idxs = rng.choice(idxs, len(idxs), replace=True)
        true_values_sampled = [true_values[idx] for idx in sample_idxs]
        scores_sampled = scores[sample_idxs, :]

        auc_vals.append(compute_auc(true_values_sampled, scores_sampled))
        auprc_vals.append(compute_auprc(true_values_sampled, scores_sampled))
        rp60_vals.append(compute_rp(true_values_sampled, scores_sampled, 60))

    auc_mean = np.mean(auc_vals)
    auprc_mean = np.mean(auprc_vals)
    rp60_mean = np.mean(rp60_vals)

    auc_moe = compute_margin_of_error(auc_vals)
    auprc_moe = compute_margin_of_error(auprc_vals)
    rp60_moe = compute_margin_of_error(rp60_vals)

    print('Metrics with MOE (95% confidence intervals) for method \'{}\':'.format(method))
    print('AUC: {}±{}'.format(round(auc_mean, 3), round(auc_moe, 3)))
    print('AUPRC: {}±{}'.format(round(auprc_mean, 3), round(auprc_moe, 3)))
    print('RP60: {}±{}'.format(round(rp60_mean, 3), round(rp60_moe, 3)))


def compute_margin_of_error(data: List[float]):
    """Compute margin of error for metrics for bootstrapped sampled classification result data.

    :param data: metrics computed from sampled classification result data
    """
    ci_lower = np.percentile(data, 2.5)
    ci_upper = np.percentile(data, 97.5)
    return (ci_upper - ci_lower) / 2


def compute_auc(true_values: list, scores: np.ndarray):
    """Compute AUC value for scores given true values.

    :param true_values: ground truth values
    :param scores: scores for classes (probabilities)
    """

    return metrics.roc_auc_score(true_values, scores[:, 1])


def compute_auprc(true_values: list, scores: np.ndarray):
    """Compute AUPRC value for scores given true values.

    :param true_values: ground truth values
    :param scores: scores for classes (probabilities)
    """

    return metrics.average_precision_score(true_values, scores[:, 1])


def compute_rp(true_values: list, scores: np.ndarray, precision_percentage: int):
    """Compute recall at the threshold for the specified precision percentage.

    :param true_values: ground truth values
    :param scores: scores for classes (probabilities)
    :param precision_percentage: precision percentage
    """

    # compute precision and recall for different thresholds
    precision, recall, thresholds = metrics.precision_recall_curve(true_values, scores[:, 1])

    # find the index where precision is closest to the specified precision and get the corresponding threshold
    closest_precision_index = np.argmin(np.abs(precision - (precision_percentage / 100)))
    thresh = thresholds[closest_precision_index-1]

    # calculate precision at the specified recall percentage
    return metrics.recall_score(true_values, (scores[:, 1] > thresh))


if __name__ == '__main__':
    main()
