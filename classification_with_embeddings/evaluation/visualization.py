import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from classification_with_embeddings.evaluation import logger


def write_classification_report(cr: str, dir_path: str, method: str) -> None:
    """Write classification report to file.

    :param cr: classification report to write to file
    :param dir_path: path to directory in which to save the file containing the classification report
    :param method: file name (embedding method used)
    """

    output_file_path = os.path.abspath(os.path.join(dir_path, method + '_cr.txt'))
    logger.info('Writing classification report to {0}'.format(output_file_path))
    with open(output_file_path, 'w') as f:
        f.write(cr)


def plot_confusion_matrix(predictions, y_test, labels, class_names, plot_path: str, method) -> None:
    """Plot confusion matrix

    :param predictions: predictions of the classifier
    :param y_test: ground truth values
    :param labels: unique labels
    :param class_names: names associated with the labels (in same order)
    :param plot_path: path to directory in which to store the plot
    :param method: plot file name (embedding method used)
    """

    output_file_path = os.path.abspath(os.path.join(plot_path, method + '.png'))
    logger.info('Saving confusion matrix plot to {0}'.format(output_file_path))

    # Plot confusion matrix and save plot.
    np.set_printoptions(precision=2)
    disp = ConfusionMatrixDisplay.from_predictions(
        labels=labels,
        display_labels=class_names,
        y_true=y_test,
        y_pred=predictions,
        normalize='true',
        xticks_rotation='vertical'
    )

    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.clf()
    plt.close()
