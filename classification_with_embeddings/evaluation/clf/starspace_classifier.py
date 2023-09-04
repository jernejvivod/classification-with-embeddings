from typing import List

import numpy as np

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.embedding import embed_util
from classification_with_embeddings.evaluation.clf.a_classifier import AClassifier


class StarSpaceClassifier(AClassifier):
    """Classifier initialized with a mapping from words and labels to their embeddings obtained using StarSpace."""

    def __init__(self):
        self.word_to_embedding = None
        self.index_to_label_key = None
        self.label_emb_mat = None

        self.classes_ = None

    def fit(self, X, y):
        """Fit classifier to training data.

        :param X: training data examples
        :param y: training data labels
        """
        if self.word_to_embedding is None:
            self.word_to_embedding = embed_util.get_word_to_embedding('./starspace_model.tsv')

            label_embeddings = [
                (key, self.word_to_embedding[key]) for key in self.word_to_embedding.keys() if LABEL_WORD_PREFIX in key
            ]
            label_embeddings = sorted(label_embeddings, key=lambda x: x[0])

            self.index_to_label_key = [e[0] for e in label_embeddings]
            self.classes_ = [int(e.replace(LABEL_WORD_PREFIX, '')) for e in self.index_to_label_key]
            self.label_emb_mat = np.transpose([e[1] for e in label_embeddings])

    def predict(self, X):
        sims = self._get_sims(X)
        return np.array([int(self.index_to_label_key[idx].replace(LABEL_WORD_PREFIX, '')) for idx in np.argmax(sims, axis=1)])

    def supports_predict_proba(self):
        return True

    def predict_proba(self, sentences: List[List[str]]):
        sims = self._get_sims(sentences)
        return np.exp(sims) / np.sum(np.exp(sims))

    def classes(self):
        return [int(e.replace(LABEL_WORD_PREFIX, '')) for e in self.index_to_label_key]

    def _get_sims(self, X: np.ndarray):
        return np.matmul(X, self.label_emb_mat)
