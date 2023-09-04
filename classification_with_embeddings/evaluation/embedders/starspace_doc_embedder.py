import os
import subprocess
from typing import Union, List, Iterator, Dict, Optional

import numpy as np

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.embedding import embed_util
from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder
from classification_with_embeddings.util.errors import EmbeddingError


class StarspaceDocEmbedder(ADocEmbedder):
    _TMP_TRAINING_DATA_PATH = './train_data_starspace.txt'
    _TMP_PREDICT_DATA_PATH = './predict_data_starspace.txt'
    _TMP_STARSPACE_MODEL_NAME = 'starspace_model'

    def __init__(self, embedding_kwargs: Optional[dict] = None, **model_init_kwargs):
        super().__init__(embedding_kwargs=embedding_kwargs, **model_init_kwargs)
        self.starspace_path = model_init_kwargs['starspace_path']

    def get_word_to_embedding(self, train_sentences: Union[List[List[str]], Iterator], y: list) -> Dict[str, np.ndarray]:

        # write training data to a temporary file
        self._write_training_data_to_file(train_sentences, y)

        # get StarSpace embeddings
        p = subprocess.run(
            [self.starspace_path, 'train', '-trainFile', self._TMP_TRAINING_DATA_PATH, '-model', './' + self._TMP_STARSPACE_MODEL_NAME] +
            self._embedding_kwargs_to_starspace_params()
        )

        if p.returncode != 0:
            raise EmbeddingError('StarSpace', p.returncode)

        word_to_embedding = embed_util.get_word_to_embedding('./starspace_model.tsv')

        # self._remove_tmp_files()

        return word_to_embedding

    def transform(self, X: List[List[str]]):
        self._write_prediction_data_to_file(X)
        p = subprocess.run(
            [os.path.join(os.path.dirname(self.starspace_path), 'embed_doc'), './' + self._TMP_STARSPACE_MODEL_NAME, self._TMP_PREDICT_DATA_PATH],
            stdout=subprocess.PIPE,
            text=True
        )
        # TODO remove prediction file
        return np.array(list(map(lambda x: [float(e) for e in x.strip().split(' ')], p.stdout.split('\n')[5::2])))

    def _write_training_data_to_file(self, train_sentences: Union[List[List[str]], Iterator], y: list):
        # write training data to file
        with open(self._TMP_TRAINING_DATA_PATH, 'w') as f:
            for idx in range(len(train_sentences)):
                f.write(' '.join(train_sentences[idx]))
                f.write(' ' + LABEL_WORD_PREFIX + str(y[idx]))
                f.write('\n')

    def _write_prediction_data_to_file(self, predict_sentences: Union[List[List[str]], Iterator]):
        # write training data to file
        with open(self._TMP_PREDICT_DATA_PATH, 'w') as f:
            for idx in range(len(predict_sentences)):
                f.write(' '.join(predict_sentences[idx]))
                f.write('\n')

    def _remove_tmp_files(self):
        os.remove(self._TMP_TRAINING_DATA_PATH)

    def _embedding_kwargs_to_starspace_params(self) -> List[str]:
        res = []
        for k, v in self.embedding_kwargs.items():
            res.append('-{}'.format(k))
            res.append(str(v))
        return res
