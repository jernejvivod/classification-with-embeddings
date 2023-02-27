import unittest

from classification_with_embeddings.embedding.embed import get_word_to_embedding
from classification_with_embeddings.evaluation.get_clf import get_clf_with_internal_clf, get_clf_starspace
from test.test_utils import get_relative_path


class TestGetClf(unittest.TestCase):
    def test_get_clf_with_internal_clf(self):
        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_model.tsv'))
        training_data_path = get_relative_path(__file__, '../mock_data/train.txt')
        clf = get_clf_with_internal_clf(word_to_embedding, training_data_path)
        self.assertIsNotNone(clf)
        self.assertIn(clf("this is a simple test"), ['0', '1'])
        self.assertIn(clf("terminal altitude"), ['0', '1'])

    def test_get_clf_starspace(self):
        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_starspace_model.tsv'))
        clf = get_clf_starspace(word_to_embedding)
        self.assertIsNotNone(clf)
        self.assertIn(clf("this is a simple test"), ['0', '1'])
        self.assertIn(clf("terminal altitude"), ['0', '1'])
