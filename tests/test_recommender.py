import unittest

from recoxplainer.config import load_config
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.models.als_model import ALS
from recoxplainer.recommender import Recommender


class RecommenderImplicitTest(unittest.TestCase):

    def setUp(self):
        cfg = load_config("configs/config.yml")
        self.als = ALS(**cfg.model.als)

        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

        self.als.fit(self.data)

    def test_train_recommend_als(self):
        recommender = Recommender(self.data, self.als)
        recommender.recommend_all()
