import unittest

from recoxplainer.config import load_config
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.models.gmf_model import GMFModel
from recoxplainer.recommender import Recommender


class GMFTest(unittest.TestCase):

    def setUp(self) -> None:
        cfg = load_config("configs/config.yml")
        self.gmf = GMFModel(**cfg.model.gmf)
        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

    def test_train_gmf(self):
        self.assertTrue(self.gmf.fit(self.data))
        recommender = Recommender(self.data, self.gmf)
        recommender.recommend_all()
