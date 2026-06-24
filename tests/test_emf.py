import unittest

from recoxplainer.config import load_config
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.models import EMFModel
from recoxplainer.recommender import Recommender


class EMFTest(unittest.TestCase):

    def setUp(self) -> None:
        cfg = load_config("configs/config.yml")
        self.emf = EMFModel(**cfg.model.emf)
        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()

    def test_train_emf(self):
        self.assertTrue(self.emf.fit(self.data))
        recommender = Recommender(self.data, self.emf)
        recommender.recommend_all()
