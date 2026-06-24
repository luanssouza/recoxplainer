import unittest

from recoxplainer.config import load_config
from recoxplainer.data_reader.data_reader import DataReader
from recoxplainer.models.autoencoder_model import ExplAutoencoderTorch
from recoxplainer.recommender import Recommender


class GMFTest(unittest.TestCase):

    def setUp(self) -> None:
        cfg = load_config("configs/config.yml")
        self.autoencoder = ExplAutoencoderTorch(**cfg.model.autoencoder)
        self.data = DataReader(**cfg.testdata)
        self.data.make_consecutive_ids_in_dataset()
        self.data.binarize()

    def test_train_autoencoder(self):
        self.assertTrue(self.autoencoder.fit(self.data))
        recommender = Recommender(self.data, self.autoencoder)
        recommender.recommend_all()
