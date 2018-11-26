from amlearn.featurize.featurizers.pipeline import all_featurizers
from amlearn.utils.basetest import AmLearnTest


class TestSro(AmLearnTest):
    def setUp(self):
        pass

    def test_all_featurizers(self):
        print(all_featurizers())
