from amlearn.utils.basetest import AmLearnTest
from amlearn.utils.backend import MLBackend, BackendContext


class TestBackend(AmLearnTest):
    def setUp(self):
        output_path = r"./amlearn_test/test_backend"
        self.backend_context = BackendContext(output_path=output_path,
                                              merge_path=True)
        self.mlbackend = MLBackend(self.backend_context)

    def test_valid_predictions_type(self):
        print(self.mlbackend.valid_predictions_type)
