from amlearn.utils.backend import BackendContext, Backend
from amlearn.utils.basetest import AmLearnTest
from amlearn.utils.verbose import VerboseReporter


class TestBackend(AmLearnTest):
    def setUp(self):
        backend = Backend(BackendContext())
        self.verbose_reporter = \
            VerboseReporter(backend, verbose=1, total_stage=3,
                            max_verbose_mod=100)

    def test_verbose(self):
        self.verbose_reporter.init(total_epoch=10000, init_msg='Test verbose',
                                   start_epoch=0, epoch_name='Epoch', stage=1)
        for i in range(10000):
            self.verbose_reporter.update(i)
