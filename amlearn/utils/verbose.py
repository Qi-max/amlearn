import time

__author__ = "Qi Wang"
__email__ = "qiwang.mse@gmail.com"


class VerboseReporter(object):

    def __init__(self, backend, total_stage=1,
                 verbose=0, max_verbose_mod=10000):
        """

        Args:
            backend:
            total_stage: total stage of whole step.
            verbose: int
                If ``verbose==1`` output is printed once in a while. If larger
                than 1 then output is printed for each update.
            max_verbose_mod: int
                max_verbose_mod
        """
        self.backend = backend
        self.verbose = verbose
        self.total_stage = total_stage
        self.max_verbose_mod_ = max_verbose_mod

    @property
    def max_verbose_mod(self):
        return self.max_verbose_mod_

    @max_verbose_mod.setter
    def max_verbose_mod(self, max_verbose_mod):
        self.max_verbose_mod_ = max_verbose_mod

    def init(self, total_epoch, start_epoch=0, init_msg=None,
             epoch_name='Epoch', stage=1):
        self.start_epoch = start_epoch
        self.total_epoch = total_epoch

        if init_msg is not None:
            self.backend.logger.info(init_msg)
        self.backend.logger.info(
            'Start stage {} of {}.'.format(stage, self.total_stage))
        verbose_header = \
            '{:>12}{:>20}{:>20}'.format(epoch_name,
                                        'Stage Remain Time',
                                        'Stage Used Time')
        self.verbose_fmt = '{epoch:>12d}{remaining_time:>20}{used_time:>20}'
        if stage > 1:
            verbose_header += '{:>20}'.format('Total Used Time')
            self.verbose_fmt += '{total_used_time:>20}'

        self.verbose_mod = 1
        self.backend.logger.info(verbose_header)

        self.stage_start_time = time.time()
        self.stage = stage
        if stage == 1:
            self.total_start_time = time.time()

    def update(self, epoch):
        """

        Args:
            epoch: int
                start by 0.

        Returns:

        """
        if (epoch + 1) % self.verbose_mod == 0:
            now_time = time.time()

            used_time = now_time - self.stage_start_time

            remaining_time = (self.total_epoch - epoch - 1) * used_time / \
                             (epoch + 1 - self.start_epoch)
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            if used_time > 60:
                used_time = '{0:.2f}m'.format(used_time / 60.0)
            else:
                used_time = '{0:.2f}s'.format(used_time)

            if self.stage > 1:
                total_used_time = now_time - self.total_start_time
                if total_used_time > 60:
                    total_used_time = '{0:.2f}m'.format(total_used_time / 60.0)
                else:
                    total_used_time = '{0:.2f}s'.format(total_used_time)

                self.backend.logger.info(self.verbose_fmt.format(
                    epoch=(epoch + 1), remaining_time=remaining_time,
                    used_time=used_time, total_used_time=total_used_time))
            else:
                self.backend.logger.info(self.verbose_fmt.format(
                    epoch=(epoch + 1), remaining_time=remaining_time,
                    used_time=used_time))

            if self.verbose == 1 and self.verbose_mod < self.max_verbose_mod_\
                    and (epoch + 1) // (self.verbose_mod * 10) > 0:
                self.verbose_mod *= 10
