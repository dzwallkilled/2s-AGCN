import logging
import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_logger(log_file=None, name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    # file handler can be added at any time
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s",
                                  datefmt='%Y-%d-%d %H:%M:%S')
    fh = None
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(sh)
    # logger.addHandler(TqdmLoggingHandler())
    if fh is not None:
        logger.addHandler(fh)
    return logger


