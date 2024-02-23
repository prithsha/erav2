import logging
import constant
from logging.handlers import RotatingFileHandler


# create file handler which logs even debug messages. 10 MB (10485760) File handler
fh = RotatingFileHandler(constant.LOG_FILE, maxBytes=10485760, backupCount=5)
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s\t%(levelname)s\t%(name)s\t%(funcName)s\t%(process)d\t%(thread)d\t%(message)s')

ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger = logging.getLogger(constant.APP_NAME)
logger.setLevel(logging.INFO)
logger.addHandler(ch)
logger.addHandler(fh)
logger.debug('Logger printing Debug configuration.')
logger.info('Logger printing Info configuration')


