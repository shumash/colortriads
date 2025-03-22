import tensorflow as tf
import logging
import os


def LOG_set_level(level):
    LogManager.set_level(level)

def LOG_redirect_to_file(filename, overwrite=False):
    if overwrite and os.path.isfile(filename):
        os.remove(filename)
    LogManager.redirect_to_file(filename)

def LOG_flush():
    LogManager.flush()

LOG = tf.compat.v1.logging
LOG.set_level = LOG_set_level
LOG.redirect_to_file = LOG_redirect_to_file
LOG.flush = LOG_flush

class LogManager(object):
    singleton = None

    def __init__(self, level=logging.INFO):
        self.level = level
        self.log = logging.getLogger('tensorflow')
        self.fh = None
        self.formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        for h in self.log.handlers:
            h.setFormatter(self.formatter)


    def __set_level(self, level):
        self.level = level
        self.log.setLevel(level)
        if self.fh is not None:
            self.fh.setLevel(level)


    def __redirect_to_file(self, filename):
        if self.fh is not None:
            self.log.removeHandler(self.fh)
        self.fh = logging.FileHandler(filename)
        self.fh.setLevel(self.level)
        self.fh.setFormatter(self.formatter)
        self.log.addHandler(self.fh)

    def __flush(self):
        if self.fh is not None:
            self.fh.flush()

    @staticmethod
    def set_level(level):
        LogManager.singleton.__set_level(level)

    @staticmethod
    def redirect_to_file(filename):
        LogManager.singleton.__redirect_to_file(filename)

    @staticmethod
    def flush():
        LogManager.singleton.__flush()

LogManager.singleton = LogManager()