import configparser
import os
from pathlib import Path


class ReadConfig:
    def __init__(self, file_path=None):
        if file_path:
            conf_path = file_path
        else:
            conf_path = Path(__file__).resolve().parent.as_posix() + '/config.ini'
            # conf_path = 'config.ini'

        self.cf = configparser.ConfigParser()
        self.cf.read(conf_path)

    def get_path(self, param):
        return self.cf.get('Paths', param)
