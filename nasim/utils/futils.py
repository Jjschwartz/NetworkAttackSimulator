"""This module contains useful functions for handling files """
import yaml


def load_yaml(file_path):
    """Load yaml file located at file path, raises error if theres an issue loading file. """
    with open(file_path) as fin:
        content = yaml.load(fin, Loader=yaml.FullLoader)
    return content
