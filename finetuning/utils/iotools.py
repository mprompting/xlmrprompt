import os
import pickle
import json
import datetime
import torch


def list_files(root_path):
    dirs = os.listdir(root_path)
    return [os.path.join(root_path, path) for path in dirs]


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def torch_load(path):
    with open(path, "rb") as handle:
        return torch.load(handle)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def str2time(string, pattern):
    return datetime.strptime(string, pattern)
