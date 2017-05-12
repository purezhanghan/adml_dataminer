"""
paths for project and relevant files
"""

import os

path = os.getcwd()
PROJECT_PATH = path + '/data'

TRAIN_PATH = os.path.join(PROJECT_PATH, "train.csv")
TEST_PATH = os.path.join(PROJECT_PATH, "test.csv")
SUBMISSION_PATH = os.path.join(PROJECT_PATH, "submission.csv")

TRAIN = os.path.join(PROJECT_PATH, "train.csv")
TEST = os.path.join(PROJECT_PATH, "test.csv")

TRAIN_NORMAL = os.path.join(PROJECT_PATH, "train_normal.csv")
TEST_NORMAL = os.path.join(PROJECT_PATH, "test_normal.csv")
