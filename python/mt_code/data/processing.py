from functools import partial

from mt_code.config import Config
from datasets import load_dataset

import numpy as np
import pandas as pd
import tensorflow as tf

def process_dataset(config: Config):
    dataset = load_dataset("wmt17", "de-en")
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    return train_dataset, validation_dataset

