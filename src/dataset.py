# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:56:05 2022

@author: Nikhil Khandelwal
"""

from torch import nn
from Config import config
import torch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        # self.labels = [label for label in df['level']]
        self.texts = [config.tokenizer(text, 
                               padding='max_length', max_length = config.MAX_LEN, truncation=True,
                                return_tensors="pt") for text in df['response_post']]

    # def classes(self):
    #     return self.labels

    def __len__(self):
        return len(self.texts)

    # def get_batch_labels(self, idx):
    #     # Fetch a batch of labels
    #     return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        # batch_y = self.get_batch_labels(idx)

        return batch_texts#, batch_y