# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:44:57 2022

@author: Nikhil Khandelwal
"""
import torch
from transformers import BertTokenizer

class config:
    SEED = 19
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    MAX_LEN = 128
    EPOCHS = 5
    LR = 2e-6 #1e-5
    adam_epsilon = 1e-8
    tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased',do_lower_case=True)
    output_model = 'model/distilbert_empathy.pth'    
