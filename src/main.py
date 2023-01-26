# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:43:14 2022

@author: Nikhil Khandelwal
"""

import pandas as pd
import numpy as np

import random
import os
import io

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


from dataset import Dataset
from model import BertClassifier
from Config import config

SEED = config.SEED

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if config.device == torch.device("cuda"):
    torch.cuda.manual_seed_all(SEED)

class predict_empathy:
    
    def __init__(self):
        try:
            self.model = BertClassifier()
            checkpoint = torch.load(config.output_model, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print("cannnot initilize model")
            raise e
            
    def predict(self, test_data):
        try:
            predictions=[]
            df_test = pd.read_csv(test_data)
            test = Dataset(df_test)
        
            test_dataloader = torch.utils.data.DataLoader(test, batch_size=config.batch_size)
        
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        
            if use_cuda:
        
                self.model = self.model.cuda()
        
            
            with torch.no_grad():
        
                for test_input in test_dataloader:
        
                    # test_label = test_label.to(device)
                    mask = test_input['attention_mask'].to(device)
                    input_id = test_input['input_ids'].squeeze(1).to(device)
        
                    output = self.model(input_id, mask)
                    output = output.argmax(dim=1)
                    if output==1:
                        output = 'Empathy'
                    else:
                        output = 'Not Empathy'
                    predictions.append(output)
                    
            return predictions
        
        except Exception as e:
            print(e)
            raise e
    

if __name__=="__main__":
    pred_obj = predict_empathy()
    test_data = r'data/empathybalaceddata_test.csv'
    predictions = pred_obj.predict(test_data)
    print(predictions)
        
    
