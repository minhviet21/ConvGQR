import sys

sys.path += ['../']
import torch
from torch import nn
import numpy as np
from transformers import (RobertaConfig, RobertaModel, AutoTokenizer,
                          RobertaForSequenceClassification, RobertaTokenizer)

import torch.nn.functional as F
from IPython import embed
import time

class BKAI(RobertaModel):
    def __init__(self, config):
        RobertaModel.__init__(self, config)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def query_emb(self, input_ids, attention_mask):
        model_output = super().forward(input_ids = input_ids, attention_mask = attention_mask)
        result = self.mean_pooling(model_output, attention_mask)
        return result 
    
    def doc_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)

    def forward(self, input_ids, attention_mask, wrap_pooler=False):
        return self.query_emb(input_ids, attention_mask)   
      
def load_model(model_type, model_path):
    if model_type == "ANCE_Query" or model_type == "ANCE_Passage":
        config = RobertaConfig.from_pretrained(
            model_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )
        model = BKAI.from_pretrained(model_path, config=config)
    else:
        raise ValueError
    return tokenizer, model
