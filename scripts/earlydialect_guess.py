#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 15:22:31 2022

@author: vanikanjirangat
"""


import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
le = LabelEncoder()
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import time
import datetime
from transformers import AutoModel, AutoTokenizer, AutoConfig,AutoModelForSequenceClassification

device_name = 'cuda'

# specify the data path
if GDI:
    
    root_dir='./data/GDI/'
elif ILI:
    root_dir='./data/ILI/'
elif AOC:
    root_dir='./data/AOC/'
elif ADI:
    root_dir='./data/ADI/'
    
m=data_source.split("/")[2]

"""Use the path of the Dataset that needs to the analyzed"""

GDI_2017=0
GDI_2018=1
GDI_2019=0
ILI=0


if GDI_2017:
  input_dir=os.path.join(root_dir,'gdi-vardial-2017')
elif GDI_2018:
  input_dir=os.path.join(root_dir,'gdi-vardial-2018')
elif GDI_2019:
  input_dir=os.path.join(root_dir,'gdi-vardial-2019')
elif ILI:
  input_dir=os.path.join(root_dir,'IndoAryan')
elif AOC:
  input_dir=os.path.join(root_dir,'aoc_data')
elif ADI:
    input_dir=os.path.join(root_dir,'Vardial_ADI')



"""Fine Tuning"""

GDI=1
ILI=0
AOC=0


"""Here we explain the fine-tuning with the best pre-trained model that we had obtained."""


class Model:
    def __init__(self,path):
        # self.args = args
        self.path=path
        self.MAX_LEN=128
        # self.tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        if GDI:
            
            self.tokenizer=BertTokenizer.from_pretrained('bert-base-cased')
        elif ILI:
            num_labels=5
            # self.config = AutoConfig.from_pretrained('ai4bharat/indic-bert',num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained('neuralspace-reverie/indic-transformers-hi-bert')
            self.config = AutoConfig.from_pretrained('neuralspace-reverie/indic-transformers-hi-bert',num_labels=num_labels)
        
        elif AOC:
            self.tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabert')
            num_labels=5
            self.config = AutoConfig.from_pretrained('aubmindlab/bert-base-arabert',num_labels=num_labels)
        elif ADI:
            self.tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabert')
            num_labels=4
            self.config = AutoConfig.from_pretrained('aubmindlab/bert-base-arabert',num_labels=num_labels)
      
        # if not os.path.isdir(self.opath):
        #     os.makedirs(self.opath)
            
            
    def extract_data(self,name1,name2=None,XY=None):
        
        if GDI:
            file =self.path+name1
            df = pd.read_csv(file, delimiter='\t', header=None, names=['sentence','label'])
            df.replace(np.nan,'NIL', inplace=True)
            
            sentences = df.sentence.values
            
            labels = df.label.values
            if XY==0:
              sentences=[x for i,x in enumerate(sentences) if labels[i]!='XY']
              labels=[x for x in labels if x!='XY']
        elif ILI:
            file =self.path+name1
            df = pd.read_csv(file, delimiter='\t', header=None, names=['sentence','label'])
            df.replace(np.nan,'NIL', inplace=True)
          
            sentences = df.sentence.values
          
            labels = df.label.values
        elif AOC:
            file1 =self.path+name1
            df = pd.read_csv(file1)
            df.replace(np.nan,'NIL', inplace=True)
        
            sentences = df["text"].values
            labels=df["label"].values
        elif ADI:
            file1 =self.path+name1
            df = pd.read_csv(file1, delimiter='\t', header=None, names=["text_id","text"])
            df.replace(np.nan,'NIL', inplace=True)
        
            sentences = df["text"].values
            file2 =self.path+name2
            df = pd.read_csv(file2, delimiter='\t', header=None, names=["text_id","labels"])
            df.replace(np.nan,'NIL', inplace=True)
            labels = df["labels"].values
        
        return (sentences,labels)
    
    def process_inputs(self,sentences,labels):
      sentences= [self.tokenizer.encode_plus(sent,add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]
      # sentence_idx = np.linspace(0,len(sentences), len(sentences),False)
      # torch_idx = torch.tensor(sentence_idx)
      tags_vals = list(labels)
      le.fit(labels)
      le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      labels=le.fit_transform(labels)
      
      print(le_name_mapping)
      # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
      input_ids = [inputs["input_ids"] for inputs in sentences]

      # Pad our input tokens
      input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")
      attention_masks = []

      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask= [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

        
      # token_type_ids=[inputs["token_type_ids"] for inputs in sentences]
      # token_type_ids=pad_sequences(token_type_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")

      inputs, labels = input_ids, labels
      masks,_= attention_masks, input_ids
      # Convert all of our data into torch tensors, the required datatype for our model

      self.inputs = torch.tensor(inputs).to(torch.int64)
      # validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
      self.labels = torch.tensor(labels).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.masks = torch.tensor(masks).to(torch.int64)
      # validation_masks = torch.tensor(validation_masks).to(torch.int64)
      # self.types=torch.tensor(types).to(torch.int64)
      self.data = TensorDataset(self.inputs,self.masks, self.labels)
      self.sampler = RandomSampler(self.data)
      self.dataloader = DataLoader(self.data, sampler=self.sampler, batch_size=32)

      # return (self.inputs,self.labels,self.masks,self.types)
    def process_inputs_test(self,sentences,labels,act_ids,batch_size=1):
      sentences= [self.tokenizer.encode_plus(sent,add_special_tokens=True, max_length=self.MAX_LEN,truncation='longest_first') for i,sent in enumerate(sentences)]
      sentence_idx = np.linspace(0,len(sentences), len(sentences),False)
      torch_idx = torch.tensor(sentence_idx)
      tags_vals = list(labels)
      le.fit(labels)
      le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
      labels=le.fit_transform(labels)
      
      print(le_name_mapping)
      # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
      input_ids = [inputs["input_ids"] for inputs in sentences]

      # Pad our input tokens
      input_ids = pad_sequences(input_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")
      attention_masks = []

      # Create a mask of 1s for each token followed by 0s for padding
      for seq in input_ids:
        seq_mask= [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

        
      # token_type_ids=[inputs["token_type_ids"] for inputs in sentences]
      # token_type_ids=pad_sequences(token_type_ids, maxlen=self.MAX_LEN,truncating="post", padding="post")

      inputs, labels = input_ids, labels
      masks,_= attention_masks, input_ids
      # Convert all of our data into torch tensors, the required datatype for our model

      self.inputs = torch.tensor(inputs).to(torch.int64)
      # validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
      self.labels = torch.tensor(labels).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.act_ids = torch.tensor(act_ids).to(torch.int64)
      # validation_labels = torch.tensor(validation_labels).to(torch.int64)
      self.masks = torch.tensor(masks).to(torch.int64)
      self.torch_idx = torch.tensor(sentence_idx).to(torch.int64)
      self.data = TensorDataset(self.inputs,self.masks, self.labels,self.torch_idx,self.act_ids)
      self.sampler = RandomSampler(self.data)
      self.dataloader = DataLoader(self.data, sampler=self.sampler, batch_size=batch_size)

    def train_save_load(self,train=1,retrain=0,label_smoothing = -1,XY=None):
      
      
      WEIGHTS_NAME = "%s_model.bin"%(m)
      

      OUTPUT_DIR = input_dir
      output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
      if retrain!=1:

        if XY==0:
          # self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=4)
          self.model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
        else:
          # self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=5)
          self.model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
      else:
        # self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=4)
        self.model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
        state_dict = torch.load(output_model_file)
        self.model.load_state_dict(state_dict)
        WEIGHTS_NAME = "%s_model_retrain.bin"%(m)

        OUTPUT_DIR = input_dir
        output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
      self.model.cuda()
      param_optimizer = list(self.model.named_parameters())
      no_decay = ['bias', 'gamma', 'beta']
      optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                                                                                                                                    'weight_decay_rate': 0.0}]
      optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5)
      
      
      train_loss_set = []
      out_train={}
      
      true=[]
      logits_all=[]
      output_dicts = []
      batch_size=32
      epochs = 4
      import time
      start_time = time.time()
      if train==1:
        for _ in trange(epochs, desc="Epoch"):
          # Trainin
          # Set our model to training mode (as opposed to evaluation mode
          self.model.train()
          # Tracking variables
          tr_loss = 0
          nb_tr_examples, nb_tr_steps = 0, 0
          # Train the data for one epoch
          for step, batch in enumerate(self.dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids,b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            # loss = model(b_input_ids, token_type_ids=b_types, attention_mask=b_input_mask, labels=b_labels)
            loss,logits= self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            if label_smoothing == -1:
              logits=logits
            else:
              criterion = LabelSmoothingLoss(label_smoothing)
              loss=criterion(logits,b_labels)
            


      
            train_loss_set.append(loss.item())    
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
          print("Train loss: {}".format(tr_loss/nb_tr_steps))
        print("--- %s seconds ---" % (time.time() - start_time)) 
        torch.save(self.model.state_dict(), output_model_file)
        

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

        output_dir = OUTPUT_DIR+'/model_save/'

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        # logits1 = [item for sublist in logits_all for item in sublist]
        # true_labels = [item for sublist in true for item in sublist]
        # if 'logits' not in out_train.keys():
        #   out_train['logits']=logits1
        # else:
        #   out_train['logits'].append(logits1)

        # if 'true' not in out_train.keys():
        #   out_train['true']=true_labels
        # else:
        #   out_train['true'].append(true_labels)
        
      else:
        state_dict = torch.load(output_model_file)
        self.model.load_state_dict(state_dict) 
      return output_dicts

    def eval(self,label_smoothing = -1):
      batch_size=32
      eval_loss = 0
      # Put model in evaluation mod
      self.model.eval()
      # Tracking variables 
      self.predictions , self.true_labels = [], []
      output_dicts=[]
    
      
      for batch in self.dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids,b_input_mask, b_labels = batch
        # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          if label_smoothing == -1:
            logits=outputs[0]
            loss=outputs[1]
          else:
            criterion = LabelSmoothingLoss(label_smoothing)
            loss=criterion(logits,b_labels)
          
          eval_loss += loss.item()
          self.dataloader.set_description(f'eval loss = {(eval_loss / i):.6f}')
      return eval_loss / len(self.dataloader)
    
    def simple_test(self):
      batch_size=32
      # Put model in evaluation mod
      self.model.eval()
      # Tracking variables 
      self.predictions , self.true_labels = [], []
      output_dicts=[]
      
      
      for batch in self.dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids,b_input_mask, b_labels = batch
        # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          logits=outputs[0]
          for j in range(logits.size(0)):
            probs = F.softmax(logits[j], -1)
            output_dict = {
                # 'index': batch_size * i + j,
                'true': b_labels[j].cpu().numpy().tolist(),
                'pred': logits[j].argmax().item(),
                'conf': probs.max().item(),
                'logits': logits[j].cpu().numpy().tolist(),
                'probs': probs.cpu().numpy().tolist(),
            }
            output_dicts.append(output_dict)
      y_true = [output_dict['true'] for output_dict in output_dicts]
      y_pred = [output_dict['pred'] for output_dict in output_dicts]
      y_conf = [output_dict['conf'] for output_dict in output_dicts]

      accuracy = accuracy_score(y_true, y_pred) * 100.
      f1 = f1_score(y_true, y_pred, average='macro') * 100.
      confidence = np.mean(y_conf) * 100.

      results_dict = {
          'accuracy': accuracy_score(y_true, y_pred) * 100.,
          'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
          'confidence': np.mean(y_conf) * 100.,
      }
      print(results_dict)
      print(classification_report(y_true,y_pred))
      print(confusion_matrix(y_true,y_pred))
      return output_dicts

    def test(self,sents):
      batch_size=32
      # Put model in evaluation mod
      self.model.eval()
      # Tracking variables 
      self.predictions , self.true_labels,self.sents,self.actsents = [], [],[],[]
      output_dicts=[]
    
      
      for batch in self.dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids,b_input_mask, b_labels,b_index,b_ids = batch
        # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          logits=outputs[0]
          

          for j in range(logits.size(0)):
            probs = F.softmax(logits[j], -1)
            output_dict = {
                # 'index': batch_size * i + j,
                'true': b_labels[j].cpu().numpy().tolist(),
                'pred': logits[j].argmax().item(),
                'conf': probs.max().item(),
                'logits': logits[j].cpu().numpy().tolist(),
                'probs': probs.cpu().numpy().tolist(),
                'actsent_ids'   : b_ids[j].cpu().numpy().tolist(),
                'sent_ids'   : b_index[j].cpu().numpy().tolist(),
                'sents' : sents[b_index[j]]
            }
            output_dicts.append(output_dict)
      y_true = [output_dict['true'] for output_dict in output_dicts]
      y_pred = [output_dict['pred'] for output_dict in output_dicts]
      y_conf = [output_dict['conf'] for output_dict in output_dicts]

      accuracy = accuracy_score(y_true, y_pred) * 100.
      f1 = f1_score(y_true, y_pred, average='macro') * 100.
      confidence = np.mean(y_conf) * 100.

      results_dict = {
          'accuracy': accuracy_score(y_true, y_pred) * 100.,
          'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
          'confidence': np.mean(y_conf) * 100.,
      }
      print(results_dict)
      print(classification_report(y_true,y_pred))
      print(confusion_matrix(y_true,y_pred))
      return output_dicts

    def test_inc(self,batch_size=1):
      #batch_size : should be no. of fragments after incremental processing
      # Put model in evaluation mod
      self.model.eval()
      # Tracking variables 
      self.predictions , self.true_labels = [], []
      output_dicts=[]
      
      
      for batch in self.dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids,b_input_mask, b_labels = batch
        # # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
          logits=outputs[0]
          for j in range(logits.size(0)):
            probs = F.softmax(logits[j], -1)
            output_dict = {
                # 'index': batch_size * i + j,
                'true': b_labels[j].cpu().numpy().tolist(),
                'pred': logits[j].argmax().item(),
                'conf': probs.max().item(),
                'logits': logits[j].cpu().numpy().tolist(),
                'probs': probs.cpu().numpy().tolist(),
            }
            output_dicts.append(output_dict)
      y_true = [output_dict['true'] for output_dict in output_dicts]
      y_pred = [output_dict['pred'] for output_dict in output_dicts]
      y_conf = [output_dict['conf'] for output_dict in output_dicts]

      accuracy = accuracy_score(y_true, y_pred) * 100.
      f1 = f1_score(y_true, y_pred, average='macro') * 100.
      confidence = np.mean(y_conf) * 100.

      results_dict = {
          'accuracy': accuracy_score(y_true, y_pred) * 100.,
          'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
          'confidence': np.mean(y_conf) * 100.,
      }
      print(results_dict)
      print(classification_report(y_true,y_pred))
      print(confusion_matrix(y_true,y_pred))
      return output_dicts



path=input_dir

m = Model(path)
XY=0

if GDI:

    sentences_train,labels_train=m.extract_data('/train.txt',name2=None,XY=0)
    
    
    
    sentences_dev,labels_dev=m.extract_data('/dev.txt',name2=None,XY=0)
    
    
    sentences_test,labels_test=m.extract_data('/gold.txt',name2=None,XY=1)
elif ILI:
    sentences_train,labels_train=m.extract_data('/train.txt',name2=None,XY=0)
    sentences_dev,labels_dev=m.extract_data('/dev.txt',name2=None,XY=0)
    sentences_test,labels_test=m.extract_data('/gold.txt',name2=None,XY=0)
elif AOC:
    sentences_train,labels_train=m.extract_data("/train/MultiTrain.Shuffled.csv",XY=0)
    sentences_dev,labels_dev=m.extract_data("/dev/MultiDev.csv",XY=0)
    sentences_test,labels_test=m.extract_data("/test/MultiTest.csv",XY=0)
elif ADI:
    sentences_train,labels_train=m.extract_data("/train/train.words","/train/train.labels",XY=0)

    sentences_dev,labels_dev=m.extract_data("/dev/dev.words","/dev/dev.labels",XY=0)
    sentences_test,labels_test=m.extract_data("/test/test.words","/test/test.labels",XY=0)



#TV=1,#use both training and validation set
if TV:
  sentences_train=np.append(sentences_train,sentences_dev)
  
  labels_train=np.append(labels_train,labels_dev)

 


m.process_inputs(sentences_train,labels_train)
# train=0: no training, train=1: do training

m.train_save_load(train=1,retrain=0,label_smoothing = -1,XY=0)#bert training 4 epochs

m.process_inputs(sentences_test,labels_test)

act_out_test=m.simple_test()


# FOR THE INCREMENTAL ANALYSIS

def synthesize(s):

  m=[]
  x=[]
  for t in s.split(' '):
    m.append(t)
    x.append(' '.join(m))
  return x
from itertools import chain
sentences_test,labels_test=m.extract_data('/gold.txt',XY=0)
extract_sents=[]
labels_sents=[]
act_ids=[]
for i,s in enumerate(sentences_test):
  extract_sents.append(synthesize(s))
  labels_sents.append([labels_test[i]]*len(extract_sents[i]))
  act_ids.append([i]*len(extract_sents[i]))
extract_sents=list(chain(*extract_sents))
labels_sents=list(chain(*labels_sents))
act_ids=list(chain(*act_ids))

#sentences_test[0:10]

print(len(extract_sents),len(labels_sents),len(act_ids))

m.process_inputs_test(extract_sents,labels_sents,act_ids)


out_test=m.test(extract_sents)


sentences_dev,labels_dev=m.extract_data('/dev.txt',XY=0)
extract_sents=[]
labels_sents=[]
act_ids=[]
for i,s in enumerate(sentences_dev):
  extract_sents.append(synthesize(s))
  labels_sents.append([labels_dev[i]]*len(extract_sents[i]))
  act_ids.append([i]*len(extract_sents[i]))
extract_sents=list(chain(*extract_sents))
labels_sents=list(chain(*labels_sents))
act_ids=list(chain(*act_ids))

m.process_inputs_test(extract_sents,labels_sents,act_ids)

#incremetal processed dev set
out_dev=m.test(extract_sents)


sentences_train,labels_train=m.extract_data('/train.txt',XY=0)
extract_sents=[]
labels_sents=[]
act_ids=[]
for i,s in enumerate(sentences_train):
  extract_sents.append(synthesize(s))
  labels_sents.append([labels_train[i]]*len(extract_sents[i]))
  act_ids.append([i]*len(extract_sents[i]))
extract_sents=list(chain(*extract_sents))
labels_sents=list(chain(*labels_sents))
act_ids=list(chain(*act_ids))

m.process_inputs_test(extract_sents,labels_sents,act_ids)

#incremetal processed train set
out_train=m.test(extract_sents)

train_path=os.path.join(input_dir,'out_train.json')

import json
with open(train_path, 'w+') as f:
  for i, output_dict in enumerate(out_train):
      output_dict_str = json.dumps(output_dict)
      f.write(f'{output_dict_str}\n')

'''
At which particular fragment lengths maximum confidence was noted.
Is the confidence maximum always at the original sentence length/raw_max_lenght?
Is there any average fragment length where maximum confidence/probability is witnessed?
This may/may not be true label.//
Is there any average fragment length where the best heuristic is satisfied?
'''

# import numpy as np
# np.save('out_test.npy', out_test)

# test_path=os.path.join(input_dir,'out_test_mbertG.json')
# test_path=os.path.join(input_dir,'out_test_VbertG.json')
# test_path=os.path.join(input_dir,'out_test_VbertG_nw.json')

# define the path for saving the outputs

test_path=os.path.join(input_dir,'out_test_%sfinal.json'%(m))

import json
with open(test_path, 'w+') as f:
  for i, output_dict in enumerate(out_test):
      output_dict_str = json.dumps(output_dict)
      f.write(f'{output_dict_str}\n')

# dev_path=os.path.join(input_dir,'out_dev_mbertG.json')
# dev_path=os.path.join(input_dir,'out_dev_VbertG.json')
# dev_path=os.path.join(input_dir,'out_dev_VbertG_nw.json')

dev_path=os.path.join(input_dir,'out_dev_%sfinal.json'%(m))

with open(dev_path, 'w+') as f:
  for i, output_dict in enumerate(out_dev1):
      output_dict_str = json.dumps(output_dict)
      f.write(f'{output_dict_str}\n')

# dev_path_act=os.path.join(input_dir,'out_dev_VbertG_nw1.json')
# with open(dev_path_act, 'w+') as f:
#   for i, output_dict in enumerate(out_dev1):
#       output_dict_str = json.dumps(output_dict)
#       f.write(f'{output_dict_str}\n')

# np.save(os.path.join(input_dir,'out_test1.npy'), out_test)

# import json
# with open(input_dir+'out_test.csv', 'w') as f:
#   for i, output_dict in enumerate(out_train):
#     output_dict_str = json.dumps(output_dict)
#     f.write(f'{output_dict_str}\n')

# t1=np.load(os.path.join(input_dir,'out_train1.npy'),allow_pickle='TRUE').item()

# t2=np.load(os.path.join(input_dir,'out_test1.npy'),allow_pickle='TRUE').item()

#FOR TEMPERATURE SCALING

def load_output(path):
    """Loads output file, wraps elements in tensor."""

    with open(path) as f:
        elems = [json.loads(l.rstrip()) for l in f]
        for elem in elems:
            elem['true'] = torch.tensor(elem['true']).long()
            elem['logits'] = torch.tensor(elem['logits']).float()
            # elem['actsent_ids']= elem['actsent_ids']
            # elem['sent_ids']= elem['sent_ids']
            # elem['sents']= elem['sents']

        return elems

def get_bucket_scores(y_score):
    """
    Organizes real-valued posterior probabilities into buckets.
    For example, if we have 10 buckets, the probabilities 0.0, 0.1,
    0.2 are placed into buckets 0 (0.0 <= p < 0.1), 1 (0.1 <= p < 0.2),
    and 2 (0.2 <= p < 0.3), respectively.
    """
    buckets=10
    bucket_values = [[] for _ in range(buckets)]
    bucket_indices = [[] for _ in range(buckets)]
    for i, score in enumerate(y_score):
        for j in range(buckets):
            if score < float((j + 1) / buckets):
                break
        bucket_values[j].append(score)
        bucket_indices[j].append(i)
    return (bucket_values, bucket_indices)


def get_bucket_confidence(bucket_values):
    """
    Computes average confidence for each bucket. If a bucket does
    not have predictions, returns -1.
    """

    return [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in bucket_values
    ]


def get_bucket_accuracy(bucket_values, y_true, y_pred):
    """
    Computes accuracy for each bucket. If a bucket does
    not have predictions, returns -1.
    """

    per_bucket_correct = [
        [int(y_true[i] == y_pred[i]) for i in bucket]
        for bucket in bucket_values
    ]
    return [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in per_bucket_correct
    ]


def calculate_error(n_samples, bucket_values, bucket_confidence, bucket_accuracy):
    """
    Computes several metrics used to measure calibration error:
        - Expected Calibration Error (ECE): \sum_k (b_k / n) |acc(k) - conf(k)|
        - Maximum Calibration Error (MCE): max_k |acc(k) - conf(k)|
        - Total Calibration Error (TCE): \sum_k |acc(k) - conf(k)|
    """

    assert len(bucket_values) == len(bucket_confidence) == len(bucket_accuracy)
    assert sum(map(len, bucket_values)) == n_samples

    expected_error, max_error, total_error = 0., 0., 0.
    for (bucket, accuracy, confidence) in zip(
        bucket_values, bucket_accuracy, bucket_confidence
    ):
        if len(bucket) > 0:
            delta = abs(accuracy - confidence)
            expected_error += (len(bucket) / n_samples) * delta
            max_error = max(max_error, delta)
            total_error += delta
    return (expected_error * 100., max_error * 100., total_error * 100.)


def create_one_hot(n_classes):
    """Creates one-hot label tensor."""
    label_smoothing=0.0
    smoothing_value = label_smoothing / (n_classes - 1)
    one_hot = torch.full((n_classes,), smoothing_value).float()
    return one_hot


def cross_entropy(output, target, n_classes):
    """
    Computes cross-entropy with KL divergence from predicted distribution
    and true distribution, specifically, the predicted log probability
    vector and the true one-hot label vector.
    """
    label_smoothing=0.0
    model_prob = create_one_hot(n_classes)
    model_prob[target] = 1. - label_smoothing
    return F.kl_div(output, model_prob, reduction='sum').item()

'''
At test time, we produced a set of first-level predictions based on the best model tuned for the task on the train- ing/development set, 
and retrained the model af- ter adding the predictions with high-confidence to the training set. 
In our case, predictions with high-confidence means the test instances that are farther than a threshold — 
in this case, 0.50 — from the decision boundary for binary classification, and 
the instances that are claimed by only one of the one-vs-rest classifiers for the multi-class problems. 
Intuitively, this is useful for the adapta- tion subtasks of MRC, 
and in case the distribution of the test instances diverge from the distribution in the training/development sets.
'''

'''
We use the dev set to get the best samples using the most feasible hueristics. A kind of strict adaptation. To see if model
can learn some intutions/ patterns to cope up with the heuristics in test case.
'''

# Load and optimize T for tempertaure scaling from dev sets using the trained model logits
import json
elems = load_output(dev_path)
n_classes = len(elems[0]['logits'])
print('no. of classes',n_classes)

best_nll = float('inf')
best_temperature = -1

temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))

for temp in tqdm(temp_values, leave=False, desc='training'):
    nll = np.mean(
        [
            cross_entropy(
                F.log_softmax(elem['logits'] / temp, 0), elem['true'], n_classes
            )
            for elem in elems
        ]
    )
    if nll < best_nll:
        best_nll = nll
        best_temp = temp

temperature = best_temp

output_dict = {'temperature': best_temp}

print()
print('*** training ***')
for k, v in output_dict.items():
    print(f'{k} = {v}')

# # ideal temperature for dev
# print(temperature)

# elems = load_output(dev_path_act)
# n_classes = len(elems[0]['logits'])
# print('no. of classes',n_classes)

# best_nll = float('inf')
# best_temperature = -1

# temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))

# for temp in tqdm(temp_values, leave=False, desc='training'):
#     nll = np.mean(
#         [
#             cross_entropy(
#                 F.log_softmax(elem['logits'] / temp, 0), elem['true'], n_classes
#             )
#             for elem in elems
#         ]
#     )
#     if nll < best_nll:
#         best_nll = nll
#         best_temp = temp

# temperature = best_temp

# output_dict = {'temperature': best_temp}

# print()
# print('*** training ***')
# for k, v in output_dict.items():
#     print(f'{k} = {v}')

# temperature

# #3.44 earlier

ideal_temperature=temperature

# With test set_ without Temperature Scaling
import json
def compute(test_path,temperature=1.0):
  elems = load_output(test_path)
  n_classes = len(elems[0]['logits'])

  labels = [elem['true'] for elem in elems]
  preds = [elem['pred'] for elem in elems]
  act_ids= [elem['actsent_ids'] for elem in elems]
  ids= [elem['sent_ids'] for elem in elems]
  sents=[elem['sents'] for elem in elems]
  print(len(labels),len(preds),len(act_ids))
  
  log_probs = [F.log_softmax(elem['logits'] / temperature, 0) for elem in elems]

  confs = [prob.exp().max().item() for prob in log_probs]
  confs_all=[prob.exp() for prob in log_probs]
  # confs_all=[float(y)  for y in x for x in confs_all]
  print(confs_all[0])
  

  log_probs_act = [F.log_softmax(elem['logits'], 0) for elem in elems]

  confs_act = [prob.exp().max().item() for prob in log_probs_act]
  confs_act_all=[prob.exp() for prob in log_probs_act]
  
  val_dict={'ids':ids,'act_ids': act_ids,'sents':sents,'labels':labels,'preds':preds,'probs_calib_pred':confs,'calibrated':confs_all,'act_probs': confs_act
            ,'probs':confs_act_all}

  nll = [
      cross_entropy(log_prob, label, n_classes)
      for log_prob, label in zip(log_probs, labels)
  ]

  bucket_values, bucket_indices = get_bucket_scores(confs)
  bucket_confidence = get_bucket_confidence(bucket_values)
  bucket_accuracy = get_bucket_accuracy(bucket_indices, labels, preds)

  accuracy = accuracy_score(labels, preds) * 100.
  avg_conf = np.mean(confs) * 100.
  avg_nll = np.mean(nll)
  expected_error, max_error, total_error = calculate_error(
      len(elems), bucket_values, bucket_confidence, bucket_accuracy
  )

  output_dict = {
      'accuracy': accuracy,
      'confidence': avg_conf,
      'temperature': temperature,
      'neg log likelihood': avg_nll,
      'expected error': expected_error,
      'max error': max_error,
      'total error': total_error,
  }

  print()
  print('*** evaluating ***')
  for k, v in output_dict.items():
      print(f'{k} = {v}')
  return val_dict,bucket_accuracy,bucket_confidence,log_probs

v1,bucket_a,bucket_c,probs=compute(test_path,temperature=ideal_temperature)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(10, 6))
# x=bucket_c
# ax.plot(bucket_c, 
#         bucket_a)
# ax.plot(x,x)

# v,bucket_a,bucket_c,probs=compute(test_path,temperature=ideal_temperature)
# fig, ax = plt.subplots(figsize=(10, 6))
# x=bucket_c
# ax.plot(bucket_c, 
#         bucket_a)
# ax.plot(x,x)

#v.keys()



df=pd.DataFrame.from_dict(v1)#temperature=1
df['act_idx']=df['act_ids'].apply(lambda x: int(x))
df['labels']=df['labels'].apply(lambda x: int(x))
df['length']=df['sents'].apply(lambda x:len(x.split(' ')))
df['raw_min_length'] = df.groupby('act_idx')['length'].transform('min')
df['raw_max_length'] = df.groupby('act_idx')['length'].transform('max')
df['equality']=(df['labels']==df['preds'])
df.to_csv(input_dir+'/dfX_%s_withcalibration.csv'%(m))

df=pd.read_csv(input_dir+'/dfX_%s_withcalibration.csv'%(m))
df=df.sort_values(['act_idx','length'])
dfX=df

df=pd.read_csv(input_dir+'/dfX_GDI2018_Vbert.csv')

df=df.sort_values(['act_idx','length'])
dfX=df

"""Input Shortening Criteria"""

dfX['p1'] = (dfX['probs_calib_pred'] > dfX['probs_calib_pred'].shift())#prob(current)>prob(previous)
dfX['p2'] = (dfX['probs_calib_pred'] < dfX['probs_calib_pred'].shift())#prob(current)<prob(previous)
dfX['p3'] = (dfX['probs_calib_pred'] < dfX['probs_calib_pred'].shift(-1))#prob(current)<prob(next)
dfX['p4'] = (dfX['probs_calib_pred'] > dfX['probs_calib_pred'].shift(-1))#prob(current)>prob(next)
dfX['l1'] = (dfX['preds'] == dfX['preds'].shift())#check label consistency, predicted_label(current)=predicted_label(previous)
dfX['l2'] = (dfX['preds'] == dfX['preds'].shift(-1))#check label consistency, predicted_label(current)=predicted_label(next)
dfX['n1']=((dfX['p2'].shift()==True) & (dfX['p3'].shift()==True))#check if previous fragment satisfy this condition
dfX['n2']=((dfX['p1'].shift()==True) & (dfX['p3'].shift()==True))

"""Experiment with different criteria"""

#ITERATE through the groups and apply heuristics
dfX=dfX[dfX['length']>=4]
cols=dfX.columns
df_new = pd.DataFrame(columns=cols)
df_h=pd.DataFrame(columns=cols)
df_nh=pd.DataFrame(columns=cols)
id_group=dfX.groupby(['act_idx'])
i=0
h=0
nh=0
r=0
for g_idx, group in id_group:
  # print(group)
  # print(group['p1'].any())
  f=0
  if (group['p4'].any()):
    r+=1
    for r_idx, row in group.iterrows():
      if (row['p4']==True):
        df_new.loc[i]=row
        df_h.loc[h]=row
        h+=1
        i+=1
        f=1# the flag is set if atleast one row in the group satisy the joint heuristic

  
    if f!=1:
      for r_idx, row in group.iterrows():
        # if row['length']==row['raw_max_length']:
        df_new.loc[i]=row
        i+=1
        df_nh.loc[nh]=row
        nh+=1
  else:
    for r_idx, row in group.iterrows():
      # if row['length']==row['raw_max_length']:
      df_new.loc[i]=row
      i+=1
      df_nh.loc[nh]=row
      nh+=1

print("No. of instances that satisfy the Heuristics",r)

df_h=df_h.groupby('act_idx').nth(0)
print("No. of instances that satisfy the Heuristics",len(df_h))

df_h=df_h.groupby('act_idx').first().reset_index()
print("No. of instances that satisfy the Heuristics (Taking the first fragment)",len(df_h))
df_h2=df_h.groupby('act_idx').nth(1)
print("No. of instances that satisfy the Heuristics by taking second fragment",len(df_h2))
df_h3=df_h.groupby('act_idx').nth(2)
print("No. of instances that satisfy the Heuristics by taking third fragment",len(df_h3))
# df_h.to_csv(input_dir+'heuristicp4.csv')
d2=df_h[df_h.equality==True]
print('Total no. of True Predictions satisying the heuristic',len(d2))
# d2.to_csv(input_dir+'heuristicp4_TP.csv')
d21=d2[d2.length==d2.raw_max_length]
print('Total no. of True Predictions satisying the heuristic and of raw_max_length',len(d21))

df_nh=df_nh[df_nh.length==df_nh.raw_max_length]
print("No. of instances that doesn't satisfy the Heuristics",len(df_nh))
d3=df_nh[df_nh.equality==True]
print('Total no. of True Predictions does not satisying the heuristic (taking raw_max_length)',len(d3))
print("Total",len(d2)+len(d3))

#ITERATE through the groups and apply heuristics
dfX=dfX[dfX['length']>=4]
cols=dfX.columns
df_new = pd.DataFrame(columns=cols)
df_h=pd.DataFrame(columns=cols)
df_nh=pd.DataFrame(columns=cols)
id_group=dfX.groupby(['act_idx'])
i=0
h=0
nh=0
r=0
for g_idx, group in id_group:
  # print(group)
  # print(group['p1'].any())
  f=0
  if (group['p4'].any() & group['l1'].any()):
    r+=1
    for r_idx, row in group.iterrows():
      if (row['p4']==True & row['l1']==True):
        df_new.loc[i]=row
        df_h.loc[h]=row
        h+=1
        i+=1
        f=1# the flag is set if atleast one row in the group satisy the joint heuristic

  
  if f!=1:
    for r_idx, row in group.iterrows():
      # if row['length']==row['raw_max_length']:
      df_new.loc[i]=row
      i+=1
      df_nh.loc[nh]=row
      nh+=1

print("No. of instances that satisfy the Heuristics",r)

df_h=df_h.groupby('act_idx').nth(0)
print("No. of instances that satisfy the Heuristics",len(df_h))
df_h=df_h.groupby('act_idx').first().reset_index()
print("No. of instances that satisfy the Heuristics (Taking the first fragment)",len(df_h))
# df_h.to_csv(input_dir+'heuristicp4.csv')
d2=df_h[df_h.equality==True]
print('Total no. of True Positives satisying the heuristic',len(d2))
# d2.to_csv(input_dir+'heuristicp4_TP.csv')
d21=d2[d2.length==d2.raw_max_length]
print('Total no. of True Positives satisying the heuristic and of raw_max_length',len(d21))

df_nh=df_nh[df_nh.length==df_nh.raw_max_length]
print("No. of instances that doesn't satisfy the Heuristics",len(df_nh))
d3=df_nh[df_nh.equality==True]
print('Total no. of True Positives does not satisying the heuristic (taking raw_max_length)',len(d3))
print("Total",len(d2)+len(d3))