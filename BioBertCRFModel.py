

    # -*- coding: utf-8 -*-
"""BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dLpGPaaRXlOJDs7JIEIx0tr6E6ahekvO
"""

import pandas as pd
import numpy as np
import csv
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from seqeval.metrics import classification_report
from config import Config as config
import os
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
log_soft = F.log_softmax
tokenizer_dir = "/workspace/amit_pg/BioBertCRF/tokenizer"
model_dir = "/workspace/amit_pg/BioBertCRF/model"
config_dir = "/workspace/amit_pg/BioBertCRF/config"
print(torch.version.cuda)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set to the ID of the GPU you want to use


MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
num_labels = 0

def loadData():
    train_data = pd.read_csv('BC5CDR-disease/train.tsv',sep='\t',quoting=csv.QUOTE_NONE,names=["Tokens","Labels"],skip_blank_lines = False)
    devel_data = pd.read_csv('BC5CDR-disease/devel.tsv',sep='\t',quoting=csv.QUOTE_NONE,names=["Tokens","Labels"],skip_blank_lines = False)
    return train_data,devel_data

def IdToLabelAndLabeltoId(train_data):
    label_list = train_data["Labels"]
    label_list = [*set(label_list)]
    num_labels = len(label_list)
    label_list = [x for x in label_list if not pd.isna(x)]
    id2label = {0 : 'B', 1 : 'I' , 2 : 'O'}
    label2id = { id2label[id]: id for id in id2label}
    return id2label,label2id


def convert_to_sentence(df):
    sent = ""
    sent_list = []
    label = ""
    label_list = []
    for tok,lab in df.itertuples(index = False):
        if pd.isna(tok) and pd.isna(lab):
            sent = sent[1:]
            sent_list.append(sent)
            sent = ""
            label = label[1:]
            label_list.append(label)
            label = ""
        else:
            sent = sent + " " +str(tok)
            label = label+ "," + str(lab)
    if sent != "":
        sent_list.append(sent)
        label_list.append(label)

    return sent_list,label_list



def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []
    attn = []
    useful_tok = []
    sentence = str(sentence)
    sentence = sentence.strip()
    text_labels = str(text_labels)

    for word, label in zip(sentence.split(), text_labels.split(",")):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label])
        labels.extend(['O'] * (n_subwords-1))
        attn.extend([1])
        attn.extend([1] * (n_subwords-1))
        useful_tok.extend([1])
        useful_tok.extend([0] * (n_subwords-1))
    #print(tokenized_sentence)
    #print(labels)
    return tokenized_sentence, labels,attn,useful_tok

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len,label2id,id2label):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.id2label = id2label
        
    def __getitem__(self, index):
        # step 1: tokenize (and adapt corresponding labels)
        sentence = self.data.Sentence[index]  
        word_labels = self.data.Labels[index]
        label2id = self.label2id
       
        tokenized_sentence, labels,attn,useful_tok = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)
        
        # step 2: add special tokens (and corresponding labels)
        tokenized_sentence = ["[CLS]"] + tokenized_sentence + ["[SEP]"] # add special tokens
        labels.insert(0, "O") # add outside label for [CLS] token
        labels.insert(-1, "O") # add outside label for [SEP] token
        attn.insert(0,1)
        attn.insert(-1,0)
        useful_tok.insert(0,1)
        useful_tok.insert(-1,0)
        # step 3: truncating/padding
        maxlen = self.max_len

        if (len(tokenized_sentence) > maxlen):
          # truncate
          tokenized_sentence = tokenized_sentence[:maxlen]
          labels = labels[:maxlen]
          attn = attn[:maxlen]
          useful_tok = useful_tok[:maxlen]
        else:
          # pad
          tokenized_sentence = tokenized_sentence + ['[PAD]'for _ in range(maxlen - len(tokenized_sentence))]
          labels = labels + ["O" for _ in range(maxlen - len(labels))]
          attn = attn + [0 for _ in range(maxlen - len(attn))]
          useful_tok = useful_tok + [0 for _ in range(maxlen - len(useful_tok))]
        # step 4: obtain the attention mask
        #attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        
        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        #print(labels)
        label_ids = [label2id[label] if label != '' else 2 for label in labels]
        # the following line is deprecated
        #label_ids = [label if label != 0 else -100 for label in label_ids]
        #attn_mask = [1 if lab < 3 else 0 for lab in label_ids]
        
        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn, dtype=torch.long),
              #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
              'targets': torch.tensor(label_ids, dtype=torch.long),
              'useful_tok' : torch.tensor(useful_tok, dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len




# define your model architecture
class MyModel(BertPreTrainedModel):
    def __init__(self, config):
        super(MyModel,self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_pretrained('./model')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 3)
        #self.init_weights()
        self.crf = CRF(3, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        attention_mask[:, 0] = 1
        attention_mask = attention_mask.type(torch.uint8)
        if labels is not None:
            attention_mask = attention_mask.type(torch.uint8)
            loss = -self.crf(log_soft(logits, 2), labels, mask=attention_mask, reduction='mean')
            return loss,logits
        else:
            return self.crf.decode(logits, mask=attention_mask),logits



# Defining the training function on the 80% of the dataset for tuning the bert model
def train(epoch,model,training_loader,device,optimizer):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    
    for idx, batch in enumerate(training_loader):
        
        ids = batch['ids'].to(device, dtype = torch.long)
        mask = batch['mask'].to(device, dtype = torch.long)
        targets = batch['targets'].to(device, dtype = torch.long)
        #print(ids)
        #print(mask)
        #print(targets)
        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss, tr_logits = outputs[0], outputs[1]
        tr_loss += loss.item()
        #logits = model(input_ids, attention_mask)[0]
        tr_logits = F.softmax(tr_logits, dim=2)
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)
        
        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_preds.extend(predictions)
        tr_labels.extend(targets)
        
        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

def valid(model, devel_loader,device,id2label):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(devel_loader):
            
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)
            useful_tok = batch['useful_tok'].to(device, dtype = torch.long)
            
            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs[0], outputs[1]
            
            eval_loss += loss.item()
            #logits = model(input_ids, attention_mask)[0]
            eval_logits = F.softmax(eval_logits, dim=2)
            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = useful_tok.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(targets)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy
    
    print(len(eval_labels))
    print(len(eval_preds))
    
    labels,predictions = [], []
    for id1,id2 in zip(eval_labels,eval_preds):
        if id1.item() == 3:
            continue
        else:
            labels.append(id2label[id1.item()])
            predictions.append(id2label[id2.item()])

    #labels = [id2label[id.item()]  for id in eval_labels if (id.item() != -100)]
    #predictions = [id2label[id.item()]  for id in eval_preds if (id.item() != -100)] 

    #print(labels[:50])
    #print(predictions[:50])
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions


def main():
    print("Started")
    train_data,devel_data = loadData()
    id2label,label2id = IdToLabelAndLabeltoId(train_data)


    train_sent,train_label = convert_to_sentence(train_data)
    devel_sent,devel_label = convert_to_sentence(devel_data)
    
    num_labels = 3
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    '''
    config = AutoConfig.from_pretrained(
        './config',
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        'dmis-lab/biobert-v1.1'
    )
    '''
    model = MyModel.from_pretrained(model_dir,num_labels = 3)
    '''
    model = AutoModelForTokenClassification.from_pretrained(
        './model'
    )
    
    model.classifier = nn.Sequential(
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
    nn.ReLU(),
    nn.Linear(model.config.hidden_size, model.config.num_labels),
    CRF(model.config.num_labels, batch_first=True)
    )
    '''
    train_data = {'Sentence' : train_sent,'Labels' : train_label}
    train_data = pd.DataFrame(train_data)
    
    devel_data = {'Sentence' : devel_sent,'Labels' : devel_label}
    devel_data = pd.DataFrame(devel_data)

   
    

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    devel_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }
    training_set = dataset(train_data, tokenizer, MAX_LEN,label2id,id2label)
    devel_set = dataset(devel_data, tokenizer, MAX_LEN,label2id,id2label)
    training_loader = DataLoader(training_set, **train_params)
    devel_loader = DataLoader(devel_set, **devel_params)

    device = 'cuda:0' if cuda.is_available() else 'cpu'
    
    
    '''
    #model = AutoModelForTokenClassification.from_pretrained(model_dir, 
                                                    num_labels=len(id2label),
                                                    id2label=id2label,
                                                    label2id=label2id)'''
    model.to(device)
    print(device)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        print(f"Training epoch: {epoch + 1}")
        train(epoch,model,training_loader,device,optimizer)
        
    model.save_pretrained("/workspace/amit_pg/BioBertCRF/model")
    tokenizer.save_pretrained('/workspace/amit_pg/BioBertCRF/tokenizer')
    #config.save_pretrained("/workspace/amit_pg/bioBert/config")
    
    labels,predictions = valid(model,devel_loader,device,id2label)
    print(classification_report([labels], [predictions]))

if __name__ == "__main__":
    main()
