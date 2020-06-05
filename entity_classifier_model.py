# Author: Isar Nejadgholi
# Text Analytics, National Research Council Canada

import torch
import numpy as np 
import math
import pickle
import torch.nn.functional as F
import os

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm.auto import trange, tqdm
from entity_classifier_utils import EntityClassifierProcessor, convert_examples_to_features,InputExample, readfile, data_2_tag_dict
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, classification_report,precision_score


#default tags for i2b2 dataset
#for other datasets tags are extracted from IOB format training set
tag_2_id = {'problem':0, 'test':1, 'treatment':2, 'other':3}  

def convert_features_to_dataset(features):
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
  all_input_segment = torch.tensor([f.input_segment for f in features], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
  
  dataset = TensorDataset(all_input_ids, all_input_mask, all_input_segment, all_label_ids)
  return dataset



class EntityIdentifier:
  #This is the entity classifier (here classifier and identifier are used interchangably)
  def __init__(self,  train_data_path = None, test_data_path = None, use_cuda=True, num_labels = 4, tag_2_id = tag_2_id):
    """
        Initializes a DistilBert ClassificationModel model.
        Args:
            num_labels (optional): The number of labels or classes in the dataset.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
    """
    if train_data_path== None:  #this is the path to training dataset of NER model in IOB format
      self.num_labels = num_labels
      self.tag_2_id = tag_2_id
    else:
      self.train_data_path = train_data_path
      self.tag_2_id = data_2_tag_dict(readfile(self.train_data_path))
      self.num_labels = len( self.tag_2_id)
    self.test_data_path = test_data_path  #this is the path to test dataset of NER model in IOB format
    if use_cuda:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise ValueError("'use_cuda' set to True when cuda is unavailable. Make sure CUDA is available or set use_cuda=False.")
    else:
        self.device = "cpu" 

    
    model_class, tokenizer_class = ( DistilBertForSequenceClassification, DistilBertTokenizer)
    model_name = 'distilbert-base-uncased'
    
    self.tokenizer = tokenizer_class.from_pretrained(model_name)

    self.model = model_class.from_pretrained(model_name,num_labels= self.num_labels)
    self.gradient_accumulation_steps = 1
    self.learning_rate = 2e-5
    self.model.to(self.device)
    self.precisions = None



  def build_model(self, num_train_epochs = 1, train_batch_size = 16 , test_batch_size = 64):
    # trains the classifier and calculates percisions for each class
    print("_________ Training __________")
    self.train(train_data_path=self.train_data_path,num_train_epochs = num_train_epochs, train_batch_size = train_batch_size)
    print("_________ Getting Precisions __________")
    self.precisions = self.get_precision(test_data_path= self.test_data_path, test_batch_size = test_batch_size)
    print('Precisions = ',self.precisions )
    print('tags disctionary : ', self.tag_2_id)


  def train(self, train_data_path=None, num_train_epochs = 4, train_batch_size = 16):
    # main training function 
    if train_data_path== None:
      train_data_path = self.train_data_path
    
    examples = self.process_data(train_data_path)
    
    param_optimizer = list(self.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    train_features = convert_examples_to_features(examples, tokenizer= self.tokenizer, max_seq_length = 20)
    train_dataset = convert_features_to_dataset(train_features)
 

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(train_dataloader) // self.gradient_accumulation_steps * num_train_epochs

    optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=math.ceil(t_total * 0.06), num_training_steps=t_total)

    global_step = 0
    
    self.model.zero_grad()
    self.model.train()       
    for _ in trange(int(num_train_epochs)):
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, tag_ids = batch

            inputs = {
                "input_ids":      batch[0],
                "attention_mask": batch[1],
                "labels":         batch[3]
            }
            output = self.model(**inputs)
          
            
            loss = output[0]
            
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)


            #print("\r%f" % loss, end='')
            
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % self.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                global_step += 1
          
    return global_step, tr_loss / nb_tr_steps


  def test_and_eval(self,test_data_path,  test_batch_size = 64):
    examples = self.process_data(test_data_path)
    test_features = convert_examples_to_features(examples, tokenizer= self.tokenizer, max_seq_length = 20)
    test_dataset = convert_features_to_dataset(test_features)
    test_dataloader = DataLoader(test_dataset,  batch_size=test_batch_size)
    
    self.model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in test_dataloader:
        batch = tuple(t.to(self.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids":      batch[0],
                "attention_mask": batch[1],
                "labels":         batch[3]
            }
            outputs = self.model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    model_outputs = preds

    preds = np.argmax(preds, axis=1)
    f1s = f1_score(out_label_ids, preds, average = 'macro')
    accuracy = accuracy_score(out_label_ids, preds)
    print('f1_score = ', f1s )
    print('Accuracy = ', accuracy)

    #with open(results_path, 'w') as f_:
      #f_.write(classification_report(out_label_ids, preds))
    print('\n\n', classification_report(out_label_ids, preds))
    return f1s, accuracy

  def get_precision(self,test_data_path, test_batch_size = 64):
    
    examples = self.process_data(test_data_path)
    test_features = convert_examples_to_features(examples, tokenizer= self.tokenizer, max_seq_length = 20)
    test_dataset = convert_features_to_dataset(test_features)
    test_dataloader = DataLoader(test_dataset,  batch_size=test_batch_size)
    
    self.model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in test_dataloader:
        batch = tuple(t.to(self.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids":      batch[0],
                "attention_mask": batch[1],
                "labels":         batch[3]
            }
            outputs = self.model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1


        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    model_outputs = preds

    preds = np.argmax(preds, axis=1)
    precisions = precision_score(out_label_ids, preds, average = None)
    
    return precisions

  def save_model(self,  model_save_dir):
    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
    output_model_file = os.path.join(model_save_dir, "recognizer_model.bin")
    to_save = {'model':model_to_save.state_dict(),
                'precisions':self.precisions,
                'tag_2_id':self.tag_2_id}
    torch.save(to_save, output_model_file)
    self.tokenizer.save_vocabulary( model_save_dir)

  def load_model(self, model_save_dir):
    output_model_file = os.path.join(model_save_dir, "recognizer_model.bin")
    loaded_data = torch.load( output_model_file)
    self.model.load_state_dict(loaded_data['model'])
    self.precisions = loaded_data['precisions']
    self.tag_2_id = loaded_data['tag_2_id']


  def predict(self,entity_list, predict_batch_size = 128):
    examples = [InputExample(guid=entity_list.index(entity),text=entity,label_id=0) for entity in entity_list]
    test_features = convert_examples_to_features(examples, tokenizer= self.tokenizer,  max_seq_length = 20)
    test_dataset = convert_features_to_dataset(test_features)
    test_dataloader = DataLoader(test_dataset,  batch_size=predict_batch_size)
    
    self.model.eval()
    #eval_loss = 0.0
    #nb_eval_steps = 0
    preds = None
    IDs = None
    for batch in test_dataloader:
        batch = tuple(t.to(self.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids":      batch[0],
                "attention_mask": batch[1],
                "labels":         batch[3]
            }
            outputs = self.model(**inputs)
            tmp_eval_loss, logits = outputs[:2]


        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            
    id_2_tag = {v:k for k,v in self.tag_2_id.items()}

    predictions = torch.nn.functional.softmax(torch.as_tensor(preds),1) 
    return predictions,id_2_tag

  def process_data(self, data_path):
  #Builds examples given the path to the IOB file (training set of NER model)
    processsor = EntityClassifierProcessor()
    examples = processsor.get_examples( data_path, tag_2_id = self.tag_2_id)
    
    return examples

