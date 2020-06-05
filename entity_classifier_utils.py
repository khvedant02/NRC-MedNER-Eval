# Author: Isar Nejadgholi
# Text Analytics, National Research Council Canada

import os
import nltk
import spacy
nlp = spacy.load("en")
import random 
import numpy as np 

from tqdm.auto import tqdm
from seqeval.metrics.sequence_labeling import get_entities

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label_id=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text = text
        self.label = label_id

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,input_ids, input_mask, input_segment, label_id ):
        
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_segment = input_segment
        self.label_id = label_id
        

def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []

    return data

def data_2_tag_dict(data):
  label_list = [item[1] for item in data]
  OIB_labels = set([item for sublist in label_list for item in sublist])
  tag_vocab = list(set([item[2:] for item in OIB_labels if len(item)>2]))
  tag_vocab.sort()
  tag_2_id = dict((item, tag_vocab.index(item)) for item in tag_vocab )
  tag_2_id['other'] = len(tag_vocab)
  return tag_2_id


class EntityClassifierProcessor(object):
#processor for training an entity classifier 
  
  def get_examples(self, data_path,tag_2_id ):
    #gets text files in IOB format and returns datafarmes to be used for training the entity classifier 
    return self._create_examples(readfile(data_path), tag_2_id)

  def _create_examples(self,data, tag_2_id):
    examples = []
    null_text = []
    null_examples = []  #examples of the class "other"
    exclude_noun_chunks = ['he','she']
    for (sentence,label) in data:
      entities = []
      entities = get_entities(label)
      
      for entity in entities:
        guid = len(examples)+1
        text = ' '.join(sentence[entity[1]:entity[2]+1])
        label_id = tag_2_id[entity[0]]
        examples.append(InputExample(guid=guid,text=text,label_id=label_id))

      O_words = [sentence[i] for i, item in enumerate(label) if item=='O']

      null_text += O_words

      if len(' '.join(null_text))>1000000:
        text = ' '.join(null_text)[:1000000]
        null_doc = nlp(text)
        null_examples.extend([np for np in null_doc.noun_chunks if (np.text.lower() not in exclude_noun_chunks) and len(np.text)>2])
        null_text = []

    
    null_doc = nlp(' '.join(null_text))
    null_examples.extend([np for np in null_doc.noun_chunks if (np.text.lower() not in exclude_noun_chunks) and len(np.text)>2])
    if len(null_examples)> np.int(len(examples)/(len(tag_2_id)-1)):
      random.seed(0)
      null_examples =random.sample(null_examples , np.int(len(examples)/(len(tag_2_id)-1)) )
    for example in null_examples:
      guid = len(examples)+1
      text = example.text#' '.join(example)
      label_id = tag_2_id['other']
      examples.append(InputExample(guid=guid,text=text,label_id=label_id))


    return examples 

  

def convert_example_to_feature(example, tokenizer  ,  max_seq_length = 20):

  tokens = tokenizer.tokenize(example.text)
    
  if len(tokens) > max_seq_length - 3:
      tokens = tokens[:(max_seq_length - 3)]

  tokens = ["CLS"] +tokens+["SEP"]

  input_mask = [1] * len(tokens)

  padding_length = max_seq_length - len(tokens)

  tokens = tokens + (["PAD"] * padding_length)  
  input_mask = input_mask + ([0] * padding_length)
  input_segment = [1]*len(input_mask)  #distilbert does not need segment
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(input_segment) == max_seq_length

  return InputFeatures(
  input_ids=input_ids,
  input_mask=input_mask,
  input_segment=input_segment,
  label_id=example.label
  )


def convert_examples_to_features(examples, tokenizer , max_seq_length = 20):
  features = [convert_example_to_feature(example, 
                                         max_seq_length = max_seq_length, 
                                         tokenizer  = tokenizer) for example in tqdm(examples)]
  return features

