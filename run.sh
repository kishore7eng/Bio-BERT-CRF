#!/bin/bash
echo $(pwd)

python trans.py # To download pretrained BERT from transformers
python BioBertCRFModel.py # To do training on BC5CDR Disease dataset
python eval.py #To do evaluatoin on testing data
export SAVE_DIR=./output
export DATA_DIR= ./resources

export DATA=BC5CDR-disease
export SEED=1

python split.py \
      --mention_dictionary $DATA/mention_dictionary.txt \
      --cui_dictionary $DATA/cui_dictionary.txt \
      --gold_labels $DATA/test.txt \
      --gold_cuis $DATA/test_cuis.txt \
      --predictions test_predictions.txt 
#To do splitted evaluation on MEM, SYN,CON

