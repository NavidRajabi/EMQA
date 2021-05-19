# EMQA

This repository contains code and data for running the experiments and reproducing the results of the paper: "[Towards More Equitable Question Answering Systems: How Much More Data You Need?](link)".

# Model


# Dataset
Download the full dataset from [here](link) and put it under the ```data``` directory.

# Running Experiments


## Requirements
After installing Python 3.6+, PyTorch 1.3.1+, and CUDA (tested with 10.1), install the Transformers library as follows:
```
pip install transformers
```

## Training
If you want to use multilingual-bert model, run the following command:
```
python run_squad.py \
  --model_type bert \
  --model_name_or_path=bert-base-multilingual-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file './tydiqa-goldp-v1.1-train.json' \
  --predict_file './tydiqa-goldp-v1.1-dev.json' \
  --per_gpu_train_batch_size 24 \
  --per_gpu_eval_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir './train_cache_output/'
  --overwrite_cache
```
Otherwise, run the following command to use XLM-Roberta-Large model instead:
```
python run_squad.py  \
      --model_type=xlm-roberta \
      --model_name=xlm-roberta-large  \
      --do_train \
      --do_eval \
      --do_lower_case \
      --train_file './tydiqa-goldp-v1.1-train.json' \
      --predict_file './tydiqa-goldp-v1.1-dev.json'  \
      --per_gpu_train_batch_size 24 \
      --per_gpu_eval_batch_size 24 \
      --learning_rate 3e-5  \
      --num_train_epochs 3  \
      --max_seq_length 384  \
      --doc_stride 128  \
      --output_dir './train_cache_output/' \
      --overwrite_cache
```

## Evaluating
For the evaluation-only situation, replace the model path of ```--model_name``` with the path to the cache directory of your pre-trained model and run the following command:
```
python run_squad.py \
  --model_type bert \
  --model_name_or_path='./train_cache_output/' \
  --do_eval \
  --do_lower_case \
  --predict_file './tydiqa-goldp-v1.1-dev.json' \
  --per_gpu_eval_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir './eval_cache_output/'
  --overwrite_cache
```

## Fine-tuning
For fine-tuning, run the following command:
```
python run_squad.py \
  --model_type bert \
  --model_name_or_path='./train_cache_output/' \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file './dataset_for_fineTuning.json' \
  --predict_file './tydiqa-goldp-v1.1-dev.json' \
  --per_gpu_train_batch_size 24 \
  --per_gpu_eval_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir './fineTune_cache_output/'
  --overwrite_cache
```


# Citation


# License


