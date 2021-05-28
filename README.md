# EMQA

This repository contains code and data for running the experiments and reproducing the results of the paper: "[Towards More Equitable Question Answering Systems: How Much More Data Do You Need?](link)".

# Model


# Dataset
Download the dataset from the following links and put them under the ```data``` directory.

* TyDi QA (Original dataset from Google): [train](https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json) | [dev](https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json)
* TyDi QA (Seperated by language): [train](https://drive.google.com/drive/folders/1AGwrx8pjvLpu2RVK0ezQhpVl9HwGx1eQ?usp=sharing) | [dev](https://drive.google.com/drive/folders/1i13z8soDaEow2B_mL1wrge-Mb8DTyKx7?usp=sharing)
* SQuAD (Original train set for zero-shot setting): [link](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
* tSQuAD: [link](https://exchangelabsgmu-my.sharepoint.com/:f:/g/personal/adebnath_masonlive_gmu_edu/EuR5t97u7kJFvPYnAAa5i3oBSpDhMPNfKBTF9rAVwraf0A?e=eXz09n)
* mSQuAD: [link](https://drive.google.com/drive/folders/1vlLgllRZubvQPr4N0Obg0BDH3f-yuLQK?usp=sharing)
* Disproportional allocations: [link](https://drive.google.com/drive/folders/1hTlUlcqd0i2BtiIdhJKUGx1tKFSGPppz?usp=sharing)

# Running Experiments


## Requirements
After creating a virtual environment, installing Python 3.6+, PyTorch 1.3.1+, and CUDA (tested with 10.1), install the Transformers library as follows:
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
      --train_file './data/tydiqa-goldp-v1.1-train.json' \
      --predict_file './data/tydiqa-goldp-v1.1-dev.json' \
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
      --train_file './data/tydiqa-goldp-v1.1-train.json' \
      --predict_file './data/tydiqa-goldp-v1.1-dev.json'  \
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
      --predict_file './data/tydiqa-goldp-v1.1-dev.json' \
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
      --train_file './data/dataset_for_fineTuning.json' \
      --predict_file './data/tydiqa-goldp-v1.1-dev.json' \
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
If you use this code, please refer to our ACL 2021 paper using the following BibTeX entry:
~~~
@inproceedings{debnath-etal-2021-towards,
    title = "Towards More Equitable Question Answering Systems: How Much More Data Do You Need?",
    author = "Debnath, Arnab  and Rajabi, Navid  and Alam, Fardina Fathmiul and Anastasopoulos, Antonios",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = "",
    pages = "",
}
~~~


# License
Our code and data for EMQA are available under the [MIT License](https://github.com/NavidRajabi/EMQA/blob/main/LICENSE).

