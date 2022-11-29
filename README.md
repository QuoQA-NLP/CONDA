# CONDA
---

## Goal
  * Understand and review paper (CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity understanding and detection)
  * Making a model which can detect toxic language using CONDA dataset
  
## Source
  * Paper : https://arxiv.org/pdf/2106.06213.pdf
  * Dataset : https://github.com/usydnlp/CONDA

## Task Description
  ![스크린샷 2022-11-29 오후 10 04 59](https://user-images.githubusercontent.com/48673702/204536569-3317cbb7-a477-4ecb-8da1-33974abb2808.png)
  1. Task1 : Classify intent of utterance
  2. Task2 : Classify slot for each token in utterance
  
## Model Structure
  ![스크린샷 2022-11-29 오후 10 06 30](https://user-images.githubusercontent.com/48673702/204536855-7c46b29b-5f36-4ca4-94d7-9ec8b2810a01.png)
  * Name : JointBert
  * Source : https://github.com/monologg/JointBERT
 

## Argument
|Argument|Description|Default|
|--------|-----------|-------|
|PLM|model name(huggingface)|bert-base-cased|
|epochs|train epochs|5|
|lr|learning rate|3e-5|
|train_batch_size|train batch size|32|
|eval_batch_size|evaluation batch size|32|
|max_len|input max length|256|
|warmup_ratio|warmup ratio|0.05|
|weight_decay|weight decay|1e-3|
|evaluation_strategy|evaluation strategy|steps|
|save_steps|model save steps|500|
|eval_steps|model evaluation steps|500|
|logging_steps|logging steps|100|
|seed|random seed|42|


## Terminal Command Example
  ```Shell
  # Training
  python train.py \
  --do_train \
  --do_eval \
  --seed 42 \
  --PLM bert-base-cased \
  --model_name JointBert \
  --train_data_file CONDA_train.csv \
  --eval_data_file CONDA_valid.csv \
  --overwrite_output_dir \
  --warmup_ratio 0.05 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --max_length 256 \
  --save_strategy steps \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_steps 500 \
  --save_total_limit 5 \
  --logging_steps 100 \
  --output_dir ./exps \
  --learning_rate 3e-5 \
  --weight_decay 1e-3

  # Predict
  python evaluate.py \
  --PLM /home/wkrtkd911/project/conda/CONDA/results \
  --model_name JointBert \
  --eval_data_file CONDA_valid.csv \
  --per_device_eval_batch_size 32 \
  --report_to none \
  --max_length 256
  ```

## Result (Validation Performance)
#### Utterance
|UCA|U-F1(E)|U-F1(I)|U-F1(A)|U-F1(O)|
|-----|----|----|----|----|
|92.09|86.21|77.33|80.55|95.42|

#### Slot
|T-F1(T)|T-F1(S)|T-F1(C)|T-F1(D)|T-F1(P)|T-F1(O)|
|-----|----|----|----|----|----|
|94.64|98.26|94.61|84.02|99.32|98.16|

#### Overall
|JSA|
|-----|
|87.54|

