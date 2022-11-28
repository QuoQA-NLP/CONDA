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