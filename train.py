import os
import torch
import random
import wandb
import numpy as np
import multiprocessing
import importlib
from utils.loader import load_dataset
from utils.preprocessor import filter, preprocess
from utils.metrics import Metrics
from utils.collator import DataCollatorForJoint
from utils.encoder import Encoder
from dotenv import load_dotenv
from datasets import DatasetDict

from arguments import (
    ModelArguments, 
    DataTrainingArguments, 
    TrainingArguments, 
    LoggingArguments
)
from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LoggingArguments)
    )
    model_args, data_args, training_args, logging_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)

    # -- Loading datasets
    print("\nLoad datasets")
    train_data_file = os.path.join(data_args.data_dir, data_args.train_data_file)
    train_dataset = load_dataset(train_data_file)

    eval_data_file = os.path.join(data_args.data_dir, data_args.eval_data_file)
    eval_dataset = load_dataset(eval_data_file)

    datasets = DatasetDict({'train' : train_dataset, 'validation' : eval_dataset})
    datasets = datasets.remove_columns(['Id', 'matchId', 'conversationId', 'chatTime', 'playerSlot', 'playerId', '__index_level_0__'])
    
    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)

    # -- Preprocess datasets
    datasets = datasets.map(preprocess, batched=True, num_proc=num_proc)
    print(datasets)

    datasets = datasets.filter(lambda x : filter(x))
    print(datasets)

    # -- Load tokenizer
    print("\nLoad tokenizer")
    model_name = model_args.PLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    special_tokens_dict = {'additional_special_tokens': ['[SEPA]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    intent_label_dict = {'O' : 0, 'E' : 1, 'A' : 2, 'I' : 3}
    slot_label_dict = {'X' : 0, 'O' : 1, 'T' : 2, 'P' : 3, 'SEPA' : 4, 'S' : 5, 'D' : 6, 'C' : 7}

    encoder = Encoder(tokenizer, data_args.max_length, intent_label_dict, slot_label_dict)
    datasets = datasets.map(encoder, batched=True, num_proc=num_proc)

    # -- Loading config & Model
    print("\nLoad Model")
    config = AutoConfig.from_pretrained(model_args.PLM)
    config.intent_num_labels = len(intent_label_dict)
    config.slot_num_labels = len(slot_label_dict)

    model_category = importlib.import_module('models.model')
    model_class = getattr(model_category, model_args.model_name)
    model = model_class.from_pretrained(model_args.PLM, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # -- DataCollator
    data_collator = DataCollatorForJoint(
        tokenizer=tokenizer, 
        padding=True,
        max_length=data_args.max_length   
    )

    # -- Metrics
    metrics = Metrics(intent_label_dict, slot_label_dict)
    compute_metrics = metrics.compute_metrics

    load_dotenv(dotenv_path=logging_args.dotenv_path)

    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    args = training_args
    training_args.dataloader_num_workers = num_proc
    wandb_name = f'PLM:{model_args.PLM}_EP:{args.num_train_epochs}_BS:{args.per_device_train_batch_size}_LR:{args.learning_rate}_WD:{args.weight_decay}_WR:{args.warmup_ratio}'
    wandb.init(
        entity='quoqa-nlp',
        project=logging_args.project_name, 
        name=wandb_name,
        group=logging_args.group_name
    )
    wandb.config.update(training_args)

    # -- Trainer
    trainer = Trainer(
        model,
        training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print('\nTraining')
    trainer.train()

    print('\nEvaluating')
    trainer.evaluate()

    trainer.save_model(model_args.save_path)
    wandb.finish()


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)


if __name__ == "__main__":
    main()