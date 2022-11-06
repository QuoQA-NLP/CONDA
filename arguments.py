from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    PLM: str = field(
        default="bert-base-cased",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    model_name: str = field(
        default="JointBert",
        metadata={
            "help": "Model class name"
        },
    )
    save_path: str = field(
        default="results", metadata={"help": "Path to save checkpoint from fine tune model"},
    )
    

@dataclass
class DataTrainingArguments:
    max_length: int = field(
        default=512, metadata={"help": "Max length of input sequence"},
    )
    data_dir: str = field(
        default="data", metadata={"help": "path of data directory"}
    )
    train_data_file: str = field(
        default="CONDA_train.csv", metadata={"help": "name of train data"}
    )
    eval_data_file: str = field(
        default="CONDA_valid.csv", metadata={"help": "name of test data"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    report_to: Optional[str] = field(default="wandb")
    output_dir: str = field(
        default="exps",
        metadata={"help": "model output directory"}
    )
    entity_side: str = field(
        default="right",
        metadata={"help": "entity size"}
    )


@dataclass
class LoggingArguments:
    dotenv_path: Optional[str] = field(
        default="wandb.env", metadata={"help": "input your dotenv path"},
    )
    project_name: Optional[str] = field(
        default="CONDA", metadata={"help": "project name"},
    )
    group_name: Optional[str] = field(
        default="Baseline", metadata={"help": "group name"},
    )
