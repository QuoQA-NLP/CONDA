import os
import importlib
import pandas as pd
import numpy as np
import multiprocessing
from utils.loader import load_dataset
from utils.preprocessor import filter, preprocess
from utils.metrics import Metrics
from utils.collator import DataCollatorForJoint
from utils.encoder import Encoder

from arguments import (
    ModelArguments, 
    DataTrainingArguments, 
    TrainingArguments, 
)
from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
)

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # -- Loading datasets
    print("\nLoad datasets")

    eval_data_file = os.path.join(data_args.data_dir, data_args.eval_data_file)
    eval_dataset = load_dataset(eval_data_file)

    eval_dataset = eval_dataset.remove_columns(['matchId', 'conversationId', 'chatTime', 'playerSlot', 'playerId', '__index_level_0__'])

    cpu_cores = multiprocessing.cpu_count()
    num_proc = int(cpu_cores // 2)

    # -- Preprocess datasets
    eval_dataset = eval_dataset.map(preprocess, batched=True, num_proc=num_proc)
    print(eval_dataset)

    eval_dataset = eval_dataset.filter(lambda x : filter(x))
    eval_ids = list(eval_dataset['Id'])
    print(eval_dataset)

    # -- Load tokenizer
    print("\nLoad tokenizer")
    model_name = model_args.PLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    intent_label_dict = {'O' : 0, 'E' : 1, 'A' : 2, 'I' : 3}
    slot_label_dict = {'X' : 0, 'O' : 1, 'T' : 2, 'P' : 3, 'SEPA' : 4, 'S' : 5, 'D' : 6, 'C' : 7}

    encoder = Encoder(tokenizer, data_args.max_length, intent_label_dict, slot_label_dict)
    eval_dataset = eval_dataset.map(encoder, batched=True, num_proc=num_proc)

    # -- Loading config & Model
    print("\nLoad Model")
    config = AutoConfig.from_pretrained(model_args.PLM)
    config.intent_num_labels = len(intent_label_dict)
    config.slot_num_labels = len(slot_label_dict)

    model_category = importlib.import_module('models.model')
    model_class = getattr(model_category, model_args.model_name)
    model = model_class.from_pretrained(model_args.PLM, config=config)

    # -- DataCollator
    data_collator = DataCollatorForJoint(
        tokenizer=tokenizer, 
        padding=True,
        max_length=data_args.max_length   
    )

    # -- Metrics
    metrics = Metrics(intent_label_dict, slot_label_dict)
    compute_metrics = metrics.compute_metrics

    # -- Trainer
    trainer = Trainer(
        model,
        training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print('\Predict')
    predictions = trainer.predict(test_dataset=eval_dataset)
    intent_predictions, slot_predictions = predictions[0]

    # Indent Postprocessing
    id2intent = {v:k for k, v in intent_label_dict.items()}
    intent_pred_args = np.argmax(intent_predictions, axis=-1)
    intent_pred_args = [id2intent[arg] for arg in intent_pred_args]

    # Slot Postprocessing
    id2slot = {v:k for k, v in slot_label_dict.items()}
    slot_pred_args = np.argmax(slot_predictions, axis=-1)
    slot_string_list = []

    for i in range(len(slot_predictions)) :
        org_data = eval_dataset[i]['input_ids']
        start, end = org_data.index(101), org_data.index(102)

        target_input_ids = org_data[start+1:end]
        target_input_tokens = tokenizer.convert_ids_to_tokens(target_input_ids)
        target_pred = slot_pred_args[i][start+1:end]

        j, cur = 1, 0
        slot_pairs = []
        while j < len(target_pred) :
            if '##' not in target_input_tokens[j] :
                slot_pairs.append(
                    (target_pred[cur], (cur,j))
                )
                cur = j
            j += 1
        slot_pairs.append((target_pred[cur], (cur, len(target_pred))))
        
        slot_strings = []
        for pair in slot_pairs :
            slot, position = pair
            slot = id2slot[slot]
            slot_tokens = tokenizer.convert_ids_to_tokens(target_input_ids[position[0]:position[1]])
            slot_str = tokenizer.convert_tokens_to_string(slot_tokens)

            slot_strings.append(slot_str + ' (' + slot + ')')
        
        slot_string_list.append(', '.join(slot_strings))
    
    eval_df = pd.read_csv(os.path.join(data_args.data_dir, data_args.eval_data_file))
    eval_df = eval_df[[True if id in eval_ids else False for id in eval_df['Id']]]

    eval_df['intentClass'] = intent_pred_args
    eval_df['slotTokens'] = slot_string_list
    eval_df.to_csv('./result.csv', index=False)

if __name__ == "__main__":
    main()