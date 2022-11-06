
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers.utils import ModelOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from typing import Optional, Union, Tuple

@dataclass
class JointClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    intent_logits: torch.FloatTensor = None
    slot_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class JointBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.intent_classifier = nn.Linear(config.hidden_size, config.intent_num_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, config.slot_num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], JointClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        cls_output = sequence_output[:, 0, :]

        intent_logits = self.intent_classifier(cls_output)
        slot_logits = self.slot_classifier(sequence_output)

        loss = None
        if slot_labels is not None and intent_labels is not None :
            loss_fct = nn.CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, self.config.intent_num_labels), intent_labels.view(-1))
            slot_loss = loss_fct(slot_logits.view(-1, self.config.slot_num_labels), slot_labels.view(-1))
            
            loss = intent_loss + slot_loss

        if not return_dict:
            output = (intent_logits, slot_logits, ) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return JointClassifierOutput(
            loss=loss,
            intent_logits=intent_logits,
            slot_logits=slot_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
