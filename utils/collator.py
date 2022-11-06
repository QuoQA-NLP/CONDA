import torch
from dataclasses import dataclass
from typing import Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


@dataclass
class DataCollatorForJoint(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):

        slot_labels_name = "slot_labels"
        slot_labels = [feature[slot_labels_name] for feature in features] if slot_labels_name in features[0].keys() else None

        max_sequence_length = max(len(l) for l in slot_labels)

        padding_side = self.tokenizer.padding_side
        for feature in features:
            remainder = [self.label_pad_token_id] * (max_sequence_length - len(feature[slot_labels_name]))
            feature[slot_labels_name] = (
                feature[slot_labels_name] + remainder if padding_side == "right" else remainder + feature[slot_labels_name]
            )

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        return batch