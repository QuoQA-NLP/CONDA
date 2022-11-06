

class Encoder :

    def __init__(self, tokenizer, max_length, intent_label_dict, slot_label_dict) :
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.intent_label_dict = intent_label_dict
        self.slot_label_dict = slot_label_dict

    def __call__(self, dataset) :

        utterances = dataset['utterance']
        intent_classes = dataset['intentClass']
        slot_words = dataset['slotWords']
        slot_classes = dataset['slotClasses']

        model_inputs = self.tokenizer(
            utterances,
            return_token_type_ids=True,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )

        offset_mappings = model_inputs['offset_mapping']

        intent_labels = []
        slot_labels = []

        for i in range(len(offset_mappings)) :

            slot_class = slot_classes[i]
            word_positions = self.get_position(utterances[i], slot_words[i])
            offset_mapping = offset_mappings[i]

            slot_label = []
            for j in range(1, len(offset_mapping) - 1) :

                start, end = offset_mapping[j]
                flag = False

                for k, p in enumerate(word_positions) :
                    if p[0] <= start and end <= p[1] :
                        slot_label.append(self.slot_label_dict[slot_class[k]])
                        flag = True
                        break

                if flag == False :
                    slot_label.append(self.slot_label_dict['X'])

            slot_label = [-100] + slot_label + [-100]

            assert len(slot_label) == len(model_inputs['input_ids'][i])
            slot_labels.append(slot_label)
            intent_labels.append(self.intent_label_dict[intent_classes[i]])

        model_inputs['slot_labels'] = slot_labels
        model_inputs['intent_labels'] = intent_labels
        return model_inputs

    def get_position(self, utterance, slot_words) :
        word_positions = []

        start = 0
        for w in slot_words :
            cur_s = utterance[start:].index(w) + start
            cur_e = cur_s + len(w)
            
            word_positions.append((cur_s, cur_e))
            start = cur_e

        return word_positions
        
