
import re

def preprocess(dataset) :

    slot_words = []
    slot_labels = []

    for slots in dataset['slotTokens'] :

        slots = slots.strip()[:-1].split(', ')
        words, labels = [], []
        for s in slots :
            word, slot = s.split(' ')
            slot = slot[1:-1]

            words.append(word)
            labels.append(slot)

        slot_words.append(words)
        slot_labels.append(labels)

    dataset['slotWords'] = slot_words
    dataset['slotClasses'] = slot_labels

    return dataset


def filter(data) :
    utterance = re.sub('\s+', ' ', data['utterance'])
    if ' '.join(data['slotWords']) in utterance :
        return True
    else :
        return False