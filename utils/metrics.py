
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction

class Metrics :

    def __init__(self, intent_label_dict, slot_label_dict) :
        self.intent_label_dict = {v:k for k, v in intent_label_dict.items()}
        self.slot_label_dict = {v:k for k, v in slot_label_dict.items()}

    def compute_metrics(self, pred: EvalPrediction):

        predictions = pred.predictions
        intent_predictions, slot_predictions = predictions[0], predictions[1]
        intent_pred_ids = np.argmax(intent_predictions, axis=-1)
        slot_pred_ids = np.argmax(slot_predictions, axis=-1)

        label_ids = pred.label_ids
        intent_labels, slot_labels = label_ids[0], label_ids[1]

        token_pred_ids, token_labels = [], []
        
        batch_size, seq_size = slot_labels.shape
        for i in range(batch_size) :
            s_p, s_l = slot_pred_ids[i], slot_labels[i]
    
            for p, l in zip(s_p, s_l) :

                if l == -100 or l == 0 :
                    continue
                
                token_pred_ids.append(p)
                token_labels.append(l)

            
        uca = accuracy_score(intent_labels, intent_pred_ids)
        u_f1 = f1_score(intent_labels, intent_pred_ids, average=None) 
        u_f1 = {'u_f1(' + self.intent_label_dict[u] + ')' : v for u, v in enumerate(u_f1)}

        t_f1 = f1_score(token_pred_ids, token_labels, average=None)
        t_f1 = {'t_f1(' + self.slot_label_dict[t] + ')' : v for t, v in enumerate(t_f1) if t > 0}

        jca = 0.0
        for i in range(batch_size) :        
            
            flag = True

            if intent_pred_ids[i] == intent_labels[i] :
                for j in range(seq_size) :
                    if slot_labels[i][j] != -100 :
                        if slot_labels[i][j] != slot_pred_ids[i][j] :
                            flag = False 
                            break
            else :
                flag = False

            if flag == True :
                jca += 1

        jca /= batch_size

        metric = {'uca' : uca, 'jca' : jca}
        metric.update(u_f1)
        metric.update(t_f1)

        return metric