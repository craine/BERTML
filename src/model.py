import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        #look at BertModel from transformers documentation
        self.bert = transformers.BertModel.from_pretrained(
            config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        #1 because it is binary classification model
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        #out1 now _ is last hidden state out2 o2 is bert puller layer
        _, o2 = self.bert(
            ids, attention_mask=mask,
            token_type_ids=token_type_ids
        )

        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
