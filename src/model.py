import transformers
import torch.nn as nn


class BERTSentimentModel(nn.Module):
    def __init__(self):
        super(BERTSentimentModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(0.2)
        self.out = nn.Linear(768, 1)  # 1 because its a Binary classification task

    def forward(self, ids, attention_mask, token_type_ids):
        output = self.bert(
            ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # o1 shape : [batch_size, maxlen, hidden_state]
        # o2 shape : [batch_size, hidden_state] (hidden state corresponding to CLS token)

        bert_output = self.bert_drop(output['pooler_output'])
        final_output = self.out(bert_output)

        return final_output