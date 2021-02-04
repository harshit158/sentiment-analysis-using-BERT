import torch
import config


class BERTDataset:
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.targets = targets
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        # preprocessing
        review = ' '.join(self.reviews[idx].split())

        # transforming raw review and target into form that is ingestible into BERT Model
        inputs = self.tokenizer.encode_plus(
            review,
            None,       # since this is not a sentence pair task
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )

        ids = inputs['input_ids']                    # Eg: [101, 2342, 122, 102] (101 -> CLS, 102 -> SEP)
        mask = inputs['attention_mask']              # Eg: [1, 1, 1, 1]
        token_type_ids = inputs['token_type_ids']    # Eg: [0, 0, 0, 0] All zeros since this is a single sentence task

        # padding the input as per max_len
        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            'ids': torch.tensor(ids),
            'mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids), 
            'target': torch.tensor(self.targets[idx])
        }