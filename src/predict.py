import torch
from torch._C import dtype
import config
from model import BERTSentimentModel

class Predict:
    def __init__(self):
        self.tokenizer=config.TOKENIZER
        self.device=config.DEVICE
        self.max_len=config.MAX_LEN

        self.model=BERTSentimentModel()
        self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
        self.model.to(config.DEVICE)
        self.model.eval()

    def predict(self, review):
        input = self.tokenizer.encode_plus(
                review,
                None,       # since this is not a sentence pair task
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True
            )

        ids = input['input_ids']                    # Eg: [101, 2342, 122, 102] (101 -> CLS, 102 -> SEP)
        mask = input['attention_mask']              # Eg: [1, 1, 1, 1]
        token_type_ids = input['token_type_ids']    # Eg: [0, 0, 0, 0] All zeros since this is a single sentence task

        # padding the input as per max_len
        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.long, device=self.device).unsqueeze(0)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        # ids = ids.to(device, dtype=torch.long)
        # mask = mask.to(device, dtype=torch.long)
        # token_type_ids = token_type_ids.to(device, dtype=torch.long)

        output = self.model(ids=ids, 
                        attention_mask=mask, 
                        token_type_ids=token_type_ids)

        output = torch.sigmoid(output).cpu().detach().numpy()
        output = output[0][0]

        return output

if __name__=="__main__":
    # accepting test sample from user
    review = input('Enter the test sentence\n')
    print(Predict().predict(review))