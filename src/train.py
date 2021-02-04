import torch
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import model_selection
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.utils.logging import enable_propagation

import config
import dataset
import engine
from model import BERTSentimentModel

def get_data_loaders(df, mode='train'):
    my_dataset = dataset.BERTDataset(
        reviews=df.review.values,
        targets=df.sentiment.values
    )

    data_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=config.TRAIN_BATCH_SIZE if mode=='train' else config.VALID_BATCH_SIZE,
        num_workers=4
    )
    return data_loader


def run():
    df = pd.read_csv(config.TRAINING_FILE).fillna("none")

    # converting sentiments from "positive" and "negative" to 1 and 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    # splitting dataset into train and validation
    df_train, df_valid = model_selection.train_test_split(
        df,
        test_size=0.2,
        random_state=15,
        stratify=df.sentiment.values
    )

    # resetting the index jumbled due to train test split
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # preparing dataloaders
    train_data_loader = get_data_loaders(df_train, mode='train')
    valid_data_loader = get_data_loaders(df_valid, mode='valid')

    # instantiating the BERT model
    model = BERTSentimentModel()

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params':[p for n, p in param_optimizer if n not in no_decay], 'weight_decay':0.001},
        {'params':[p for n, p in param_optimizer if n in no_decay], 'weight_decay':0.0}
    ]

    num_train_steps = int((len(df_train)/config.TRAIN_BATCH_SIZE)*config.EPOCHS) 
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = 0,
        num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model)
        
        # calculating evaluation metrics
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy score = {accuracy}")
        
        # saving model
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy

if __name__ == "__main__":
    run()