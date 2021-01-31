import torch
import pandas as pd
from sklearn import metrics
from sklearn import model_selection

import config
import dataset
from model import BERTSentimentModel


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
    # train dataloader
    train_dataset = dataset.BERTDataset(
        ...
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    # valid dataloader
    valid_dataset = dataset.BERTDataset(
        ...
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1
    )

    # instantiating the BERT model
    model = BERTSentimentModel()
