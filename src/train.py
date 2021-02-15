import argparse

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
        batch_size=config.TRAIN_BATCH_SIZE if mode == 'train' else config.VALID_BATCH_SIZE,
        num_workers=4
    )
    return data_loader


def run(args):
    df = pd.read_csv(config.TRAINING_FILE, nrows=args['samples']).fillna("none")

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
    print(f'Train Batches: {len(train_data_loader)}')
    valid_data_loader = get_data_loaders(df_valid, mode='valid')
    print(f'Valid Batches: {len(valid_data_loader)}')

    # instantiating the BERT model
    model = BERTSentimentModel()

    # moving the model to appropriate device (on GPU if available else CPU)
    device = torch.device(config.DEVICE)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay],
            'weight_decay':0.001},
        {'params': [p for n, p in param_optimizer if n in no_decay],
            'weight_decay':0.0}
    ]

    num_train_steps = int(
        (len(df_train)/config.TRAIN_BATCH_SIZE)*config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS if not args['epochs'] else args['epochs']):
        engine.train_fn(train_data_loader, model, optimizer, scheduler, device)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)

        # calculating evaluation metrics
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy score = {accuracy}")

        # saving model
        if accuracy > best_accuracy:
            if args['save_epochs']:
                torch.save(model.state_dict(), config.MODEL_PATH.format(epoch+1))
            else:
                torch.save(model.state_dict(), config.MODEL_PATH.format(''))
            best_accuracy = accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--samples',
                        type=int,
                        help='Number of samples to take from the original dataframe (before train-valid split)')

    parser.add_argument('-e', '--epochs',
                        type=int,
                        help='Number of epochs to train for')

    parser.add_argument('-se', '--save_epochs',
                        action='store_true',
                        help='If true, saves the weights for each epoch (only if accuracy > best_accuracy')
    

    args = parser.parse_args()

    args = vars(args)
    run(args)