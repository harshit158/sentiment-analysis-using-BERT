import torch
import torch.nn as nn
from tqdm import tqdm

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train_fn(data_loader, model, optimizer):
    model.train()

    for idx, data in tqdm(enumerate(data_loader), total = len(data_loader)):
        # getting input format from raw data
        ids = data['ids']
        mask = data['attention_mask']
        token_type_ids = data['token_type_ids']
        targets = data['target']

        optimizer.zero_grad()

        # running forward pass
        outputs = model(
            ids = ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )

        # calculating loss function
        loss = loss_fn(outputs, targets)
        loss.backward()

        # stepping in the right direction
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model):
    model.eval()
    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for idx, data in tqdm(enumerate(data_loader), total = len(data_loader)):
            # getting input format from raw data
            ids = data['ids']
            mask = data['attention_mask']
            token_type_ids = data['token_type_ids']
            targets = data['target']

            # running forward pass
            outputs = model(
                ids = ids,
                attention_mask = mask,
                token_type_ids = token_type_ids
            )

            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    
    return final_outputs, final_targets