import transformers
import torch

# Network Configurations
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10

# Model
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

# Inputs Paths
TRAINING_FILE = "../input/imdb.csv"

# Output Paths
MODEL_PATH = "../weights/model{}.bin"  # The path to store the trained weights

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    print('GPU found !!')