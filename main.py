import torch.optim as optim
from data_loader import *
from model import *
from trainer import *

EPOCHS_NUM = 1

torch.manual_seed(1)

word_to_idx = load_word_to_idx()
tag_to_idx = load_tag_to_idx()
training_data = load_data(0, 40000);
model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters())
training_data_length = len(training_data)
model, train_history, eval_histories = train(model, EPOCHS_NUM, training_data, optimizer)
