import torch.optim as optim
from data_loader import *
from model import *

torch.manual_seed(1)

word_to_idx = load_word_to_idx()
tag_to_idx = load_tag_to_idx()
training_data = load_data(0, 30000)
eval_data = load_data(30000, 40000)

model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters())
training_data_length = len(training_data)

precheck_sent = prepare_sequence(training_data[0][0], word_to_idx)
print(model(precheck_sent))
