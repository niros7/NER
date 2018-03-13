import torch.optim as optim
from data_loader import *
from model import *
from trainer import *
from plotter import *

if __name__ == "__main__":
    sentence = input("enter the sentence: ")
    word_to_idx = load_word_to_idx()
    tag_to_idx = load_tag_to_idx()
    prepared_sentence = utils.prepare_sequence(sentence, word_to_idx)
    model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
    load_checkpoint("last", model)
    result = model(prepared_sentence)
    result_to_tag_names =

EPOCHS_NUM = 1

torch.manual_seed(1)

word_to_idx = load_word_to_idx()
tag_to_idx = load_tag_to_idx()
training_data = load_data(0, 40000);
model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters())
training_data_length = len(training_data)
model, train_history, eval_histories = train(model, EPOCHS_NUM, training_data, optimizer)

plot_training_loss(train_history, eval_histories)



