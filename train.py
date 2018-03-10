import sys
import re
from model import *
from utils import *
from os.path import isfile
import pandas as pd


def load_data():
    dframe = pd.read_csv("ner_dataset.csv", encoding="ISO-8859-1", error_bad_lines=False)

    df = dframe[:5000]
    start_of_sentences_idx = df['Sentence #'].dropna().index

    training_data = []
    all_tags = df['Tag'].unique().ravel()

    for curr_idx in range(len(start_of_sentences_idx) - 1):
        curr_sentence_idx = start_of_sentences_idx[curr_idx]
        next_sentence_idx = start_of_sentences_idx[curr_idx + 1]
        sentence = df[['Word']][curr_sentence_idx:next_sentence_idx - 1].values.ravel()
        tags = df[['Tag']][curr_sentence_idx:next_sentence_idx - 1].values.ravel()
        training_data.append((sentence, tags))

    training_data.sort(key=lambda tup: len(tup[0]), reverse=True)
    word_to_idx = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    tag_to_idx = {PAD: 0, EOS: 1, SOS: 2}
    for tag in all_tags:
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)

    data = []
    batch_x = []
    batch_y = []
    batch_len = 40  # maximum sequence length of a mini-batch
    for line in training_data:
        sentence, tags = line
        sentence, tags = [word_to_idx[w] for w in sentence], [tag_to_idx[t] for t in tags]
        seq_len = len(sentence)
        pad = [PAD_IDX] * (batch_len - seq_len)
        batch_x.append(sentence + [EOS_IDX] + pad)
        batch_y.append(tags + [EOS_IDX] + pad)
        if len(batch_x) == BATCH_SIZE:
            data.append((Var(LongTensor(batch_x)), LongTensor(batch_y)))  # append a mini-batch
            batch_x = []
            batch_y = []
    print("data size: %d" % (len(data) * BATCH_SIZE))
    print("batch size: %d" % BATCH_SIZE)
    return data, word_to_idx, tag_to_idx


def train():
    num_epochs = 10
    data, word_to_idx, tag_to_idx = load_data()
    model = lstm_crf(len(word_to_idx), tag_to_idx)
    optim = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    epoch = 0
    if CUDA:
        model = model.cuda()
    filename = "walla"
    print(model)
    print("training model...")
    for i in range(epoch + 1, epoch + num_epochs + 1):
        loss_sum = 0
        for x, y in data:
            model.zero_grad()
            loss = model.loss(x, y)  # forward pass and compute loss
            loss.backward()  # compute gradients
            optim.step()  # update parameters
            loss = scalar(loss)
            loss_sum += loss
        if i % SAVE_EVERY and i != epoch + num_epochs:
            print("epoch = %d, loss = %f" % (i, loss_sum / len(data)))
        else:
            save_checkpoint(filename, model, i, loss_sum / len(data))


if __name__ == "__main__":
    print("cuda: %s" % CUDA)
    train()
