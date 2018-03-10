from utils import *
import pandas as pd
import torch

file_name = "ner_dataset.csv"


def load_data(from_row, to_row):
    dframe = pd.read_csv(file_name, encoding="ISO-8859-1", error_bad_lines=False)
    df = dframe[from_row:to_row]
    start_of_sentences_idx = df['Sentence #'].dropna().index
    word_to_idx = load_word_to_idx()
    tag_to_ix = load_tag_to_idx()
    training_data = []

    for curr_idx in range(len(start_of_sentences_idx) - 1):
        curr_sentence_idx = start_of_sentences_idx[curr_idx]
        next_sentence_idx = start_of_sentences_idx[curr_idx + 1]
        sentence = df[['Word']][curr_sentence_idx:next_sentence_idx - 1].values.ravel()
        tags = df[['Tag']][curr_sentence_idx:next_sentence_idx - 1].values.ravel()

        sentence = prepare_sequence(sentence, word_to_idx)
        tags = torch.LongTensor([tag_to_ix[t] for t in tags])

        training_data.append((sentence, tags))

    return training_data


def load_tag_to_idx():
    df = pd.read_csv(file_name, encoding="ISO-8859-1", error_bad_lines=False)
    all_tags = df['Tag'].unique().ravel()
    tag_to_idx = {START_TAG: 0, STOP_TAG: 1}
    for tag in all_tags:
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)

    return tag_to_idx


def load_word_to_idx():
    df = pd.read_csv(file_name, encoding="ISO-8859-1", error_bad_lines=False)
    all_words = df['Tag'].unique().ravel()
    word_to_idx = {}
    for word in all_words:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

    return word_to_idx
