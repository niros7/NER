import torch
from torch import autograd
from datetime import date
import data_loader

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100


def convert_to_tags(result):
    tags = []
    tag_to_idx = data_loader.load_tag_to_idx()
    for numT in result[1]:
        tags.extend(key for key, value in tag_to_idx.items() if value == numT)
    return tags


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_idx):
    word_tensor = []
    for word in seq:
        if word not in to_idx:
            to_idx[word] = len(to_idx)
        word_tensor.append(to_idx[word])

    tensor = torch.LongTensor(word_tensor)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def load_checkpoint(filename, model=None):
    print("loading model...")
    checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (epoch, loss))
    return epoch, checkpoint["train"], checkpoint["eval"]


def save_checkpoint(model, epoch, loss, train_history, eval_histories):
    print("saving model...")
    d = date.today()
    filename = d.strftime("%d%m%y")
    checkpoint = {}
    checkpoint["state_dict"] = model.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["loss"] = loss
    checkpoint["train"] = train_history
    checkpoint["eval"] = eval_histories
    save_file_name = filename + ".epoch%d" % epoch
    torch.save(checkpoint, save_file_name)

    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
