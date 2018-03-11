import torch
from torch import autograd
from datetime import date

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 4


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def load_checkpoint(filename, model = None):
    print("loading model...")
    checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch


def save_checkpoint(model, epoch, loss):
    print("saving model...")
    d = date.today()
    filename= d.strftime("%d%m%y")
    checkpoint = {}
    checkpoint["state_dict"] = model.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["loss"] = loss
    save_file_name = filename + ".epoch%d" % epoch
    torch.save(checkpoint,save_file_name )
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
