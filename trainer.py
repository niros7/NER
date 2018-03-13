import utils
from random import randint


def train(model, epochs, data, optimizer):
    train_history = []
    eval_histories = []
    data_length = len(data)
    training_data_length = int(data_length * 0.8)
    train_data = data[:training_data_length]
    validation_data = data[training_data_length:]
    for epoch in range(1, epochs + 1):
        model.train(True)
        loss_sum = 0
        for idx, d in enumerate(train_data):
            curr_training_data = train_data[randint(0, training_data_length - 1)]
            sentence, tags = curr_training_data
            model.zero_grad()
            neg_log_likelihood = model.neg_log_likelihood(sentence, tags)
            loss_sum += neg_log_likelihood
            avg_loss = loss_sum / idx + 1
            train_history.append((epoch, avg_loss.data.numpy()[0]))
            print("epoch = %d, train= %d/%d ,avg_loss = %f, curr_loss = %f" % (epoch, idx, training_data_length, avg_loss, neg_log_likelihood))
            neg_log_likelihood.backward()
            optimizer.step()
        eval_losses = validate(model, validation_data)
        eval_histories.append([(epoch, l) for l in eval_losses])
        utils.save_checkpoint(model, epoch, avg_loss, train_history, eval_histories)

    eval_history = []
    for eval_round in eval_histories:
        for epoch, loss in eval_round:
            eval_history.append((epoch, loss))

    return model, train_history, eval_history


def validate(model, eval_data):
    eval_history = []
    model.eval()
    loss_sum=0
    for idx, curr_data in enumerate(eval_data):
        sentence,tags = curr_data
        model.zero_grad()
        neg_log_likelihood = model.neg_log_likelihood(sentence, tags)
        loss_sum += neg_log_likelihood
        avg_loss = loss_sum / idx+1
        eval_history.append(avg_loss.data.numpy()[0])
        print("Validation  train= %d/%d ,avg_loss = %f" % (idx, len(eval_data), avg_loss))

    return eval_history
