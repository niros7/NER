import utils

def train(model, epochs, data, optimizer):
    model.train()
    datalength = len(data)
    training_data_length = int(datalength * 0.8)
    trainingdata = data[0:training_data_length]
    validationdata = data[training_data_length:]
    for epoch in range(epochs):
        loss_sum = 0
        for idx, curr_trainingdata in enumerate(trainingdata):
            sentence, tags = curr_trainingdata
            model.zero_grad()
            neg_log_likelihood = model.neg_log_likelihood(sentence, tags)
            loss_sum += neg_log_likelihood
            avg_loss= loss_sum / idx+1
            print("epoch = %d, train= %d/%d ,avg_loss = %f" % (epoch, idx, training_data_length, avg_loss))
            neg_log_likelihood.backward()
            optimizer.step()
        validate(model,validationdata)
        # utils.save_checkpoint(model,epoch,avg_loss);


def validate(model,data):
    model.eval()
    loss_sum=0
    data_length = len(data)
    for idx, curr_data in enumerate(data):
        sentence,tags = curr_data
        model.zero_grad()
        neg_log_likelihood = model.neg_log_likelihood(sentence, tags)
        loss_sum += neg_log_likelihood
        avg_loss = loss_sum / idx+1
        print("Validation  train= %d/%d ,avg_loss = %f" % (idx, data_length, avg_loss))