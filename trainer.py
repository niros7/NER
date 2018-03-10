def train(model, epochs, data, optimizer):
    training_data_length = len(data)
    for epoch in range(epochs):
        loss_sum = 0
        for idx, data in enumerate(data):
            sentence, tags = data
            model.zero_grad()
            neg_log_likelihood = model.neg_log_likelihood(sentence, tags)
            loss_sum += neg_log_likelihood
            print("epoch = %d, train= %d/%d ,avg_loss = %f" % (epoch, idx, training_data_length, loss_sum / idx))
            neg_log_likelihood.backward()
            optimizer.step()
