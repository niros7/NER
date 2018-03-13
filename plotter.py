import matplotlib.pyplot as plt


def plot_training_loss(train_history, eval_histories):
    y_train_loss = [t[1] for t in train_history]
    y_eval_loss = [t[1] for t in eval_histories]
    plt.plot(y_train_loss)
    plt.plot(y_eval_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('batch')
    plt.legend(['eval', 'train'], loc='upper left')
    plt.show()
