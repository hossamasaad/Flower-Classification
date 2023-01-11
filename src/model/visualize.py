import matplotlib.pyplot as plt


def draw_loss(history):
    loss_train = history.history['train_loss']
    loss_val = history.history['val_loss']
    epochs = len(loss_train)

    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def draw_accuracy(history):
    acc_train = history.history['acc']
    acc_val = history.history['val_acc']
    epochs = len(acc_train)

    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, acc_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()