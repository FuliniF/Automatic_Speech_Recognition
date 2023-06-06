import matplotlib.pyplot as plt

def ploting(train_accuracy, train_loss, test_accuracy, test_loss):
    plt.plot(train_accuracy, label = 'accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy (%)')
    plt.title('Train Accuracy (LSTM)')
    plt.savefig("./Plots/train_acc_lstm.png")
    plt.legend()
    plt.close()

    plt.plot(train_loss, label = 'loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Train Loss (LSTM)')
    plt.savefig("./Plots/train_loss_lstm.png")
    plt.legend()
    plt.close()

    plt.plot(test_accuracy, label = 'accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy (%)')
    plt.title('Test Accuracy (LSTM)')
    plt.savefig("./Plots/test_acc_lstm.png")
    plt.legend()
    plt.close()

    plt.plot(train_loss, label = 'loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Test Loss (LSTM)')
    plt.savefig("./Plots/test_loss_lstm.png")
    plt.legend()
    plt.close()
    
    return