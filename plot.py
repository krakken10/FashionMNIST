import matplotlib.pyplot as plt

class Plot:
    def __init__(self, History):
        self.History = History
    
    def loss(self):
        plt.plot(self.History.history['loss'], label='loss')
        plt.plot(self.History.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    def accuracy(self):
        plt.plot(self.History.history['acc'], label='acc')
        plt.plot(self.History.history['val_acc'], label='val_acc')
        plt.legend()
        plt.show()