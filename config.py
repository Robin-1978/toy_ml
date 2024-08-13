# model = LSTMModel.LSTMModel(input_size=16, hidden_size=32, num_classes=16, num_layers=2, dropout=0.0)
class LSTMHyperParameters:
    def __init__(self):
        self.epochs = 1000
        self.batch_size = 64
        self.learning_rate = 0.001

        self.input_size = 16
        self.hidden_size = 32
        self.num_classes = 16
        self.num_layers = 2
        self.dropout = 0.0

class LSTMRedBallHyperParameters:
    def __init__(self):
        self.epochs = 1000
        self.batch_size = 64
        self.learning_rate = 0.001

        self.input_size = 33 *6
        self.hidden_size = 32
        self.num_classes = 33 * 32 * 31
        self.num_layers = 2
        self.dropout = 0.0