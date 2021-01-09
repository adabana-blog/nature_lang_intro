from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dot, Flatten, Embedding, LSTM


class LSTMModel:

    def __init__(self, input_dim, output_dim,
                 emb_dim=300, hid_dim=100,
                 embeddings=None, trainable=True):
        self.input = Input(shape=(None,), name="Input")
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       mask_zero=True,
                                       trainable=trainable,
                                       name="embedding")
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       make_zero=True,
                                       trainable=trainable,
                                       weights=[embeddings],
                                       name="embedding")
        self.lstm = LSTM(hid_dim, name="rnn")
        self.fc = Dense(output_dim, activation="softmax")

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        output = self.lstm(embedding)
        y = self.fc(output)
        return Model(inputs=x, outputs=y)
