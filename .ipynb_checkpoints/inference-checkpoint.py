import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class InferenceAPI:

    def __init__(self, model, vocab, preprocess):
        self.vocab = vocab
        self.model = model
        self.preprocess = preprocess

    def predict_from_texts(self, texts):
        x = self.preprocess(texts)
        x = self.vocab.texts.to_sequences(x)
        return self.predict_from_sequence(x)

    def predict_from_sequence(self, sequences):
        sequences = pad_sequences(sequences, truncating="post")
        y = self.model.predict(sequences)
        return np.argmax(y, -1)
