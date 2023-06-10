from keras import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.optimizers import Adam
from transformers import AutoModel, AutoTokenizer
import torch

def gendermodel():
    model_name = "bertmodel"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    input_text = "Example sentence to encode"

    # Tokenize input text
    tokens = tokenizer.tokenize(input_text)

    # Convert token to ids and add padding
    max_length = 16
    padding = "max_length"
    token_ids = tokenizer.encode_plus(tokens, max_length=max_length, padding=padding, return_tensors='pt')

    # Pass token ids through the model to get embeddings
    with torch.no_grad():
        embeddings = model(**token_ids).last_hidden_state
    model = Sequential([
        AutoModel.
        Bidirectional(LSTM(units=128, recurrent_dropout=0.2, dropout=0.2)),
        Dense(1, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    return model