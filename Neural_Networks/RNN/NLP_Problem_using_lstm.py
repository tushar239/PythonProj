# LSTM for NLP — Sentiment classification on IMDB
# Prereq: pip install tensorflow

'''
Quick explanation (important bits)

Data shape after padding: (samples, maxlen) where maxlen = number of timesteps the LSTM will see.

Embedding layer turns each integer token into a dense vector of size embedding_dim. After embedding, each sample shape is (maxlen, embedding_dim) → that’s (timesteps, features) for the LSTM.

LSTM(units=128) produces a last hidden state of shape (128,) per sample (unless return_sequences=True).

Dense(1, activation='sigmoid') is used for binary sentiment (0/1).

Use binary_crossentropy for binary classification with sigmoid output.

'''
'''
Variations & tips

Stacked LSTM: use LSTM(..., return_sequences=True) for intermediate layers, then another LSTM after that.

Pretrained embeddings (GloVe): load embedding matrix and pass to Embedding(..., weights=[embedding_matrix], trainable=False) for better performance.

Bidirectional LSTM: wrap with tf.keras.layers.Bidirectional(LSTM(...)) to capture both directions.

Long sequences: increase maxlen carefully (memory increases); consider truncating or using attention/Transformers for very long text.

Regularization: dropout on LSTM (dropout, recurrent_dropout) and Dense helps avoid overfitting.

Batch size: larger batches give smoother gradients, smaller batches are noisier but can generalize better; try 32/64/128.

Tokenization alternative: instead of imdb.load_data, use Tokenizer() if you have raw text dataset.
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1) Parameters
vocab_size = 20000      # top N words to keep
maxlen = 200            # max tokens per review (timesteps)
embedding_dim = 100     # embedding size (features per timestep)
lstm_units = 128
batch_size = 64
epochs = 10

# 2) Load IMDB dataset (already tokenized as integer word indices)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 3) Pad/truncate sequences so every sample has length = maxlen
x_train = pad_sequences(x_train, maxlen=maxlen, padding='pre', truncating='pre')
x_test  = pad_sequences(x_test,  maxlen=maxlen, padding='pre', truncating='pre')

print("Shapes:", x_train.shape, y_train.shape)   # (num_samples, maxlen), (num_samples,)

# 4) Build model: Embedding -> LSTM -> Dense
#    Input shape to LSTM will be (timesteps=maxlen, features=embedding_dim)
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
    LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.1),   # returns last hidden state by default
    Dropout(0.3),
    Dense(1, activation='sigmoid')   # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5) Train with early stopping
es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
history = model.fit(
    x_train, y_train,
    validation_split=0.15,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[es],
    verbose=2
)

# 6) Evaluate on test set
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

# 7) Predict on new raw text (example)
# If you want to predict raw text you must use the same tokenization/word index mapping.
word_index = imdb.get_word_index()

def encode_text(text, word_index, vocab_size=vocab_size, maxlen=maxlen):
    # simple whitespace tokenizer + index mapping (lowercasing)
    tokens = text.lower().split()
    seq = [word_index.get(w, 2) for w in tokens]  # 2 = unknown in IMDB mapping convention
    return pad_sequences([seq], maxlen=maxlen, padding='pre', truncating='pre')

sample_text = "This movie was fantastic — I loved it and the acting was great!"
x_sample = encode_text(sample_text, word_index)
pred = model.predict(x_sample)[0,0]
print(f"Predicted sentiment probability (positive): {pred:.3f}") # Predicted sentiment probability (positive): 0.290

# Save model (optional)
# model.save("lstm_imdb_model.h5")
