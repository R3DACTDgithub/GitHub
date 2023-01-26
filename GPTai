import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM
from keras.models import Sequential

# Initialize an empty list to store user input
user_input = []

# Ask the user to enter a sentence
sentence = input("Enter a sentence: ")

# Add the sentence to the list of user input
user_input.append(sentence)

# Tokenize the input and create a vocabulary index
tokenizer = Tokenizer()
tokenizer.fit_on_texts(user_input)
vocab_size = len(tokenizer.word_index) + 1

# Convert the sentences to numerical sequences
sequences = tokenizer.texts_to_sequences(user_input)

# Pad the sequences to the same length
padded_sequences = pad_sequences(sequences)

# Create the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=padded_sequences.shape[1]))
model.add(LSTM(100))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model on the user input
model.fit(padded_sequences, padded_sequences, epochs=100)

# Generate a new sentence based on what the model has learned
generated_sentence = model.predict(padded_sequences)
print(generated_sentence)
