from keras.datasets import imdb
import numpy as np
from keras import models,layers
import matplotlib.pyplot as plt


def vector_sequences(sequences, dimension = 10000):
    result = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences) :
        result[i,sequence] = 1.
    return result

(trian_data, trian_lables),(test_data,test_lables) = imdb.load_data(num_words = 10000)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value , key) for (key, value) in word_index.items()])
decode_review = " ".join([reverse_word_index.get(i-3,'?') for i in trian_data[0]])
print(decode_review)

x_trian = vector_sequences(trian_data)
x_test = vector_sequences(test_data)

y_train = np.asarray((trian_lables).astype('float32'))
y_test = np.asarray((test_lables).astype('float32'))

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
x_val = x_trian[:10000]
p_x_trian = x_trian[10000:]

y_val = y_train[:10000]
p_y_trian = y_train[10000:]

history = model.fit(p_x_trian,
                    p_y_trian,
                    epochs = 7,
                    batch_size = 512,
                    validation_data = (x_val,y_val))

history_dict = history.history
print(history_dict.keys())
loss_value = history_dict['loss']
val_loss_values = history_dict['val_loss']

epoch = range(1,len(loss_value)+1)

plt.plot(epoch,loss_value,'bo',label= "Training loss")
plt.plot(epoch,val_loss_values,'b',label= "Validation loss")
plt.title("Training and validtion loss")
plt.xlabel("Epochs")
plt.ylabel('Loss')

plt.show()

loss, acc = model.evaluate(x_test,y_test)
print(loss,acc)

