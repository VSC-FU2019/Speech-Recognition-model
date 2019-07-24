import os
import time
from random import shuffle as shuf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import wavfile
from python_speech_features import mfcc, delta
from sklearn.metrics import confusion_matrix, classification_report

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, Sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout, Input, Bidirectional
from keras.models import Model, load_model

class DataGenerator(Sequence):
    def __init__(self, df, feature=13, max_frames=200, batch_size=2, shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(df.shape[0])
        self.max_frames = max_frames
        self.classes = []

    def __len__(self):
        return int(np.floor(self.df.shape[0] / self.batch_size))

    def load_data(self, indices_in_batch):
        paths = []
        for i in indices_in_batch:
            path = self.df[i]
            paths.append(path)

        return paths

    def extract_feature(self, path):
        fs, y = wavfile.read(path)
        y = y / np.max(abs(y))
        mfcc_feat = mfcc(y, fs)
        mfcc_feat = delta(mfcc_feat, 2)
        data = pad_sequences(mfcc_feat.T, self.max_frames, dtype=float, padding='post', truncating='post').T
        if(path.split('/')[-2] == '0.Background'):
            target = 0
        elif (path.split('/')[-2] == '1.Bat den'):
            target = 1
        elif (path.split('/')[-2] == '2.Tat den'):
            target = 2
        elif (path.split('/')[-2] == '3.Bat dieu hoa'):
            target = 3
        elif (path.split('/')[-2] == '4.Tat dieu hoa'):
            target = 4
        elif (path.split('/')[-2] == '5.Bat quat'):
            target = 5
        elif (path.split('/')[-2] == '6.Tat quat'):
            target = 6
        elif (path.split('/')[-2] == '7.Bat tivi'):
            target = 7
        elif (path.split('/')[-2] == '8.Tat tivi'):
            target = 8
        elif (path.split('/')[-2] == '9.Mo cua'):
            target = 9
        elif (path.split('/')[-2] == '10.Dong cua'):
            target = 10
        elif (path.split('/')[-2] == '11.Khoa cua'):
            target = 11
        elif (path.split('/')[-2] == '12.Mo cong'):
            target = 12
        elif (path.split('/')[-2] == '13.Dong cong'):
            target = 13
        elif (path.split('/')[-2] == '14.Khoa cong'):
            target = 14
        elif (path.split('/')[-2] == '15.Doremon'):
            target = 15

        return data, target

    def __getitem__(self, batch_index):
        indices_in_batch = self.indices[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        if self.shuffle:
            shuf(indices_in_batch)

        x_data = np.zeros((self.batch_size, self.max_frames, self.feature))
        targets = []
        paths = self.load_data(indices_in_batch)
        for count, i in enumerate(indices_in_batch):
            frames, label = self.extract_feature(i)
            self.classes.append(label)
            targets.append(label)
            x_data[count] = frames
        
        outputs = np.asarray(targets)
        return x_data, outputs


def build_model():
    input_data = Input(shape=(None, 13), name='the_input')
    x = Bidirectional(LSTM(32,return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(input_data)
    x = Bidirectional(LSTM(32,dropout=0.2, recurrent_dropout=0.2, return_sequences=True)) (x)
    x = Bidirectional(LSTM(32,dropout=0.2, recurrent_dropout=0.2)) (x)
    x = Dense(64)(x)
    x = Dense(64)(x)
    output = Dense(16, activation = 'softmax')(x)
    model = Model(inputs=[input_data], outputs=[output])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['acc'])
    return model
            
print('Hello VSC')
model = build_model()
model.summary()

np.random.seed(312)
num_epochs = 5
batch_size = 2

path = 'dataset/train'
data_train = []

for r, d, f in os.walk(path):
    for _f in f:
        data_train.append(os.path.join(r, _f))

shuf(data_train)

path = 'dataset/validate'
data_validate = []

for r, d, f in os.walk(path):
    for _f in f:
        data_validate.append(os.path.join(r, _f))

shuf(data_validate)

path = 'dataset/test'
data_test = []

for r, d, f in os.walk(path):
    for _f in f:
        data_test.append(os.path.join(r, _f))

shuf(data_test)

##################### save to CSV ######################
df_train = pd.DataFrame(data_train, column=['file'])
#df_train.to_csv('data_train.csv', index=None)
df_validate = pd.DataFrame(data_validate, column=['file'])
#df_validate.to_csv('data_val.csv', index=None)
df_test = pd.DataFrame(data_test, column=['file'])
#df_test.to_csv('data_test.csv', index=None)

training_generator = DataGenerator(df_train)
validate_generator = DataGenerator(df_validate)
t = time.time()
######################## TRAINING ###########################
print('Training')
step_p_epoch = training_generator.__len__()
history = model.fit_generator(generator=training_generator, steps_per_epoch=step_p_epoch, epochs=num_epochs, verbose=1, validation_data=validate_generator)

model.save('models/' + str(t) + '.h5')
print('Saved model')
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
x_epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.plot(epochs, acc, 'ro', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('reports/Graph ' + str(t) + '.png')
print('Saved graph image')

test_generator = DataGenerator(df_test)
Y_pred = model.predict_generator(test_generator, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
y_true = np.asarray(test_generator.classes)[:y_pred.shape[0]]
class_names = ['background','bat den', 'tat den', 'bat dieu hoa', 'tat dieu hoa', 'bat quat', 'tat quat', 'bat tivi', 'tat tivi', 'mo cua', 'dong cua', 'khoa cua', 'mo cong', 'dong cong' , 'khoa cong' , 'doremon']
cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),xticklabels=class_names, yticklabels=class_names,
                            title='Confusion Matrix', ylabel='True label', xlabel='Predicted label')

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fmt = '.2f'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.savefig('reports/Confusion matrix ' + str(t) + '.png')

#print(classification_report(y_true, y_pred, target_names=target_names))
