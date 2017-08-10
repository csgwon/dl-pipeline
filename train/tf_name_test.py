from tools import *
from tf_model import *
import pandas as pd
import numpy as np
from keras.optimizers import Adam

def prepare_data(path):
    data = pd.read_csv(path, sep='\t').dropna()  
    X = np.array([encode_input(x) for x in data['name']])
    y = np.array([label_to_number[x] for x in data['label']])
    return X,y

def train(model, X, y, num_epochs):
    model.fit(X, y, batch_size=32, shuffle=True, epochs=num_epochs)

if __name__ == '__main__':
    model = build_CNN(n_classes=len(set(name_data['label'])), vocab_size=len(chars), max_seq_length=max_name_len)
    opt = 'sparse_categorical_crossentropy'
    model.compile(optimizer=Adam(lr=1.0e-3), loss=opt, metrics=['accuracy'])
    X, y = prepare_data('data/names/names_train_new.csv')
    #nb_classes = max(y)+1
    #y = np.eye(nb_classes)[np.array([y]).reshape(-1)]
    train(model, X, y, num_epochs=100)
    model.save('tf_charcnn.h5')
