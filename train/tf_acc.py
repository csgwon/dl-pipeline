from tools import *
from keras.models import load_model

import sys

def prepare_data(path):
    data = pd.read_csv(path, sep='\t').dropna()  
    X = np.array([encode_input(x) for x in data['name']])
    y = np.array([label_to_number[x] for x in data['label']])
    return X,y

charcnn = load_model('tf_charcnn.h5')

def predict(name):
    x = encode_input(name)
    return labels[np.argmax(charcnn.predict(x[np.newaxis,:,:]))]


def acc(path):
    data = pd.read_csv(path, sep='\t').dropna()  
    out = np.array([label_to_number[predict(name)] for name in data['name']])
    y = np.array([label_to_number[x] for x in data['label']])
    diff = out-y
    return 1.0-np.count_nonzero(diff)/float(len(diff))

if __name__ == '__main__':
    print(str(acc(sys.argv[1])))
