from tools import *
from model import *
import torch
from torch.autograd import Variable
import numpy as np

import sys

charcnn = torch.load('charcnn.pth')
charcnn = charcnn.cuda()
charcnn.train(False)

def predict(name):
    name = Variable(torch.from_numpy(encode_input(name)).float())
    #name = Variable(test_batch['X'][0])
    name = name.view(1,-1,max_name_len)
    name = name.cuda()
    preds = charcnn(name)
    top_pred, index = torch.max(preds, dim=1)
    return labels[index.data.tolist()[0]]

label_to_number = {y: i for i, y in enumerate(set(name_data['label']))}

data = pd.read_csv(sys.argv[1], sep='\t')
data = name_data.dropna()

X = data['name']
y = np.array([label_to_number[l] for l in data['label']])

predictions = [label_to_number[predict(x)] for x in X]
diff = np.array(predictions)-y

print(str(1.0-np.count_nonzero(diff)/float(len(X))))
