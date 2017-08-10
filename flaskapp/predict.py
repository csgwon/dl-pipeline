from tools import *
from model import *
import torch
from torch.autograd import Variable

import sys

def predict(name):
    name = Variable(torch.from_numpy(encode_input(name)).float())
    #name = Variable(test_batch['X'][0])
    name = name.view(1,-1,max_name_len)
    preds = charcnn(name)
    top_pred, index = torch.max(preds, dim=1)
    return labels[index.data.tolist()[0]]


charcnn = CharCNN(n_classes=len(labels), vocab_size=len(chars), max_seq_length=max_name_len)
charcnn.load_state_dict(torch.load('charcnn.pth'))

charcnn.train(False)

print(predict(sys.argv[1]))
