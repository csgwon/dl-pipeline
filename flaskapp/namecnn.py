from tools import *
import torch
import numpy as np

import sys

charcnn = torch.load('/var/www/html/flaskapp/charcnn.pth')

charcnn.train(False)

def predict(name):
    from torch.autograd import Variable
    name = encode_input(name)
    name = Variable(torch.from_numpy(name).float())
    name = name.view(1,-1,max_name_len)
    preds = charcnn(name)
    top_pred, index = torch.max(preds, dim=1)
    return labels[index.data.tolist()[0]]

def predict2(num):
    return str(np.sqrt(float(num)))

#print(predict(sys.argv[1]))
