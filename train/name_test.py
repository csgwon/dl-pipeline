from tools import *
from model import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

class NamesDataset(Dataset):
    """Name Classification dataset"""
    def __init__(self, path):
        self.data = pd.read_csv(path, sep='\t').dropna()
        self.X = self.data['name']
        self.y = self.data['label']
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        content = torch.from_numpy(encode_input(self.data['name'][index])).float()
        label = label_to_number[self.data['label'][index]]
        sample = {'X': content, 'y': label}
        return sample

name_dataset = NamesDataset('data/names/names_train_new.csv')

dataloader = DataLoader(name_dataset, batch_size=32, shuffle=True, num_workers=0)

charcnn = CharCNN(n_classes=len(set(name_data['label'])), vocab_size=len(chars), max_seq_length=max_name_len)

criterion = nn.CrossEntropyLoss()

from tqdm import tqdm_notebook

def train(model, dataloader, num_epochs):
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_history_avg = []
    loss_history = []
    #bar = tqdm_notebook(total=len(dataloader))
    for i in range(num_epochs):
        per_epoch_losses = []
        for batch in dataloader:
            X = Variable(batch['X'])
            y = Variable(batch['y'])
            if cuda:
                X = X.cuda()
                y = y.cuda()
            model.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            per_epoch_losses.append(loss.data[0])
            #bar.set_postfix(loss=loss.data[0])
            #bar.update(1)
        loss_history_avg.append(np.mean(per_epoch_losses))
        loss_history.append( loss.data[0] )
        print('epoch[%d] loss: %.4f' % (i, loss.data[0]))
    return loss_history, loss_history_avg

loss_history, loss_history_avg = train(charcnn, dataloader, 100)

torch.save(charcnn, 'charcnn.pth')
