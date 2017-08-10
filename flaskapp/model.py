import torch
import torch.nn as nn

class CharCNN(nn.Module):    
    def __init__(self, n_classes, vocab_size, max_seq_length, channel_size=128, pool_size=2):
        super(CharCNN, self).__init__()
        self.conv_stack = nn.ModuleList([nn.Conv1d(vocab_size, channel_size, 3, padding=1), 
                                         nn.ReLU(),
                                         nn.BatchNorm1d(num_features=channel_size),
                                         nn.MaxPool1d(pool_size),
                                         nn.Conv1d(channel_size, channel_size, 3, padding=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(num_features=channel_size),
                                         nn.MaxPool1d(pool_size)])
        self.dropout1 = nn.Dropout(p=0.5)
        self.output = nn.Linear(512, n_classes)  
    def forward(self, x):
        for op in self.conv_stack:
            x = op(x)
        x = x.view(x.size(0),-1)
        x = self.dropout1(x)
        x = self.output(x)
        return x
