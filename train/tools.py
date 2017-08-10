# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

name_data = pd.read_csv('data/names/names_train_new.csv', sep='\t')
name_data = name_data.dropna()

labels = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 'Greek', 'Irish', 'Italian', 'Japanese', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese']
label_to_number = {y: i for i, y in enumerate(labels)}

chars = 'abcdefghijklmnopqrstuvwxyz-,;!?:\'\\|_@#$\%^&*˜‘+-=<>()[]{} '
char_to_index = {char:i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

max_name_len = int(max(name_data['name'].apply(len)))

def add_begin_end_tokens(name):
    # return "^{}$".format(name)
    begin_token_marker = "^"
    end_token_marker = '$'
    tokened_name = "".join((begin_token_marker, name, end_token_marker))
    return tokened_name

def encode_input(name, maxlen=max_name_len):
    name = add_begin_end_tokens(name.lower().strip())
    encoding = np.zeros((len(chars), maxlen), dtype=np.int64)
    for i, char in enumerate(name[:maxlen]):
        index = char_to_index.get(char, 'unknown')
        if index is not 'unknown':
            encoding[index,i] = 1
    return encoding

def decode_input( encoding, maxlen=max_name_len ):
    name = ''
    for i in range(maxlen):
        idx = np.nonzero(encoding[:,i])
        if len(idx[0]) > 0:
            enc_char = index_to_char.get(idx[0][0], 'unknown')
            if enc_char is not 'unknown':
                name += enc_char
    return name

