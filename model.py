from utils import split_data, get_vocabs
import torch
import torch.nn as nn
import numpy as np

def load_data(title_fpath, tag_fpath):
    title_train, title_test, tag_train, tag_test = split_data(title_fpath, tag_fpath)

    token2idx, tag2idx, idx2tag, vocab_size, tag_vocab_size, vocab_counts, tag_counts = get_vocabs(title_train, tag_train)

    batch_size = 32
    n_iters = 280
    epochs = n_iters / (len(title_train) / batch_size) # 10

    train_loader = torch.utils.data.DataLoader(dataset=title_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=title_test, batch_size=batch_size, shuffle=True)

load_data("data/tokenised_titles_without_punctuation.txt", "data/tags.txt")

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.non_linear = nn.Tanh()
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        pass



def collate_fn(batch, pad_value):

    title_lengths = np.array([len(title) for title in batch])
    longest_title = np.max(title_lengths)
    # calculate the number of pads required for each title
    no_pads_array = np.abs(title_lengths - longest_title)

    padded_batch = []

    for title, no_pads in zip(batch, no_pads_array):
        padded_batch.append(np.pad(title, no_pads, 'constant', constant_values=pad_value))

    return padded_batch