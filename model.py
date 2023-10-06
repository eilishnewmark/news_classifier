from utils import split_data, get_vocabs, TitleTagObject
import torch
import torch.nn as nn
import numpy as np

def load_data(title_fpath, tag_fpath):
    title_train, title_test, tag_train, tag_test = split_data(title_fpath, tag_fpath)
    token2idx, tag2idx, idx2tag, vocab_size, tag_vocab_size, vocab_counts, tag_counts = get_vocabs(title_train, tag_train)

    batch_size = 32
    n_iters = 280
    epochs = int(n_iters / (len(title_train) / batch_size)) # 10

    title_tag_objects = []
    for title, tag in zip(title_train, tag_train):
        lowered_title = title.lower()
        split_title = lowered_title.strip().split()
        title_length = len(split_title)
        title_idxs = np.array([token2idx[token] for token in split_title])
        tag_idx = tag2idx[tag.strip()]
        title_tag_objects.append(TitleTagObject(title_idxs=title_idxs, tag=tag_idx, title_length=title_length))

    train_loader = torch.utils.data.DataLoader(dataset=title_tag_objects, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=title_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return title_tag_objects, train_loader, test_loader, token2idx, epochs

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.non_linear = nn.Tanh()
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        linear1_out = self.input_linear(x) 
        non_linear_out = self.non_linear(linear1_out)
        output = self.output_linear(non_linear_out)
        return output

def compute_loss(output, tag_targets):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, tag_targets)
    return loss

def collate_fn(batch, pad_value=None):

    title_lengths = np.array([title.title_length for title in batch])
    longest_title = max(title_lengths)
    # calculate the number of pads required for each title
    no_pads_array = np.abs(title_lengths - longest_title)

    padded_batch = []

    for title, no_pads in zip(batch, no_pads_array):
        idxs = title.title_idxs
        padded_batch.append(idxs + no_pads * pad_value)

    return torch.LongTensor(np.array(padded_batch)), longest_title