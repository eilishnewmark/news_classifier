from utils import get_vocabs, TitleTagObject, read_in_data
import torch
import torch.nn as nn
import numpy as np


class Data():
    def __init__(self, data_fpaths):
        self.title_train, self.title_test, self.tag_train, self.tag_test = read_in_data(data_fpaths)

    def compile_data(self, mode="train"):
        token2idx, tag2idx, _, _, _, _, _ = get_vocabs(self.title_train + self.title_test, self.tag_train + self.tag_test)
        if mode == "train":
            title_data, tag_data = self.title_train, self.tag_train
        elif mode == "test":
            title_data, tag_data = self.title_test, self.tag_test

        title_lengths = []
        title_tag_objects = []
        for title, tag in zip(title_data, tag_data):
            lowered_split_title = title.lower().strip().split()
            title_lengths.append(len(lowered_split_title))
            title_idxs = np.array([token2idx[token] for token in lowered_split_title])
            tag_idx = tag2idx[tag.strip()]
            title_tag_objects.append(TitleTagObject(title_idxs=title_idxs, tag_idx=tag_idx, title_length=len(lowered_split_title)))

        return title_tag_objects, title_lengths
    
    def collate_fn(self, batch):
        token2idx, _, _, _, _, _, _ = get_vocabs(self.title_train, self.tag_train)
        _, train_title_lengths = self.compile_data(mode="train")
        _, test_title_lengths = self.compile_data(mode="test")
        longest_title = max(train_title_lengths + test_title_lengths)
        pad_value = int(token2idx['PAD'])

        titles = np.stack([np.pad(
            title.title_idxs, (0, longest_title - title.title_length), mode='constant', constant_values=pad_value) for title in batch])
        tags = [int(tag.tag_idx) for tag in batch]

        titles = torch.Tensor(titles)
        tags = torch.Tensor(tags)
        tags = torch.Tensor.int(tags)

        return titles, tags

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
