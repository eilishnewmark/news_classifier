from utils import get_vocabs, TitleTagObject, read_in_data
import torch
import torch.nn as nn
import numpy as np


class Data():
    def __init__(self, data_fpaths):
        self.title_train, self.title_test, self.tag_train, self.tag_test = read_in_data(data_fpaths)
        self.token2idx, self.tag2idx, self.title_lengths, self.vocabsize, self.tag_vocabsize = get_vocabs(self.title_train, self.tag_train)

    def compile_data(self, mode="train"):
        if mode == "train":
            title_tag_objects = []
            for title, tag in zip(self.title_train, self.tag_train):
                lowered_split_title = title.lower().strip().split()
                title_idxs = np.array([self.token2idx[token] for token in lowered_split_title])
                tag_idx = self.tag2idx[tag.strip()]
                title_tag_objects.append(TitleTagObject(title_idxs=title_idxs, tag_idx=tag_idx, title_length=len(lowered_split_title)))

            return title_tag_objects
        
        elif mode == "test":
            test_token2idx, _, _, _, _, = get_vocabs(self.title_test, self.tag_test)

            train_vocab = self.token2idx.keys()
            test_vocab = test_token2idx.keys()
            unk_tokens = [token for token in test_vocab if token not in train_vocab]
            print("No. of UNK tokens:", len(unk_tokens))

            title_tag_objects = []
            for title, tag in zip(self.title_test, self.tag_test):
                lowered_split_title = title.lower().strip().split()
                lowered_split_title = [token for token in lowered_split_title if token in train_vocab]
                title_idxs = np.array([self.token2idx[token] for token in lowered_split_title])
                tag_idx = self.tag2idx[tag.strip()]
                title_tag_objects.append(TitleTagObject(title_idxs=title_idxs, tag_idx=tag_idx, title_length=len(lowered_split_title)))

            return title_tag_objects

    
    def collate_fn(self, batch):
        longest_title = max(self.title_lengths)
        pad_value = -1e9
        titles = np.stack([np.pad(
            title.title_idxs, (0, np.abs(longest_title - title.title_length)), mode='constant', constant_values=pad_value) for title in batch])
        tags = [int(tag.tag_idx) for tag in batch]

        titles = torch.Tensor(titles)
        tags = torch.Tensor(tags)
        tags = torch.Tensor.int(tags)

        return titles, tags

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        torch.nn.init.uniform_(self.input_linear.weight)
        self.non_linear = nn.Sigmoid()
        self.output_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # linear1_out = self.input_linear(x) 
        # non_linear_out = self.non_linear(linear1_out)
        output = self.output_linear(x)
        return output
