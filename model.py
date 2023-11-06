from utils import get_vocabs, TitleTagObject, read_in_data
import torch
import torch.nn as nn
import numpy as np

class Data():
    def __init__(self, data_fpaths):
        self.title_train, self.title_test, self.tag_train, self.tag_test = read_in_data(data_fpaths)
        self.token2idx, self.tag2idx, self.title_lengths, self.tag_count = get_vocabs(self.title_train, self.tag_train, vocab_dir="train")
        self.train_vocab = self.token2idx.keys()

    def compile_data(self, mode="train"):
        if mode == "train":
            title_tag_objects = []
            for title, tag in zip(self.title_train, self.tag_train):
                lowered_split_title = title.lower().strip().split()
                lowered_split_title = [token for token in lowered_split_title if token in self.train_vocab]
                title_idxs = np.array([self.token2idx[token] for token in lowered_split_title])
                tag_idx = self.tag2idx[tag.rstrip()]
                title_tag_objects.append(TitleTagObject(title_idxs=title_idxs, tag_idx=tag_idx, title_length=len(lowered_split_title)))

            return title_tag_objects
        
        elif mode == "test":
            test_token2idx, _, _, _ = get_vocabs(self.title_test, self.tag_test, vocab_dir="test")

            test_vocab = test_token2idx.keys()
            unk_tokens = [token for token in test_vocab if token not in self.train_vocab]
            print("No. of UNK tokens:", len(unk_tokens))

            title_tag_objects = []
            for title, tag in zip(self.title_test, self.tag_test):
                lowered_split_title = title.lower().strip().split()
                lowered_split_title = [token for token in lowered_split_title if token in self.train_vocab]
                title_idxs = np.array([self.token2idx[token] for token in lowered_split_title])
                tag_idx = self.tag2idx[tag.strip()]
                title_tag_objects.append(TitleTagObject(title_idxs=title_idxs, tag_idx=tag_idx, title_length=len(lowered_split_title)))

            return title_tag_objects

    
    def collate_fn(self, batch):
        longest_title = max(self.title_lengths)
        pad_value = len(self.train_vocab) + 1

        titles = np.stack([np.pad(
            title.title_idxs, (0, np.abs(longest_title - title.title_length)), mode='constant', constant_values=pad_value) for title in batch])
        # offsets = [len(title) for title in batch]
        tags = [int(tag.tag_idx) for tag in batch]

        titles = torch.LongTensor(titles)
        # offsets = torch.tensor(offsets[:-1]).cumsum(dim=0) # TODO: write documentation
        tags = torch.Tensor(tags)
        tags = torch.Tensor.int(tags)

        return titles, tags


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_tags):
        super(Model, self).__init__()
        self.vocab_size = len(vocab_size)
        self.embedding_dim = embedding_dim
        self.num_tags = num_tags
        self.pad_idx = self.vocab_size + 1
        num_embeddings = self.vocab_size + 2
        self.embedding_bag = nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=self.embedding_dim, mode='sum', padding_idx=self.pad_idx, sparse=False)
        self.linear = nn.Linear(self.embedding_dim, self.num_tags)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.5
        self.embedding_bag.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, input, offsets):
        embeddings = self.embedding_bag(input, offsets)
        return self.linear(embeddings)
