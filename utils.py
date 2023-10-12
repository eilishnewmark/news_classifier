from collections import Counter, namedtuple
from sklearn.model_selection import train_test_split
import torch.nn as nn

TitleTagObject = namedtuple("TitleTagObject", ['title_idxs', 'tag_idx', 'title_length'])

def split_data(titles_fpath, tags_fpath, output_fpaths=None):
     # TODO: make so that you can implement a max input length of titles
     with open(titles_fpath, "r") as title_data:
          with open(tags_fpath, "r") as tag_data:
               titles = title_data.readlines()
               tags = tag_data.readlines()
     title_train, title_test, tag_train, tag_test = train_test_split(titles, tags, train_size=0.9)

     with open(output_fpaths["title_train"], "w") as train_titles:
          with open(output_fpaths["tag_train"], "w") as train_tags:
               for title, tag in zip(title_train, tag_train):
                    train_titles.write(title)
                    train_tags.write(tag)
     with open(output_fpaths["title_test"], "w") as test_titles:
          with open(output_fpaths["tag_test"], "w") as test_tags:
               for title, tag in zip(title_test, tag_test):
                    test_titles.write(title)
                    test_tags.write(tag)
     return

def read_in_data(output_fpaths=None):
     with open(output_fpaths["title_train"], "r") as train_titles:
          with open(output_fpaths["tag_train"], "r") as train_tags:
               with open(output_fpaths["title_test"], "r") as test_titles:
                    with open(output_fpaths["tag_test"], "r") as test_tags:
                         title_train = train_titles.readlines()
                         tag_train = train_tags.readlines()
                         title_test = test_titles.readlines()
                         tag_test = test_tags.readlines()
     assert len(title_train) == len(tag_train), "Length of training inputs and targets unequal"
     assert len(title_test) == len(tag_test), "Length of test inputs and targets unequal"
     
     return title_train, title_test, tag_train, tag_test

def get_vocabs(titles, tags, vocab_dir=None):
    """
    titles/tags = list of title/tag strings
    vocab_dir = either 'train', 'test', or None based on whether you want to write vocab \
     and token counts to data_processing/train or data_processing/test directory and None if you don't \
     want to write vocab to file
     """
    # get vocab set and size from all titles
    stripped_lowered_titles = [title.lower().strip() for title in titles]
    all_tokens = []
    title_lengths = []
    for title in stripped_lowered_titles:
         split_title = title.split()
         title_lengths.append(len(split_title))
         for token in split_title:
              all_tokens.append(token)  
     
    # get tag set and size from all tags
    stripped_tags = [tag.strip() for tag in tags]
    tag_vocab = list(set(stripped_tags))
    tag_count = len(tag_vocab)

    # get sorted tuple list of token and tag counts in the data
    vocab_counts = Counter(all_tokens)
    tag_counts = Counter(stripped_tags)
    
    vocab = list(set(all_tokens))
    filtered_vocab = [token for token in vocab if vocab_counts[token] > 2]
    vocab_size = len(filtered_vocab)

    # get dictionaries of token/tag to idx mapping for one hot vectors
    token2idx = {token:idx for idx, token in enumerate(vocab)}
    tag2idx = {tag:idx for idx, tag in enumerate(tag_vocab)}
    idx2tag = {idx:tag for (tag, idx) in tag2idx.items()} 

    if vocab_dir:
     print(f"Total tokens in {vocab_dir} vocab: {vocab_size}\n")
     with open(f"data_processing/vocabs/{vocab_dir}/vocab_counts.csv", "w") as vocab_f:
               vocab_f.write("Token, Count\n")
               for token, count in vocab_counts.most_common():
                    vocab_f.write(f"{token},{count}\n")
     with open(f"data_processing/vocabs/{vocab_dir}/tag_counts.csv", "w") as tags_f:
               tags_f.write("Token, Count\n")
               for tag, count in tag_counts.most_common():
                    tags_f.write(f"{tag},{count}\n")

    return token2idx, tag2idx, title_lengths, tag_count

def compute_loss(output, tag_targets):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, tag_targets.long())
    return loss
