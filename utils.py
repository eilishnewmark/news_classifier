from collections import Counter, namedtuple
from sklearn.model_selection import train_test_split

TitleTagObject = namedtuple("TitleTagObject", ['title', 'tag', 'title_length'])

def split_data(title_fpath, tag_fpath):
     with open(title_fpath, "r") as title_data:
          with open(tag_fpath, "r") as tag_data:
               titles = title_data.readlines()
               tags = tag_data.readlines()
     title_train, title_test, tag_train, tag_test = train_test_split(titles, tags, train_size=0.8)

     return title_train, title_test, tag_train, tag_test

def get_vocabs(titles, tags):
    """titles/tags = list of title/tag strings"""
    # get vocab set and size from all titles
    stripped_lowered_titles = [title.lower().strip() for title in titles]
    all_tokens = []
    for title in stripped_lowered_titles:
         split_title = title.split()
         for token in split_title:
              all_tokens.append(token)   
    vocab = list(set(all_tokens)) + ["PAD"]
    vocab_size = len(vocab) + 1 # PAD

    # get tag set and size from all tags
    stripped_tags = [tag.strip() for tag in tags]
    tag_vocab = list(set(stripped_tags))
    tag_vocab_size = len(tag_vocab)
    
    # get sorted tuple list of token and tag counts in the data
    vocab_counts = Counter(all_tokens).most_common()
    tag_counts = Counter(stripped_tags).most_common()

    # get dictionaries of token/tag to idx mapping for one hot vectors
    title_indices = [i for i in range(0, vocab_size)]
    token2idx = {token:idx for (token, idx) in zip(vocab, title_indices)}
    tag_indices = [i for i in range(0, tag_vocab_size)]
    tag2idx = {tag:idx for (tag, idx) in zip(tag_vocab, tag_indices)}
    idx2tag = {idx:tag for (tag, idx) in tag2idx.items()} 

    return token2idx, tag2idx, idx2tag, vocab_size, tag_vocab_size, vocab_counts, tag_counts