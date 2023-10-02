from collections import Counter

def get_vocabs(title_txtfile, tag_txtfile, title_vocab_outfile, tag_vocab_outfile):
    with open(title_txtfile, "r") as titles:
        with open(tag_txtfile, "r") as tags:
            title_data = titles.readlines()
            tag_data = tags.readlines()
    
    # get vocab set and size from all titles
    stripped_lowered_titles = [title.lower().strip() for title in title_data]
    all_tokens = [title.split() for title in stripped_lowered_titles]
    vocab = [token for token in all_tokens if token not in vocab]
    vocab_size = len(vocab)

    # get tag set and size from all tags
    stripped_tags = [tag.strip() for tag in tag_data]
    tag_vocab = [tag for tag in stripped_tags if tag not in tag_vocab]
    tag_vocab_size = len(tag_vocab)
    
    # get sorted dictionary of token and tag counts in the data
    vocab_counts = Counter(all_tokens).most_common()
    tag_counts = Counter(stripped_tags).most_common()

    # write title and tag vocabs and counts to text files
    with open(title_vocab_outfile, "w") as title_outfile:
        with open(tag_vocab_outfile, "w") as tag_outfile:
            for token, count in vocab_counts.items():
                title_outfile.write(f"{token}\t{count}")
                title_outfile.write("\n")
            for tag, count in tag_counts.items():
                    tag_outfile.write(f"{tag}\t{count}")
                    tag_outfile.write("\n")

    # get dictionaries of token/tag to idx mapping for one hot vectors
    title_indices = [i for i in range(0, vocab_size)]
    token2idx = {token:idx for (token, idx) in zip(vocab, title_indices)}
    tag_indices = [i for i in range(0, tag_vocab_size)]
    tag2idx = {tag:idx for (tag, idx) in zip(tag_vocab, tag_indices)}
    idx2tag = {idx:tag for (tag, idx) in tag2idx.items()} 

    return token2idx, tag2idx, idx2tag, vocab_size, tag_vocab_size

get_vocabs("tokenised_titles.txt", "tags.txt", "vocabs/title_vocab.txt", "vocabs/tag_vocab.txt")