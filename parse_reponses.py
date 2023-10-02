import json
import os
import re
# import nemo_text_processing
# from nemo_text_processing.text_normalization.normalize import Normalizer
# from nltk.tokenize import wordpunct_tokenize

# normalise text
# tokenise text
# remove stop tokens?
# get vocabulary, one hot vectors

def get_titles_and_tags():
    # tags = ["environment", "politics", "technology", "science", "society", "football", "food"]

    titles_and_tags = []

    directory = os.fsencode("responses")
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):

            with open(f"responses/{filename}", "r") as f:
                data = f.read()
            data = json.loads(data)
            for article in data['response']['results']:
                title = article['webTitle']
                titles_and_tags.append([title, filename[:-6]])

    return titles_and_tags


def save_titles_and_tags():
    titles_and_tags = get_titles_and_tags()

    with open("titles.txt", "w+") as tf:
        with open("tags.txt", "w+") as f:
            for title, tag in titles_and_tags:
                f.write(tag + "\n")
                tf.write(title + "\n")
    
    return


def normalise_titles(infile, outfile):
    normalizer = Normalizer(input_case='cased', lang='en')

    with open(infile, "r") as f:
            data = f.readlines()

    remove_authors = [line[:line.find("|")] if line.find("|") else line for line in data]
    stripped = [line.strip("\n") for line in remove_authors]
    normalised = normalizer.normalize_list(stripped, punct_post_process=True)
    
    with open(outfile, "w") as pf:
        for title in normalised:
            pf.write(title + "\n")

def tokenise_titles(infile, outfile):
    with open(infile, "r") as f:
            data = f.readlines()
    
    stripped = [title.strip("\n") for title in data]
    punct = "!?.,-:;\"()'"

    tokenised = []

    for title in stripped:
        result = re.sub(r"[‘’]", "'", title)
        result = re.sub(r'([' + re.escape(punct) + '])', r' \1 ', result)
        result = re.sub("  ", " ", result)
        tokenised.append(result)

    with open(outfile, "w") as pf:
        for title in tokenised:
            for word in title:
                pf.write(word)
            pf.write("\n")


tokenise_titles("normalised_titles.txt", "tokenised_titles.txt")