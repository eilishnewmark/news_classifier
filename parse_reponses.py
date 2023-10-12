import json
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from unidecode import unidecode
from nemo_text_processing.text_normalization.normalize import Normalizer



def get_titles_and_tags():
    # tags = ["environment", "politics", "technology", "science", "society", "football", "food"]

    titles_and_tags = []

    directory = os.fsencode("responses")
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        tag = re.search(r"-([a-z]+).json", filename)
        if filename.endswith(".json"):
            with open(f"responses/{filename}", "r") as f:
                data = f.read()
            data = json.loads(data)
            try:
                file_data = data['response']['results']
                for article in file_data:
                    title = article['webTitle']
                    titles_and_tags.append([title.rstrip(), tag.group(1)])
            except:
                continue
    return titles_and_tags

def save_titles_and_tags(title_outpath, tags_outpath):
    titles_and_tags = get_titles_and_tags()
    print(len(titles_and_tags))
    with open(title_outpath, "w+") as tf:
        with open(tags_outpath, "w+") as f:
            for title, tag in titles_and_tags:
                f.write(tag + "\n")
                tf.write(title + "\n")
    return

def preprocess_titles(infile, outfile):
    """Removes any authors/headings in the title that appear before/after a '|'."""
    with open(infile, "r") as f:
            data = f.readlines()

    remove_beginning = [line[line.find("|") + 1:] if (-1 < line.find("|") < len(line)/2) else line for line in data]
    remove_authors = [line[:line.find("|")] if (line.find("|") > len(line)/2) else line for line in remove_beginning]
    stripped = [line.strip("\n") for line in remove_authors]

    with open(outfile, "w") as pf:
        for title in stripped:
            pf.write(title + "\n")

def normalise_titles(infile, outfile):
    normalizer = Normalizer(input_case='cased', lang='en')

    with open(infile, "r") as f:
            data = f.readlines()

    stripped = [line.strip("\n") for line in data]
    normalised = normalizer.normalize_list(stripped, punct_post_process=True)
    
    with open(outfile, "w") as pf:
        for title in normalised:
            pf.write(title + "\n")

def tokenise_titles(infile, outfile, remove_all_punctuation=True):
    with open(infile, "r") as f:
            data = f.readlines()
    stripped = [title.strip("\n") for title in data]

    if remove_all_punctuation:
        punct = "" # remove all punctuation
    else:
        punct = "!?.,-:;\"()'…[0-9]"

    tokenised = []

    stops = set(stopwords.words('english'))

    for title in stripped:
        #1. Remove punctuation 
        # replace these special characters with ASCII counterparts
        result = re.sub(r"[‘’]", "'", title)
        result = re.sub(r"–", "-", result)
        # turn ellipses into single token
        result = re.sub(r"\. *\. *\.", "…", result)
        if not remove_all_punctuation:
            # remove all weird punctuation
            result = re.sub(r'([' + re.escape(punct) + '])', r' \1 ', result)
        # turn tokens into unicode (get rid of accents etc)
        result = unidecode(result)
        # if not approved punctuation, delete
        result = re.sub(rf"[^\w {punct}]", " ", result)

        #2. Remove stop words
        if remove_all_punctuation:
            result = word_tokenize(result)
            result = [word for word in result if word.lower() not in stops]
            result = " ".join(result)

        # remove multiple spaces
        result = re.sub(" {2,}", " ", result)

        tokenised.append(result)

    with open(outfile, "w") as pf:
        for title in tokenised:
            for word in title:
                pf.write(word)
            pf.write("\n")

def main():
    """Order of files on preprocessing line:
    - responses/*.json
    - tags.txt and titles.txt (get_titles_and_tags(), save_titles_and_tags())
    - preprocessed_titles.txt
    - normalised_titles.txt
    - tokenised_titles.txt"""
    # get_titles_and_tags()
    # save_titles_and_tags("titles.txt", "tags.txt")
    # preprocess_titles("titles.txt", "preprocessed-titles.txt")
    # normalise_titles("preprocessed-titles.txt", "normalised-titles.txt")
    tokenise_titles("data_processing/normalised-titles.txt", "tokenised-titles_without_punc.txt", remove_all_punctuation=True)

main()