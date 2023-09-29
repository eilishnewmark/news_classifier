import json
import os


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


save_titles_and_tags()