import json

def main():
    with open("responses/politics.txt", "r") as f:
        data = f.read()
    data = json.loads(data)
    print(data)

main()

# remove punctuation
# normalise text
# tokenise text
# remove stop tokens?
# get vocabulary, one hot vectors
