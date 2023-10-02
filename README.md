# News Classifier
by Eilish and David <3

### Plan
- Module to extract a database (article text, section/tags) from The Guardian API
- Module to train a FFNN classifier (Keras/Pytorch) 
- Output data to visualise in Tableau (can compare the model accuracies using database gold truth) 

### Design choices made
- Only kept defined list of punctuation, deleted other
- Tokenised punctuation so that contraction words were separated into separate tokens (e.g. "weren ' t")
- Deleted stop tokens from the vocabulary 

