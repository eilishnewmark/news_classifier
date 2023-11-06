# News Classifier

### Plan
- Extract a database (article text, section/tags) from The Guardian API
- Train an EmbeddingBag classifier with linear output layer (Pytorch) 
TODO: output test data results to visualise in Tableau for evaluation
TODO: make a validation data set to use during training for hyperparameter tuning

### Design choices made so far
- If keeping stop tokens: only kept defined list of punctuation, deleted others
    \> Tokenised punctuation so that contraction words were separated into separate tokens (e.g. "weren ' t")
- If deleting stop tokens: also delete all punctuation tokens
- Delete any words that have vocab count of < 1 when processing train and test data
- Replace words in test data unseen in training data with UNK token
- Changed from FFNN to EmbeddingBag model to enhance the feature space, boosting accuracy


