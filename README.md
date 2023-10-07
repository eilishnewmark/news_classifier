# News Classifier

### Plan
- Module to extract a database (article text, section/tags) from The Guardian API
- Module to train a FFNN classifier (Keras/Pytorch) 
- Output data to visualise in Tableau

### Design choices made
- If keeping stop tokens: only kept defined list of punctuation, deleted others
    \> Tokenised punctuation so that contraction words were separated into separate tokens (e.g. "weren ' t")
- If deleting stop tokens: also delete all punctuation tokens


