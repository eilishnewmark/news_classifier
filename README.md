# News Classifier

### Plan
- Module to extract a database (article text, section/tags) from The Guardian API
- Module to train an FFNN classifier (Pytorch) 
- Output data to visualise in Tableau  

### Design choices made so far
- If keeping stop tokens: only kept defined list of punctuation, deleted others
    \> Tokenised punctuation so that contraction words were separated into separate tokens (e.g. "weren ' t")
- If deleting stop tokens: also delete all punctuation tokens


