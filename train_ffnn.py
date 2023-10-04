from utils import split_data, get_vocabs
from model import load_data, collate_fn, FFNN, compute_loss
import torch
import torch.nn as nn
import numpy as np

_, train_loader, test_loader, token2idx, epochs = load_data("data/tokenised_titles_without_punctuation.txt", "data/tags.txt")

padded_batch, longest_title = collate_fn(train_loader, pad_value=token2idx["PAD"])

model = FFNN(input_dim=longest_title, hidden_dim=100, output_dim=7)

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  


iter = 0
for epoch in range(epochs):
    for i, (titles, tags) in enumerate(train_loader):
        # Load images with gradient accumulation capabilities
        titles = titles.view(-1, longest_title).requires_grad_()
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # Forward pass to get output/logits
        outputs = model(titles)
        # Calculate Loss: softmax --> cross entropy loss
        loss = compute_loss(outputs, tags)
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()

        iter += 1
        print(f"Epoch: {epoch}, Loss: {loss}\n")
