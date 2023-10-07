from utils import split_data, compute_loss
from model import FFNN, Data
import torch
from tqdm import tqdm

class Train(Data):
    def __init__(self, batch_size=int, epochs=int, lr=float, hidden_dim=int):
        Data.__init__(self, data_fpaths=split_data_fpaths)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.hidden_dim = hidden_dim

    def setup_data(self):
        train_data, train_title_lengths = self.compile_data(mode="train")
        test_data, test_title_lengths = self.compile_data(mode="test")
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        full_lengths = train_title_lengths + test_title_lengths
        longest_title = max(full_lengths)
        return train_loader, test_loader, longest_title
    
    def setup_model(self):
        # TO DO: get output dim automatically from this to 
        _, _, longest_title = self.setup_data()
        model = FFNN(input_dim=longest_title, hidden_dim=self.hidden_dim, output_dim=7)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)  
        return model, optimizer

    def run_epoch(self):
        train_loader, test_loader, _ = self.setup_data()
        model, optimizer = self.setup_model()
        print(f"No. of epochs: {self.epochs}")

        step = 0
        for epoch in tqdm(range(self.epochs)):
            for titles, tags in train_loader:
                titles = titles.requires_grad_()
                optimizer.zero_grad()
                outputs = model(titles)
                loss = compute_loss(outputs, tags)
                loss.backward()
                optimizer.step()

                step += 1

                if step % 100 == 0:
                    correct = 0
                    total = 0
                    for titles, tags in test_loader:
                        titles = titles.requires_grad_()
                        outputs = model(titles)
                        _, predicted = torch.max(outputs.data, 1)
                        total += tags.size(0)
                        correct += (predicted == tags).sum()
                    
                    accuracy = 100 * correct/total
                    print(f"Training step (no. of batches seen): {step}, Loss: {loss.item()}, Accuracy: {accuracy}")
                        
            print(f"Epoch: {epoch}, Loss: {loss}\n")

title_data_fpath = "data/big/tokenised_titles_without_punctuation.txt"
tag_data_fpath = "data/big/tags.txt"
split_data_fpaths = {"title_train":"data/split-data/train/titles.txt", "tag_train":"data/split-data/train/tags.txt", "title_test":"data/split-data/test/titles.txt", "tag_test":"data/split-data/test/tags.txt"}
split_data(title_data_fpath, tag_data_fpath, output_fpaths=split_data_fpaths)
train = Train(batch_size=10, epochs=5, lr=0.00001, hidden_dim=50)
train.run_epoch()
