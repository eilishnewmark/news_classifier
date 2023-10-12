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
        train_data = self.compile_data(mode="train")
        test_data = self.compile_data(mode="test")
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        return train_loader, test_loader
    
    def setup_model(self):
        model = FFNN(input_dim=max(self.title_lengths), hidden_dim=self.hidden_dim, output_dim=self.tag_count)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)  
        return model, optimizer

    def run_epoch(self):
        train_loader, test_loader = self.setup_data()
        model, optimizer = self.setup_model()
        print(f"No. of epochs: {self.epochs}")

        step = 0
        for epoch in tqdm(range(self.epochs)):
            for i, (titles, tags) in enumerate(train_loader):
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
                    print(f"Epoch {epoch} | Training step: {step} | {i}/{len(train_loader)} batches | Loss: {loss.item()} | Accuracy: {accuracy} |")
                        
            print(f"Epoch: {epoch}, Loss: {loss}\n")

title_data_fpath = "data/tokenised-titles_without_punc.txt"
tag_data_fpath = "data/tags.txt"
split_data_fpaths = {"title_train":"data/split-data/train/titles.txt", "tag_train":"data/split-data/train/tags.txt", "title_test":"data/split-data/test/titles.txt", "tag_test":"data/split-data/test/tags.txt"}
split_data(title_data_fpath, tag_data_fpath, output_fpaths=split_data_fpaths)
train = Train(batch_size=20, epochs=30, lr=0.001, hidden_dim=100)
train.run_epoch()

