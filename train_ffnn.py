from utils import split_data, compute_loss
from model import Model, Data
import torch
from tqdm import tqdm

class Train(Data):
    def __init__(self, batch_size=int, epochs=int, lr=float, hidden_dim=int):
        Data.__init__(self, data_fpaths=split_data_fpaths)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.train_loader, self.test_loader = self.setup_data()
        self.model, self.optimizer, self.scheduler = self.setup_model()

    def setup_data(self):
        train_data = self.compile_data(mode="train")
        test_data = self.compile_data(mode="test")
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
        return train_loader, test_loader
    
    def setup_model(self):
        model = Model(vocab_size=self.train_vocab, embedding_dim=self.hidden_dim, num_tags=self.tag_count)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)  
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
        return model, optimizer, scheduler

    def run_epoch(self):
        print(f"No. of epochs: {self.epochs}")

        self.model.train()

        step = 0
        for epoch in tqdm(range(self.epochs)):
            for i, (titles, tags) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model.forward(titles, offsets=None)
                loss = compute_loss(outputs, tags)
                loss.backward()
                self.optimizer.step()

                step += 1

                with torch.no_grad():
                    if step % 200 == 0:
                        correct, total = 0, 0
                        for titles, tags in self.test_loader:
                            outputs = self.model(titles, offsets=None)
                            _, predicted = torch.max(outputs.data, 1)
                            total += tags.size(0)
                            correct += (predicted == tags).sum()
                        
                        accuracy = 100 * correct/total
                        print(f"Training step: {step} | {i}/{len(self.train_loader)} batches | Loss: {loss.item()} | Accuracy: {accuracy} |")
                        
            print(f"\n***Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}***\n")
            self.scheduler.step()
    
    def run_inference(self):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for idx, (titles, tags) in enumerate(self.test_loader):
                outputs = self.model(titles, offsets=None)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == tags).sum()
                total += tags.size(0)
        return correct / total

title_data_fpath = "data/tokenised-titles_without_punc.txt"
tag_data_fpath = "data/tags.txt"
split_data_fpaths = {"title_train":"data/split-data/train/titles.txt", "tag_train":"data/split-data/train/tags.txt", "title_test":"data/split-data/test/titles.txt", "tag_test":"data/split-data/test/tags.txt"}
split_data(title_data_fpath, tag_data_fpath, output_fpaths=split_data_fpaths)
train = Train(batch_size=64, epochs=10, lr=5, hidden_dim=64)
train.run_epoch()
test_set_acc = 100 * train.run_inference()
print("\n***RESULTS ON TEST SET: ", test_set_acc, "***")

