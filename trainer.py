import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, scheduler, device):
        self.model = model.to(device)
        self.train_loader = train_loader.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            train_loss = 0
            for batch_inputs, batch_targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
            self.scheduler.step(train_loss)

            if self.scheduler.get_last_lr()[0] < 1e-5 :
                break
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

class Validator:
    def __init__(self, model, val_loader, criterion, device):
        self.model = model.to(device)
        self.val_loader = val_loader.to(device)
        self.criterion = criterion
        self.device = device

    def validate(self, ):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_targets in self.val_loader:
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)
                val_loss += loss.item()
        val_loss /= len(self.val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

if __name__ == "__main__":
    pass