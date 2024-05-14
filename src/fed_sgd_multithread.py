# Multithreaded Federated Learning with PyTorch
import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score
import joblib
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm

device = 'mps'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Client:
    def __init__(self, id, data, model, lr=0.01, weight_decay=10e-5):
        self.id = id
        self.data = data
        self.model = model.to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_func = nn.CrossEntropyLoss()

    # TODO: Add early stopping
    def train(self, num_epochs, params, progress_bar):
        self.model.load_state_dict(params)
        self.model.train()
        running_loss = 0.0
        for epoch in range(num_epochs):
            for batch_idx, (x, y) in enumerate(self.data):
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            progress_bar.update(1)
        progress_bar.set_postfix({'loss': running_loss/len(self.data)})
        return running_loss/len(self.data)
    
    def get_params(self):
        return self.model.state_dict()
    
    def set_params(self, params):
        self.model.load_state_dict(params)

class Server:
    def __init__(self, clients, model):
        self.clients = clients
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
    
    def aggregate(self):
        params = [client.get_params() for client in self.clients]
        # simple averaging of the clients model parameters
        avg_params = {}
        for key in params[0].keys():
            avg_params[key] = torch.stack([params[i][key] for i in range(len(params))], 0).mean(0)
        self.model.load_state_dict(avg_params)
    
    def test(self, data):
        self.model.eval()
        test_loss = 0.0
        test_accuracy = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data):
                y_pred = self.model(x)
                loss = self.loss_func(y_pred, y)
                # compute the accuracy
                y_pred = y_pred.argmax(dim=1)
                test_accuracy += accuracy_score(y.numpy(), y_pred.numpy())
                test_loss += loss.item()
        return (test_loss / len(data)), (test_accuracy / len(data))
    
    def get_params(self):
        return self.model.state_dict()
    
    def set_params(self, params):
        self.model.load_state_dict(params)

def main():
    # Set the random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    T = 15 # number of rounds
    K = 16 # number of clients
    C = 1 # fraction of clients to be sampled per round
    num_clients = max(int(K*C), 1)
    E = 2 # number of local epochs
    B = 32 # local batch size
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainset, valset = data.random_split(trainset, [int(len(trainset)*0.8), int(len(trainset)*0.2)])
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = []
    num_samples = 1000 #len(trainset) // num_clients
    for i in range(num_clients):
        indices = list(range(i*num_samples, (i+1)*num_samples))
        sampler = data.SubsetRandomSampler(indices)
        loader = data.DataLoader(trainset, batch_size=B, sampler=sampler)
        trainloader.append(loader)

    valoader = data.DataLoader(valset, batch_size=B, shuffle=True)

    testloader = data.DataLoader(testset, batch_size=B, shuffle=True)

    clients = []
    for i in range(num_clients):
        client = Client(i, trainloader[i], Net(), lr=0.01, weight_decay=10e-5)
        clients.append(client)
    
    server = Server(clients, Net())
    
    for r in range(T):
        params = server.get_params()
        progress_bar = tqdm.tqdm(total=E*num_clients, position=0, leave=False, desc='Round %d' % r)
        joblib.Parallel(n_jobs=num_clients, backend="threading")(joblib.delayed(client.train)(E, params, progress_bar) for client in clients)
        server.aggregate()
        val_loss, val_acc = server.test(valoader)
        print('Val loss: %.3f, accuracy: %.3f' % (val_loss, val_acc))
    
    test_loss, test_acc = server.test(testloader)
    print('-----------------------------')
    print('-- Test loss: %.3f, accuracy: %.3f --' % (test_loss, test_acc))

if __name__ == '__main__':
    main()