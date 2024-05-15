# Naive Implementation of the Federated Learning algorithm FedSGD on the MNIST dataset
# In this implementation, the server is responsible for aggregating the model updates from the clients, and updating the global model
# SGD version when B = max and E = 1

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
from sklearn.metrics import accuracy_score
from models import Net, Cnn

device = "mps"


class Client:
    def __init__(self, id, data, model, lr=0.01, weight_decay=10e-6):
        self.id = id
        self.data = data
        self.model = model
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.data):
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print("---- Epoch %d: loss=%.3f" % (epoch, running_loss / len(self.data)))
        return running_loss / len(self.data)

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
            avg_params[key] = torch.stack(
                [params[i][key] for i in range(len(params))], 0
            ).mean(0)
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

    T = 5  # number of rounds
    K = 10  # number of clients
    C = 1  # fraction of clients to be sampled per round
    num_clients = max(int(K * C), 1)
    E = 10  # number of local epochs
    B = 128  # local batch size

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainset, valset = data.random_split(
        trainset, [int(len(trainset) * 0.8), int(len(trainset) * 0.2)]
    )
    testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    trainloader = []
    num_samples = len(trainset) // num_clients
    for i in range(num_clients):
        indices = list(range(i * num_samples, (i + 1) * num_samples))
        sampler = data.SubsetRandomSampler(indices)
        loader = data.DataLoader(trainset, batch_size=B, sampler=sampler)
        trainloader.append(loader)

    valoader = data.DataLoader(valset, batch_size=B, shuffle=True)

    testloader = data.DataLoader(testset, batch_size=B, shuffle=True)

    clients = []
    for i in range(num_clients):
        client = Client(
            i, trainloader[i], Cnn().to(device), lr=0.01, weight_decay=10e-6
        )
        clients.append(client)

    server = Server(clients, Cnn())

    for r in range(T):
        print("Round %d" % r)
        for client in clients:
            print("-- Client %d" % client.id)
            client.set_params(server.get_params())
            loss = client.train(E)
            print("-- Client %d: loss=%.3f" % (client.id, loss))
        server.aggregate()
        val_loss, val_acc = server.test(valoader)
        print("Val loss: %.3f, accuracy: %.3f" % (val_loss, val_acc))

    test_loss, test_acc = server.test(testloader)
    print("-----------------------------")
    print("-- Test loss: %.3f, accuracy: %.3f --" % (test_loss, test_acc))


if __name__ == "__main__":
    main()
