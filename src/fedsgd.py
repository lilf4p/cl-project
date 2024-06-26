# Naive Implementation of the Federated Learning algorithm FedSGD on the MNIST dataset
# In this implementation, the server is responsible for aggregating the model updates from the clients, and updating the global model
# SGD version when B = max and E = 1

from ast import Dict
from unittest import result
from matplotlib.pyplot import cla
from sympy import ordered
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset

import tqdm
import joblib
import numpy as np
import random

from src.models import Net, Cnn
from src.Client import Client
from src.Server import Server
from src.utils import EarlyStopper

device = "mps"


def fedSgdSeq(
    model=Net(),
    T=5,
    K=10,
    C=1,
    E=10,
    B=128,
    num_samples=1000,
    lr=0.01,
    weight_decay=0,
    patience=1,
    verbose=True,
):
    """
    Run the sequential implementation of FedSGD on the MNIST dataset

    Args:
        model (nn.Module): Model
        T (int): Number of rounds
        K (int): Number of clients
        C (int): Fraction of clients to be sampled per round
        E (int): Number of local epochs
        B (int): Local batch size
        num_samples (int): Number of samples per client (size of the local dataset)
        lr (float): Learning rate of the client optimizer
        weight_decay (float): Weight decay of the client optimizer
        patience (int): Patience for early stopping

    Returns:
        (float, float): Test loss and test accuracy
    """

    if verbose:
        print("Running the Sequential implementatiopn of FedSGD on MNIST dataset")
    if verbose:
        print(
            f"- Parameters: T={T}, K={K}, C={C}, E={E}, B={B}, num_samples={num_samples}, lr={lr}, weight_decay={weight_decay}, patience={patience}"
        )
    if verbose:
        print(f"- Model: {model.get_type()}")

    # Set the random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    num_clients = max(int(K * C), 1)

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

    if num_samples * num_clients > len(trainset):
        raise ValueError("The number of samples per client is too big")

    trainloader = []
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
            i,
            trainloader[i],
            Cnn() if model.get_type() == "Cnn" else Net(),
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        clients.append(client)

    server = Server(clients, model)  # experiment parameters

    # Federated learning Algorithm
    val_losses = []
    val_accs = []
    early_stopper = EarlyStopper(patience, min_delta=10e-3)
    for r in range(T):
        params = server.get_params()
        progress_bar = tqdm.tqdm(
            total=E * num_clients, position=0, leave=False, desc="Round %d" % r
        )
        for client in clients:
            loss = client.train(E, params, progress_bar)
        server.aggregate()
        val_loss, val_acc = server.test(valoader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if early_stopper.early_stop(val_loss):
            if verbose:
                print("Early stopping")
            break
        if verbose:
            print("Server - Val loss: %.3f, Val accuracy: %.3f" % (val_loss, val_acc))

    test_loss, test_acc = server.test(testloader)
    if verbose:
        print("-----------------------------")
    if verbose:
        print("-- Test loss: %.3f, accuracy: %.3f --" % (test_loss, test_acc))

    result = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "val_losses": val_losses,
        "val_accs": val_accs,
    }

    return result


def fedSgdPar(
    model=Cnn(),
    T=5,
    K=10,
    C=1,
    E=10,
    B=128,
    num_samples=1000,
    lr=0.01,
    weight_decay=0,
    patience=1,
    verbose=True,
    noiid=False,
    path="models/model.pth",
):
    """
    Run the parallel implementation of FedSGD on the MNIST dataset

    Args:
        model (nn.Module): Model
        T (int): Number of rounds
        K (int): Number of clients
        C (int): Fraction of clients to be sampled per round
        E (int): Number of local epochs
        B (int): Local batch size
        num_samples (int): Number of samples per client (size of the local dataset)
        lr (float): Learning rate of the client optimizer
        weight_decay (float): Weight decay of the client optimizer
        patience (int): Patience for early stopping

    Returns:
    (float, float): Test loss and test accuracy
    """

    if verbose:
        print("Running the Parallel implementation FedSGD on MNIST dataset")
    if verbose:
        print(
            f"- Parameters: T={T}, K={K}, C={C}, E={E}, B={B}, num_samples={num_samples}, lr={lr}, weight_decay={weight_decay}, patience={patience}"
        )
    if verbose:
        print(f"- Model: {model.get_type()}")

    # Set the random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    clients_each_round = max(int(K * C), 1)

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

    if verbose:
        print("- Data Split: ", len(trainset), len(valset), len(testset))

    if num_samples * K > len(trainset):
        raise ValueError("The number of samples per client is too big")

    trainloader = []
    if noiid:
        if verbose:
            print("Using no-IID data")
        offset = 0
        for i in range(K):
            indices = [j for j in range(len(trainset)) if trainset[j][1] == i % 10]
            indices = indices[offset : offset + num_samples]  # avoid overlap
            sampler = data.SubsetRandomSampler(indices)
            loader = data.DataLoader(trainset, batch_size=B, sampler=sampler)
            trainloader.append(loader)
            if i % 10 == 9 and offset + num_samples < len(indices):
                offset += num_samples
    else:
        if verbose:
            print("Using IID data")
        for i in range(K):
            indices = list(range(i * num_samples, (i + 1) * num_samples))
            sampler = data.SubsetRandomSampler(indices)
            loader = data.DataLoader(trainset, batch_size=B, sampler=sampler)
            trainloader.append(loader)

    valoader = data.DataLoader(valset, batch_size=B, shuffle=True)

    testloader = data.DataLoader(testset, batch_size=B, shuffle=True)

    clients = []
    for i in range(K):
        client = Client(
            i,
            trainloader[i],
            Cnn() if model.get_type() == "Cnn" else Net(),
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        clients.append(client)

    server = Server(clients, model)

    # Federated learning Algorithm
    val_losses = []
    val_accs = []
    early_stopper = EarlyStopper(patience, min_delta=10e-3)
    for r in range(T):
        params = server.get_params()
        progress_bar = tqdm.tqdm(
            total=E * clients_each_round, position=0, leave=False, desc="Round %d" % r
        )
        random_clients = random.sample(clients, clients_each_round)
        joblib.Parallel(n_jobs=8, backend="threading")(
            joblib.delayed(client.train)(E, params, progress_bar)
            for client in random_clients
        )
        server.aggregate(random_clients)
        val_loss, val_acc = server.test(valoader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if early_stopper.early_stop(val_loss):
            if verbose:
                print("Early stopping")
            break
        if verbose:
            print("Server - Val loss: %.3f, Val accuracy: %.3f" % (val_loss, val_acc))

    # test the model
    test_loss, test_acc = server.test(testloader)
    print("-- Test loss: %.3f, Test accuracy: %.3f --" % (test_loss, test_acc))

    # save the model
    server.save(path)

    result = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "val_losses": val_losses,
        "val_accs": val_accs,
    }

    return result


if __name__ == "__main__":
    fedSgdSeq()
