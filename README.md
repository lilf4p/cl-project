# Federated Learning from scratch using Pytorch
## Continual Learning Project - 2024
This repository contains the code for Federated Learning from scratch using Pytorch.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Author](#contact)
5. [References](#references)

## Introduction
Federated Learning is a machine learning setting where many clients (e.g. mobile devices or whole organizations) collaboratively train a model under the orchestration of a central server (e.g. service provider), while keeping the training data decentralized. Federated Learning enables a new paradigm for on-device training that minimizes the need for data to leave the device, effectively addressing user privacy concerns and data security.

This project aims to implement Federated Learning from scratch using Pytorch. The code is based on the paper "Communication-Efficient Learning of Deep Networks from Decentralized Data" by H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. The paper can be found [here](https://arxiv.org/abs/1602.05629).

## Installation
To install the required dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Usage
To run the default experiment, use the following command from the root directory of the project:
```bash
python src/fedsgd.py
```
Alternatively, you can import the `fedSgdPar` function from the `fedsgd.py` file and run it with your own parameters. The function signature is as follows:
```python
result0 = fedSgdPar(model=Net(), T=20, K=100, C=0.1, E=5, B=10,
                    num_samples=480, lr=0.1, patience=5, weight_decay=10e-5, noiid=False)
```

## Author
- [Leonardo Stoppani](https://github.com/lilf4p)

## References
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)