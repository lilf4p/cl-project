import torch.nn as nn
import torch.optim as optim


class Client:
    """
    Client class for the federated learning process

    Args:
        id (int): Client id
        data (DataLoader): Data loader
        model (nn.Module): Model
        lr (float): Learning rate
        weight_decay (float): Weight decay
        device (str): Device
    """

    def __init__(self, id, data, model, lr=0.01, weight_decay=10e-6, device="mps"):
        self.id = id
        self.data = data
        self.device = device
        self.model = model.to(device)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_func = nn.CrossEntropyLoss()

    # TODO: Improve the training method
    #   - Add early stopping (inner validation and patience)
    #    to avoid overfitting for bigger subsets and more epochs settings
    #   - Other training optimizations
    #   - Find the best hyperparameters
    def train(self, num_epochs, patience, params, progress_bar):
        """
        Train the model for a number of epochs

        Args:
            num_epochs (int): Number of epochs
            params (dict): Model parameters
            progress_bar (tqdm): Progress bar

            Returns:
                float: Loss
        """
        self.model.load_state_dict(params)
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (x, y) in enumerate(self.data):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.loss_func(y_pred, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            progress_bar.update(1)
        progress_bar.set_postfix({"loss": running_loss / len(self.data)})
        return running_loss / len(self.data)

    def get_params(self):
        return self.model.state_dict()

    def set_params(self, params):
        self.model.load_state_dict(params)
