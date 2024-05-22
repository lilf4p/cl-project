import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class Server:
    """
    Server class for the federated learning process

    Args:
        clients (list): List of clients
        model (nn.Module): Model

    """

    def __init__(self, clients, model):
        self.clients = clients
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()

    def aggregate(self):
        """
        Aggregate the model parameters from the clients

        Args:
            None

        Returns:
            None
        """
        params = [client.get_params() for client in self.clients]
        # simple averaging of the clients model parameters
        avg_params = {}
        for key in params[0].keys():
            avg_params[key] = torch.stack(
                [params[i][key] for i in range(len(params))], 0
            ).mean(0)
        self.model.load_state_dict(avg_params)

    def test(self, data):
        """
        Test the model

        Args:
            data (DataLoader): Data loader

        Returns:
            tuple: Loss and accuracy
        """
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
