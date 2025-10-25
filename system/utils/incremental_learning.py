import torch
import torch.nn as nn
from typing import List, Tuple
import copy

class IncrementalLearning:
    def __init__(self, model: nn.Module, fisher_matrix: List[torch.Tensor], prev_params: List[torch.Tensor], 
                 ewc_lambda: float, device: str = 'cpu') -> None:
        """
        Initialize Incremental Learning with EWC
        Args:
            model: The local model instance.
            fisher_matrix: Fisher information matrix for important parameters.
            prev_params: Parameters of the model trained on previous tasks.
            ewc_lambda: Regularization strength for EWC.
            device: Device for computations (CPU or CUDA).
        """
        self.model = model
        self.fisher_matrix = fisher_matrix
        self.prev_params = prev_params
        self.ewc_lambda = ewc_lambda
        self.device = device

    # def compute_fisher_information(self, train_loader: torch.utils.data.DataLoader, loss_fn: nn.Module):
    #     """
    #     Computes Fisher Information Matrix for current task.
    #     """
    #     self.model.eval()
    #     fisher_matrix = [torch.zeros_like(p) for p in self.model.parameters()]

    #     for x, y in train_loader:
    #         x, y = x.to(self.device), y.to(self.device)
    #         self.model.zero_grad()
    #         output = self.model(x)
    #         loss = loss_fn(output, y)
    #         loss.backward()

    #         for i, param in enumerate(self.model.parameters()):
    #             fisher_matrix[i] += param.grad.pow(2) / (len(train_loader) )
    #             # fisher_matrix[i] += param.grad.pow(2) / (len(train_loader) * len(train_loader.dataset)) #use this matrix if Nan error is there..


    #     self.fisher_matrix = fisher_matrix

    def ewc_loss(self, current_params: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes EWC regularization loss.
        """
        ewc_loss = 0.0
        for param, fisher, prev_param in zip(current_params, self.fisher_matrix, self.prev_params):
            ewc_loss += torch.sum(fisher * (param - prev_param).pow(2))
        return self.ewc_lambda * ewc_loss

    def update_previous_params(self):
        """
        Updates previous task parameters with current model parameters.
        """
        self.prev_params = [p.clone().detach() for p in self.model.parameters()]
