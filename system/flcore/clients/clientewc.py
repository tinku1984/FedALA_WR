import torch
from flcore.clients.clientbase import Client
from utils.incremental_learning import IncrementalLearning

class ClientEWC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.ewc_lambda = args.ewc_lambda
        self.incremental_learning = IncrementalLearning(
            model=self.model,
            fisher_matrix=[],
            prev_params=[],
            ewc_lambda=self.ewc_lambda,
            device=self.device
        )

    def local_initialization(self, received_global_model):
        """
        Initialize the client's local model with the received global model.
        """
        self.model.load_state_dict(received_global_model.state_dict())


    def train(self):
        train_loader = self.load_train_data()
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)  # Adjust parameters as needed

        self.model.train()

        for epoch in range(self.local_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(x)
                loss = self.loss(output, y)

                # Add EWC loss
                if len(self.incremental_learning.prev_params) > 0:
                    ewc_loss = self.incremental_learning.ewc_loss(self.model.parameters())
                    loss += ewc_loss

                # Backward pass
                loss.backward()

                # Gradient clipping (Newly added)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Adjust max_norm as needed

                self.optimizer.step()
            # Step the learning rate scheduler
            scheduler.step()

        # Compute Fisher Information Matrix after training
        self.incremental_learning.compute_fisher_information(train_loader, self.loss)
        self.incremental_learning.update_previous_params()
