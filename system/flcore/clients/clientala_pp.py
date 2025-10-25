import torch
import torch.nn.functional as F
from flcore.clients.clientala import clientALA
from torch.optim.lr_scheduler import StepLR


class clientALA_PP(clientALA):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.ewc_lambda = args.ewc_lambda
        self.memory_buffer = []  # Replay buffer
        self.buffer_size = args.buffer_size

      
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        print('I am inside FedALA++ with Uncertainty-Based Sampling ...........')

    def train(self, current_round=1):
        trainloader = self.load_train_data()
        self.model.train()

        # Compute replay loss once per epoch (not per mini-batch)
        if self.memory_buffer:
            replay_loss = self.compute_replay_loss(self.memory_buffer)
            # alpha = self.ewc_lambda * (1 / (1 + 0.1 * current_round)**2)
            # print(f'self ewc lambda:{self.ewc_lambda}, Buffer size {self.buffer_size}')
            alpha = self.ewc_lambda

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(x)
                loss = self.loss(output, y)

                # Apply replay loss (without recomputing it)
                if self.memory_buffer:
                    # loss += alpha * replay_loss
                    loss = loss * (1-self.ewc_lambda) + alpha * replay_loss

                # Backward pass
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
            
        # Update memory buffer using Uncertainty-Based Sampling
        self.update_memory_buffer(trainloader)


    def compute_replay_loss(self, memory_buffer):
        replay_loss = 0.0
        for x_mem, y_mem in memory_buffer:
            x_mem, y_mem = x_mem.to(self.device), y_mem.to(self.device)
            output_mem = self.model(x_mem)
            replay_loss += self.loss(output_mem, y_mem)
        return (replay_loss / len(memory_buffer)).detach()

    def update_memory_buffer(self, trainloader):
        """
        Update memory buffer using Uncertainty-Based Sampling (Entropy-based selection).
        """
        uncertainty_scores = []
        
        
        for x, y in trainloader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                output = self.model(x)
                probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
                entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)  # Compute entropy

            for i in range(len(y)):
                uncertainty_scores.append((entropy[i].item(), x[i].cpu(), y[i].cpu()))

        # Sort samples by uncertainty (descending order) and select top-N uncertain samples
        uncertainty_scores.sort(reverse=True, key=lambda x: x[0])
        top_uncertain_samples = uncertainty_scores[: self.buffer_size]
        
        self.memory_buffer = [(sample[1].unsqueeze(0), sample[2].unsqueeze(0)) for sample in top_uncertain_samples]
        
        # print(f'Updated memory buffer with {len(self.memory_buffer)} uncertain samples.')

    def get_significant_updates(self, top_percent=100):
        return list(self.model.parameters())
