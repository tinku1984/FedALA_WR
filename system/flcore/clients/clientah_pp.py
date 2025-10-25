import torch
from flcore.clients.clientah import clientAH
from utils.incremental_learning import IncrementalLearning
from torch.optim.lr_scheduler import StepLR


class clientAH_PP(clientAH):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.ewc_lambda = args.ewc_lambda
        self.memory_buffer = []  # Replay buffer
        self.buffer_size = args.buffer_size
        # self.num_rounds = args.global_rounds

        self.incremental_learning = IncrementalLearning(
            model=self.model,
            fisher_matrix=[
                torch.zeros_like(param).to(self.device) for param in self.model.parameters()
            ],
            prev_params=[],
            ewc_lambda=self.ewc_lambda,
            device=self.device,
        )
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        print('I am inside FedAH++ ...........')
        
        # Initialize learning rate scheduler
        # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.01)  # Decay LR every 10 steps by factor of 0.1

    def train(self, current_round=1):
        trainloader = self.load_train_data()
        self.model.train()

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(x)
                loss = self.loss(output, y)

                # Replay loss
                if self.memory_buffer:
                    replay_loss = self.compute_replay_loss(self.memory_buffer)
                    # alpha = max(0.1, 0.5 * (1 - self.current_round / self.total_rounds))  # Decay replay weight


                    # alpha = max(0.1, 0.5 * (1 - epoch / self.local_epochs))  # Decay replay weight
                    # print(f'value of Alpha : {alpha}, self.current_round: {self.current_round}, self.total_rounds: {self.total_rounds}')
                    alpha = 0.2 * (1 / (1 + 0.1 * current_round)**2)
                    loss += alpha * replay_loss

                # Backward pass
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
            
            # Adjust learning rate at the end of each epoch
            # self.scheduler.step()  # Update learning rate based on the scheduler
            
            # Log current learning rate
            # current_lr = self.scheduler.get_last_lr()[0]
            # print(f"Epoch {epoch + 1}: Learning Rate = {current_lr}")


        # Update memory buffer with new samples
        self.update_memory_buffer(trainloader)

    def compute_replay_loss(self, memory_buffer):
        """
        Compute replay loss using the memory buffer.
        """
        replay_loss = 0.0
        for x_mem, y_mem in memory_buffer:
            x_mem, y_mem = x_mem.to(self.device), y_mem.to(self.device)
            output_mem = self.model(x_mem)
            replay_loss += self.loss(output_mem, y_mem)
        return replay_loss / len(memory_buffer)  # Average the replay loss

    def update_memory_buffer(self, trainloader):
        """
        Update the memory buffer with new samples while ensuring diversity and class balance.
        """
        class_counts = {}
        memory_buffer_set = set()

        # Initialize class_counts from current buffer
        for _, label in self.memory_buffer:
            label = label.item()
            class_counts[label] = class_counts.get(label, 0) + 1

        num_classes = len(class_counts) or 1  # Ensure at least one class
        max_samples_per_class = self.buffer_size // num_classes

        for x, y in trainloader:
            for i in range(len(y)):
                label = y[i].item()

                # Only add if below class-specific limit and unique
                if class_counts.get(label, 0) < max_samples_per_class:
                    sample = (x[i].unsqueeze(0).cpu(), y[i].unsqueeze(0).cpu())
                    if sample not in memory_buffer_set:  # Prevent duplicates
                        self.memory_buffer.append(sample)
                        memory_buffer_set.add(sample)
                        class_counts[label] = class_counts.get(label, 0) + 1

                        # Maintain buffer size
                        if len(self.memory_buffer) > self.buffer_size:
                            self.memory_buffer.pop(0)

        # Adjust class limits dynamically
        if len(self.memory_buffer) >= self.buffer_size:
            self._adjust_memory_buffer(class_counts)

    def _adjust_memory_buffer(self, class_counts):
        """
        Adjust the memory buffer by probabilistically removing samples to maintain class balance.
        """
        removal_probs = [1 / class_counts[label.item()] for _, label in self.memory_buffer]
        removal_probs = torch.tensor(removal_probs) / sum(removal_probs)

        while len(self.memory_buffer) > self.buffer_size:
            remove_idx = torch.multinomial(removal_probs, 1).item()
            _, label = self.memory_buffer.pop(remove_idx)
            class_counts[label.item()] -= 1

    def get_significant_updates(self, top_percent=100):
        """
        Get the top 'top_percent' parameters. This is retained for compatibility but not used in Replay-Based Learning.
        """
        return list(self.model.parameters())
