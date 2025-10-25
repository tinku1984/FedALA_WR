import time
import torch
import copy
import numpy as np
from utils.data_utils import read_client_data
from flcore.servers.serverah import FedAH
# from flcore.servers.serverala import FedALA
# from flcore.clients.clientala_pp import clientALA_PP
from flcore.clients.clientah_pp import clientAH_PP

class FedAH_PP(FedAH):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientAH_PP)
        self.ewc_lambda = args.ewc_lambda
        self.buffer_size = args.buffer_size
    
    # def aggregate_parameters(self):
    #     # Aggregation logic for FedALA++
    #     super().aggregate_parameters()  # Ensure you call the parent method if needed
    
    def set_clients(self, client_class):
        """
        Override client initialization to ensure clientAH_PP is used instead of clientAH.
        """
        self.clients = []
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = client_class(self.args, id=i, train_samples=len(train_data), test_samples=len(test_data), train_slow=train_slow, send_slow=send_slow)
            self.clients.append(client)
    
    def select_clients(self):
        """
        Select a subset of clients for the current communication round.
        """
        self.selected_clients = list(np.random.choice(self.clients, self.num_join_clients, replace=False))
        return self.selected_clients
    
    # def aggregate_parameters(self):
    #     assert (len(self.uploaded_models) > 0)

    #     self.global_model = copy.deepcopy(self.uploaded_models[0])
    #     for param in self.global_model.parameters():
    #         param.data.zero_()    
    
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------FedAH++ Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break


    def aggregate_parameters(self):
        """
        Aggregate sparse client updates using Fisher Information-based selection.
        """
        total_weight = 0
        aggregated_params = [torch.zeros_like(param) for param in self.global_model.parameters()]

        client_weights = self.compute_attention_weights()

        for client, weight in zip(self.selected_clients, client_weights):
            total_weight += weight
            sparse_updates = client.get_significant_updates(top_percent=100)  # Allow more updates

            # Debugging log for sparse updates
            # print(f"Client {client.id} is sending {len(sparse_updates)} updates")

            for param, sparse_update in zip(aggregated_params, sparse_updates):
                if sparse_update is not None and sparse_update.shape == param.shape:
                    param.data += sparse_update.data * weight


        for param in aggregated_params:
            param.data /= total_weight

        # Debugging log for final aggregation
        print(f"Global model updated with aggregated parameters")
        for param, aggregated_param in zip(self.global_model.parameters(), aggregated_params):
            param.data = aggregated_param.data.clone()

    
    def evaluate_global_model(self):
        """
        Evaluate the global model on the test dataset.
        Returns:
            Accuracy of the global model.
        """
        stats = self.evaluate()
        accuracy = stats.get("accuracy", 0)
        print(f"Global model accuracy: {accuracy}")  # Debug log
        return accuracy

    

    def compute_attention_weights(self):
        """
        Compute attention weights based on client updates.
        """
        weights = []
        for client in self.selected_clients:
            similarity = self.compute_update_similarity(client)
            weights.append(similarity)
            # print(f"Client {client.id} similarity: {similarity}")  # Debug: log similarity
        return torch.softmax(torch.tensor(weights), dim=0).tolist()


    def compute_update_similarity(self, client):
        """
        Compute similarity between client update and global model.
        """
        similarity = 0.0
        for param, global_param in zip(client.model.parameters(), self.global_model.parameters()):
            similarity += torch.cosine_similarity(
                param.data.flatten(), global_param.data.flatten(), dim=0
            ).item()
            similarity += torch.norm(param.data - global_param.data).item()

        return similarity