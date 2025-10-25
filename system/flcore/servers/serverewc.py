import time
from flcore.clients.clientewc import ClientEWC
from flcore.servers.serverbase import Server
from threading import Thread

class FedEWC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # Select slow clients
        self.set_slow_clients()
        self.set_clients(ClientEWC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # Multi-threaded training (optional)
            # threads = [Thread(target=client.train) for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy:")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round:")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def send_models(self):
        """
        Sends global model to all selected clients for training.
        """
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.local_initialization(self.global_model)
