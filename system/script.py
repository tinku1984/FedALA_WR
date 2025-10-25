import os
import subprocess

# Parameters for grid search
# buffer_sizes = [10, 20, 50, 100]
# ewc_lambdas = [0.1, 1, 5, 10]

buffer_sizes = [10, 20]
ewc_lambdas = [0.1, 1]


# Example for one configuration
os.environ["MODEL_TYPE"] = "CNN"  # Set the model type dynamically

# command = [
#     "python", "main.py",
#     "-data", "MNIST",
#     "-m", os.environ["MODEL_TYPE"],
#     "-algo", "FedALA++",
#     "-gr", "5",
#     "-did", "0",
#     "-grid_search",
#     "--ewc_lambda", "1.0",
#     "--buffer_size", "200",
# ]

# subprocess.run(command)



for buffer_size in buffer_sizes:
    for ewc_lambda in ewc_lambdas:
        # Dynamically construct the command
        command = [
        "python", "main.py",
        "-data", "Cifar10_pat",
        "-m", os.environ["MODEL_TYPE"],
        "-algo", "FedALA++",
        "-gr", "10",
        "-did", "0",
        "--times","1",
        "--local_epochs","1",
        "--ewc_lambda", "1.0",
        "--buffer_size", "200",
    ]
        
        # Print the command for debugging
        print(f"Executing: {' '.join(command)}")
        
        # Run the subprocess
        result = subprocess.run(command, capture_output=True, text=True)

        # Check for errors
        if result.returncode != 0:
            print(f"Error executing command: {result.stderr}")
        else:
            print(f"Output: {result.stdout}")
