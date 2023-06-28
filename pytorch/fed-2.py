import torch
import torchvision
import torchvision.transforms as transforms
import syft as sy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)

# Create a hook for PySyft
hook = sy.frameworks.torch.hook()

# Define the parties
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# Send the data to the workers
bob_train_dataset = trainset.send(bob)
alice_train_dataset = trainset.send(alice)

# Define the model
model = Net()

# Train the model
model.train(bob_train_dataset, alice_train_dataset, epochs=10)

# Evaluate the model
model.evaluate(testloader)
