import torch
import torchvision
import torchvision.transforms as transforms
import syft as sy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)

# Split the data into two parties
trainset1, trainset2 = torch.utils.data.random_split(trainset, [30000, 30000])
testset1, testset2 = torch.utils.data.random_split(testset, [5000, 5000])

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.fc1 = torch.nn.Linear(1024, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 1024)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)

# Define the parties
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# Send the datasets to the parties
federated_train_loader = sy.FederatedDataLoader(
    torch.utils.data.TensorDataset(trainset1.dataset.data, trainset1.dataset.targets).federate((bob, alice)),
    batch_size=64,
    shuffle=True
)

federated_test_loader = sy.FederatedDataLoader(
    torch.utils.data.TensorDataset(testset1.dataset.data, testset1.dataset.targets).federate((bob, alice)),
    batch_size=64,
    shuffle=False
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch: %d, Loss: %.3f' % (epoch + 1, running_loss / len(federated_train_loader)))

model.eval()
test_loss = 0
correct = 0

with torch.no_grad():
    for data, target in federated_test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(federated_test_loader.dataset)
accuracy = 100.0 * correct / len(federated_test_loader.dataset)

print('Test Loss: %.3f, Accuracy: %.2f%%' % (test_loss, accuracy))
