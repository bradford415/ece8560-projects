"""classify.py

Bradley Selee
ECE8560
Project 1
Februrary 23, 2023
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from pathlib import Path
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class Net(nn.Module):
    """Simple multilayer perceptron for the CIFAR Dataset"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        

    def forward(self, x):
        x = torch.flatten(x, 1) # Flatten the 3D input to 1D
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc5(x)
        #x = self.bn5(x)

        return x


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    """Function to train the model and perform back propagation,this function is called every epoch"""
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    return correct, loss


def test(model, device, test_loader, criterion):
    """Function to test our trained model on the test set every epoch, no back propagation is done"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)

    return correct, test_loss


def inference(model, device, image, classes):
    """A single forward pass of our trained model used for inferencing a single image"""
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        prediction = torch.argmax(output)
        print(f'prediction result: {classes[prediction]}')
            

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('mode', type=str,
                        help='train or evaluate the model (train or eval)')
    parser.add_argument('image', type=str, nargs='?',
                        help='path to .png image for inference (32x32x3)')
    batch_size = 64
    test_batch_size = 1
    epochs = 15
    lr = 0.001
    gamma = 0.1
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # Assign cuda specific parameters if a GPU is available
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': test_batch_size, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Data augmentation image transforms
    train_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))
        ])
    test_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))
        ])

    # Extract train and test datasets
    cifar_train = datasets.CIFAR10('../data', train=True, download=True,
                       transform=train_transform)
    cifar_test = datasets.CIFAR10('../data', train=False,
                       transform=test_transform)

    # Convert dataset to a 'loader object'
    train_loader = torch.utils.data.DataLoader(cifar_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(cifar_test,**test_kwargs)

    criterion = nn.CrossEntropyLoss()

    model = Net().to(device)
    if args.mode == 'test':
        model.load_state_dict(torch.load('model/cifar10_model.ckpt'))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loop through every epoch and train our model, evaluating the test accuracy on every epoch
    if args.mode == 'train':
        #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        print('Loop\t Train loss\t Train acc (%)\t Test Loss\t Test Acc (%)')
        for epoch in range(1, epochs + 1):
            train_acc, train_loss = train(args, model, device, train_loader, criterion, optimizer, epoch)
            #scheduler.step()
            test_acc, test_loss = test(model, device, test_loader, criterion)
            scheduler.step(test_loss)
            print(f'{epoch}/{epochs}\t  {train_loss:.4f}\t  {(train_acc/len(cifar_train))*100:.4f}\t  {test_loss:.4f}\t  {(test_acc/len(cifar_test))*100:.4f}')
    elif args.mode == 'test': # Test our trained model on a single image by doing a single forward pass
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        cv_image = cv2.imread(args.image)
        if cv_image is None:
            print('Error, no input image detected')
            exit()
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv_image_resized = cv2.resize(cv_image, (32, 32), cv2.INTER_LINEAR)
        processed_image = test_transform(cv_image_resized)
        batch_image = torch.unsqueeze(processed_image, 0) 
        inference(model, device, batch_image, classes)

    # Save our model once its trained
    if args.mode == 'train':
        model_name = 'model/cifar10_model.ckpt'
        Path('model/').mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_name)
        print(f'Model saved in file: {model_name}')
        

if __name__ == '__main__':
    main()