import torch
from torch import nn, optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import numpy as np
from CBAMResnet18 import CBAMResnet18
from visdom import Visdom
'''模型训练'''

batch_size = 256
epochs = 100
learning_rate = 1e-1
seed = 123456
torch.manual_seed(seed)
device = torch.device("cuda:0")

'''猫狗数据集'''
# data_path = "./train"
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])
# ])
#
# dataset = MyDataset(data_path, transform)
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

'''cifar100数据集'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
train_loader = DataLoader(datasets.CIFAR100("../data", train=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.CIFAR100("../data", train=False, transform=transform), batch_size=batch_size, shuffle=True)

'''数据可视化'''
# imgs, _ = next(iter(train_loader))
# viz = Visdom()
# viz.images(imgs, win="train")
# exit(0)

''' two classification'''
net = CBAMResnet18(100)
# net = SEResnet18(2)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
net.to(device)
criterion.to(device)
total_loss = []
viz = Visdom()
viz.line([[0., 0.]], [0.], win="train", opts=dict(title="train&&val loss",
                                                  legend=['train', 'val']))
for epoch in range(epochs):
    net.train()
    total_loss.clear()
    for batch, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        logits = net(input)
        loss = criterion(logits, label)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch%50==0:
            print("epoch:{} batch:{} loss:{} lr:{}".format(epoch, batch, loss.item(), optimizer.state_dict()['param_groups'][0]['lr']))
    scheduler.step()

    net.eval()
    correct = 0
    test_loss = 0
    for input, label in test_loader:
        input, label = input.to(device), label.to(device)
        logits = net(input)

        '''crossentropy'''
        test_loss += criterion(logits, label).item() * input.shape[0]
        pred = logits.argmax(dim=1)

        correct += pred.eq(label).float().sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    viz.line([[float(np.mean(total_loss)), test_loss]], [epoch], win="train", update="append")
    if (epoch+1)%5==0:
        torch.save(net.state_dict(), "resnet18_{}.pkl".format(epoch))


