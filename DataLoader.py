import torchvision
from torchvision.transforms import ToTensor

Class DataLoader():

    def __init__(dataset):
    
    traindata="torchvision.datasets."+str(dataset)+"(root="./data",train=True,download=True,transform=ToTensor)"
    testdata="torchvision.datasets."+str(dataset)+"(root="./data",train=False,download=True,transform=ToTensor)"

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=self.config.batch_size,
                                          shuffle=True)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=self.config.batch_size,
                                          shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")