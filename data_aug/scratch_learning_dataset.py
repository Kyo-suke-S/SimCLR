from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import os

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
#print(parent_dir)


def get_stl10_data_loaders(download, data_dir, shuffle=True, batch_size=256):
  #data_dir = parent_dir+"/datasets"
  train_dataset = datasets.STL10(data_dir, split='train', download=download,
                                  transform=transforms.ToTensor())

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  
  test_dataset = datasets.STL10(data_dir, split='test', download=download,
                                  transform=transforms.ToTensor())

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader

def get_cifar10_data_loaders(download, data_dir, shuffle=True, batch_size=256):
  #data_dir = parent_dir + "/cifer-10_data"
  train_dataset = datasets.CIFAR10(data_dir, train=True, download=download,
                                  transform=transforms.ToTensor())

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=0, drop_last=False, shuffle=shuffle)
  
  test_dataset = datasets.CIFAR10(data_dir, train=False, download=download,
                                  transform=transforms.ToTensor())

  test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                            num_workers=10, drop_last=False, shuffle=shuffle)
  return train_loader, test_loader