from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import PIL


class MultiMNIST(Dataset):
    ''' Loads the MultiMNIST dataset given root directory and 
        path to csv file with one-hot encoding of groundtruth labels.
        
        Expects folder structure to be:
            root/
            |-- train/
            |---- image1.png
            |---- image2.png
            |---- ...
            |-- val/
            |---- ...
            |-- test/
            |---- ...
            |-- train.csv
            |-- val.csv
            |-- test.csv
            
    
        Args:
            - root: root path to dataset where images and csv files are stored
            - mode: corresponds to folder/file name, e.g. root/<mode>.csv and root/<mode>/
            - transform: optional torchvision image transformer
    '''
    
    def __init__(self, root='./data/double_mnist_seed_123_image_size_64_64/', mode='train', transform=None):
        super().__init__()
        self.root = root
        self.df = pd.read_csv(root+mode+'.csv', sep=';')
        self.mode = mode
        self.transform = transform
        
        
    def __len__(self):
        return len(self.df)
        
        
    def __getitem__(self, index):
        filepath = self.df.iloc[index,1]
        img = Image.open('{}{}/{}'.format(self.root, self.mode, filepath))
        
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img).float()
        
        targets = torch.tensor(self.df.iloc[index,2:].to_numpy(dtype=np.float32))
        
        return img, targets
        
        