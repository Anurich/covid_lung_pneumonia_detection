import torch
import torch.nn as nn
import pandas as pd
import os
from einops import rearrange
import cv2
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split

def savefiles(dataX, dataY, name):
    data =  pd.DataFrame(pd.concat((pd.Series(dataX), pd.Series(dataY)), axis=1, keys=["images", "labels"]))
    data.to_csv(name)

def preprocessing():
    # let's prepare the csv file and word to index mapping
    folderName = 'COVID-19_Radiography_Dataset/'
    folder_list = os.listdir(folderName)
    word_to_index = {v:k for k, v in enumerate(folder_list)}
    index_to_word = {v:k for k, v in word_to_index.items()}
    df = pd.DataFrame(columns=["image", "labels"])
    i=0
    for name in folder_list:
        folder = os.path.join(folderName, name)
        list_of_images = os.listdir(folder)
        for im_name in list_of_images:
            image_path = os.path.join(folder, im_name)
            df.loc[i] = [image_path, word_to_index.get(name)]
            i+=1

    # let's shuffle and than split the data in train test and validation
    df = df.sample(frac=1)
    x_train, x_test, y_train, y_test = train_test_split(df["image"], df["labels"], test_size=0.3, stratify=df["labels"])
    x_train, x_val,  y_train, y_val  = train_test_split(x_train, y_train, test_size=0.3,stratify = y_train)
    savefiles(x_train, y_train, "train.csv")
    savefiles(x_val, y_val, "validation.csv")
    savefiles(x_test, y_test, "test.csv")

#preprocessing()

def read_csv(filename):
    # reading training and test data
    return pd.read_csv(filename)


class dataset(torch.utils.data.Dataset):
    def __init__(self, dataset,  batch_size, transformer=None):
        super().__init__()
        self.dataset = dataset
        self.transformer = transformer
        self.batch_size = batch_size

    def readImages(self,xData):
        img = cv2.imread(xData)
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img =  rearrange(img, 'h w c -> c h w')
        return img

    def __getitem__(self,idx):
        data = self.dataset.loc[idx]
        xData, yData = data["images"], data["labels"]
        x_images =  self.readImages(xData)
        # we need to read the image
        if self.transformer != None:
            x_images = self.transformer(x_images)
        return x_images, torch.Tensor([yData])

    def __len__(self):
        return len(self.dataset)


def load_data(batch_size = 32, is_train="train", transforms = None):
    data  = None
    if is_train.lower() in "train":
        data = read_csv("train.csv")
    elif is_train.lower() in "validation":
        data = read_csv("validation.csv")
    else:
        data= read_csv("test.csv")

    all_label_data =  torch.Tensor(data["labels"].values).long()
    label_unique, counts  = np.unique(data["labels"], return_counts =True)
    # now we calculate the weight for the class
    class_weights  = [sum(counts)/ c for c in counts]
    assign_weights = [class_weights[e] for e in data["labels"]]
    sampler = WeightedRandomSampler(assign_weights, len(data["labels"]))
    dd = dataset(data, batch_size, transforms)
    dataloader = DataLoader(dd, sampler=sampler, batch_size=batch_size)
    return dataloader


'''
from torchvision import transforms
trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)), transforms.ToTensor()])
data = load_data(2, False, trans)
try:
    for d in data:
        x,y  = d
        print(x.shape, y.shape)
except:
    pass
'''
