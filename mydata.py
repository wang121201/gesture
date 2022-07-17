from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self,dimages,class_labels, transform=None):
        images = []
        labels = []
        for img,lab in dimages:
            images.append(img)
            labels.append(lab)
        self.class_num = len(class_labels)
        # self.encode_labels = dict(zip(class_labels,torch.arange(0,self.class_num)))
        self.encode_labels = dict(zip(class_labels,torch.eye(self.class_num,self.class_num)))
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        label = self.encode_labels[self.labels[index]]
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等
        return img, label

    def __len__(self):
        return len(self.labels)


    # def showImg(self,head):
    #     count = 1
    #     f = plt.figure(figsize=(40, 40))
    #     for index in range(self.class_num - 1):
    #         loc = head + index*200
    #         img,label = self.__getitem__(loc)
    #         label = self.labels[loc]
    #         f.add_subplot(3, 3, count)
    #         img =  transforms.ToPILImage()(img).convert('RGB')
    #         plt.imshow(img)
    #         plt.title(label, fontsize=10)
    #         plt.xticks([])
    #         plt.yticks([])
    #         count = count + 1
    #     plt.suptitle("Hand Sign Images", size=24)
    #     plt.show()
