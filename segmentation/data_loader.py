#import matplotlib.pyplot as plt
import os
import imageio
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import image
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image, ImageDraw
import json
from skimage.draw import polygon, polygon2mask
#from torch.utils.data import dataset
from augmentation import GeometryAugmentation, IntensityAugmentation

class myDataLoader(torch.utils.data.Dataset):
    def __init__(self):

        self.images = []
        self.labels = []
        self.patient_id = []
        self.radius_pts = []
        self.data_dir = './Data/XR_WRIST'

        for dir1 in os.listdir(self.data_dir):

            for dir2 in os.listdir(os.path.join(self.data_dir, dir1)):
                for file in os.listdir(os.path.join(self.data_dir, dir1, dir2)):
                    file_path = os.path.join(self.data_dir, dir1, dir2, file)

                    # print(self.patient_id)
                    '''
                    if file.endswith('.png'):
                        image = Image.open(file_path)
                        #newsize = (512, 512)
                        #image = image.resize(newsize)

                        #sz = transforms.Resize((512,512))
                        #image = sz(image)
                        channel = transforms.Grayscale()
                        image = channel(image)
                        tnsr = transforms.ToTensor()
                        sz_x,sz_y = image.size
                        #sz_y = image.shape[2]
                        #sz_x = image.shape[1]
                        print(sz_x)
                        print(sz_y)
                        x_pad_1 = int(np.floor((512 - sz_x) / 2))
                        x_pad_2 = int(np.ceil((512 - sz_x) / 2))

                        y_pad_1 = int(np.floor((512 - sz_y) / 2))

                        y_pad_2 = int(np.ceil((512 - sz_y) / 2))

                        image = tnsr(image)
                        pad = torch.nn.ZeroPad2d((x_pad_1,x_pad_2,y_pad_1,y_pad_2))
                        image = pad(image)

                        #print(self.x)
                        #pil = transforms.ToPILImage()
                        #image = pil(image)

                        #image = tnsr(image)
                        #image = image[0, ...]

                        self.images.append(image)

                    '''

                    if file.endswith('.json'):
                        self.patient_id.append(dir1)
                        with open(file_path, 'r') as file:
                            lab = json.load(file)
                            for entry in range(7):
                                if lab['shapes'][entry]['label'] == 'Wrist Joint Line':
                                    pts = lab['shapes'][entry]['points']

                                if lab['shapes'][entry]['label'] == 'Radius':
                                    radius_pts = lab['shapes'][entry]['points']

                            Ip = None
                            Ip = lab['imagePath']
                            # print(Ip)
                            Imagepth = os.path.join(self.data_dir, dir1, dir2, Ip)
                            image = Image.open(Imagepth)
                            channel = transforms.Grayscale()
                            image = channel(image)
                            tnsr = transforms.ToTensor()
                            sz_x, sz_y = image.size

                            x_pad_1 = int(np.floor((512 - sz_x) / 2))
                            x_pad_2 = int(np.ceil((512 - sz_x) / 2))

                            y_pad_1 = int(np.floor((512 - sz_y) / 2))

                            y_pad_2 = int(np.ceil((512 - sz_y) / 2))

                            image = tnsr(image)
                            pad = torch.nn.ZeroPad2d((x_pad_1, x_pad_2, y_pad_1, y_pad_2))
                            image = pad(image)
                            self.images.append(image)

                            pts[1][0] += x_pad_1
                            pts[1][1] += y_pad_1
                            pts[0][0] += x_pad_1
                            pts[0][1] += y_pad_1
                            pts = np.transpose(pts)
                            self.labels.append(pts)
                            for p in radius_pts:
                                p[0] += x_pad_1
                                p[1] += y_pad_1
                            self.radius_pts.append(radius_pts)


    def __getitem__(self, i):
        img = torch.zeros((512, 512), dtype=torch.int32)
        dict = self.labels[i]
        poly = np.asarray(self.radius_pts[i])
        trx = transforms.ToTensor()
        poly = trx(poly)
        r = torch.round(poly[:,:,1]).type(torch.int16)
        c = torch.round(poly[:,:,0]).type(torch.int16)
        rr, cc = polygon(r[0,:],c[0,:], shape=(512,512))
        img[rr,cc] = 1
        img = img.type(torch.float)
        #t = torch.Tensor(dict)
        radius_pts = torch.empty((3,2))
        poly1 = torch.Tensor(self.radius_pts[i])
        shape = poly1.size()
        radius_pts[0] = poly1[0,:]
        radius_pts[1] = poly1[-1,:]
        mid = np.ceil(shape[0]/2)
        mid = int(mid)
        radius_pts[2] = poly1[mid,:]

        dict_out = {}
        t = torch.Tensor(dict).reshape((1, 1, 4))
        a = 'images'
        b = 'points'
        c = 'patient_id'
        d = 'radius'
        e = 'radius_pts'
        dict_out[a] = self.images[i]
        dict_out[b] = t
        dict_out[c] = self.patient_id[i]
        dict_out[d] = img
        dict_out[e] = radius_pts
        return dict_out

    def __len__(self):
        return len(self.patient_id)



class testtool():
    def __init__(self):
        self.len = 0


    def test(self):
        data1 = myDataLoader()
        item = data1.__getitem__(0)
        self.len= data1.__len__()
        g_aug = GeometryAugmentation()
        i_aug = IntensityAugmentation()
        #print("the number of images are ", self.len)
        #pts = item['points']
        image = item['images']
        mask = item['radius']
        pts = item['radius_pts']
        pts = pts.numpy().T
        tf_item_p = g_aug.perform_augmentation(item)
        tf_image_p = tf_item_p['images']
        tf_mask_p = tf_item_p['radius']
        tf_item_cia = i_aug.perform_augmentation(item)
        tf_item = i_aug.perform_augmentation(tf_item_p)
        tf_image_cia = tf_item_cia['images']
        tf_mask_cia = tf_item_cia['radius']
        tf_image = tf_item['images']
        tf_mask = tf_item['radius']

        fig, axs = plt.subplots(2, 2)
        trx = transforms.ToPILImage()
        image = trx(image)
        tf_image_p = trx(tf_image_p)
        tf_image_cia = trx(tf_image_cia)
        tf_image = trx(tf_image)

        #t = torch.empty(2,2)
        #t = pts.reshape((2,2)).numpy().T
        #a = t[0]
        #b = t[1]
        #a =[t[0][0], t[1][0]]
        #b = [t[0][1],t[1][1]]
        #print(t)
        #print(a)
        #print(b)
        #plt.figure(figsize=(10, 10))
        #plt.plot(a, b, color="white", linewidth=1)
        #plt.plot(a, b, marker='o', color="white")

        #plt.plot(mask)
        axs[0,0].plot(pts[0],pts[1],marker='o', color='white')
        axs[0, 0].imshow(image)
        axs[0, 0].imshow(mask, alpha=0.5)
        axs[0, 0].set_title("original")
        axs[0, 1].imshow(tf_image_p)
        axs[0, 1].imshow(tf_mask_p, alpha=0.5)
        axs[0, 1].set_title("geometric_aug")
        axs[1, 0].imshow(tf_image_cia)
        axs[1, 0].imshow(tf_mask_cia, alpha=0.5)
        axs[1, 0].set_title("intensity_aug")
        axs[1, 1].imshow(tf_image)
        axs[1, 1].imshow(tf_mask, alpha=0.5)
        axs[1, 1].set_title("augmented")
        fig.tight_layout()
        plt.show()

        return 0



#test1 = testtool()
#test1.test()

def Kfold(n, list):
    length = len(list)
    return [list[i * length // n: (i + 1) * length // n] for i in range(n)]

class data_kfold():
    def __init__(self):
        self.len = 0


    def kfold_split(self,data):
        patient_ids_list = []
        data2 = data
        lent = data2.__len__()
        #print(lent)
        for i in range(lent):
            item = data2.__getitem__(i)
            patient_ids_list.append(item['patient_id'])
        patient_ids_list = Kfold(5,patient_ids_list)
        #print(len(patient_ids_list[0]))
        #same patient ID same fold
        for i in range(len(patient_ids_list)-1):
            j=i+1
            for k in patient_ids_list[i]:
                for l in patient_ids_list[j]:

                    if k == l:
                         patient_ids_list[i].append(l)
                         patient_ids_list[j].remove(l)




        total = sum(len(x) for x in patient_ids_list)
        idx = np.zeros(total)
        #print(total)
        for y in range(total):
            idx[y] = y
        #print(idx)
        idx2 = patient_ids_list
        start = 0
        for i in idx2:
            for j in range(len(i)):

                i[j] = start
                start = start+ 1

        #print(idx2)
        train_idx = np.concatenate(idx2[0:3])
        valid_idx = idx2[3]
        test_idx = idx2[4]
        #print(valid_idx)
        train = torch.utils.data.Subset(data2,train_idx)
        valid = torch.utils.data.Subset(data2,valid_idx)
        test = torch.utils.data.Subset(data2, test_idx)

        #print(train.__len__())
        #print(valid.__len__())
        #print(test.__len__())

        return train,valid,test



#print(x.size())
#print(x)
#tn = transforms.ToPILImage()
#x= tn(x['images'])



#plt.imshow(x, interpolation='nearest')
#plt.show()
'''
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

class CNN(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d((2,2), stride=(2,2))
        self.fc = nn.Linear(64*64*128,4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

    def training_step(self, batch, batch_nb):
        x = batch['images']
        print(x.size())
        y = batch['points']
        print(y.size())
        loss = F.l1_loss(self(x),y )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch.cuda

#import CNN
#import Dataset
model = CNN()
data1 = myDataLoader()
train_loader = DataLoader(data1, batch_size=8)
trainer = Trainer(
    accelerator="auto",
    devices=1
    if torch.cuda.is_available() else None,
    max_epochs = 10 )
trainer.fit(model, train_loader)
'''


