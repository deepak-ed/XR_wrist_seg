
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
import numpy
import torch
from torchvision import datasets, transforms, utils
from PIL import Image, ImageOps
import json
from skimage import transform
from skimage import img_as_float
import matplotlib.backends.backend_agg as plt_backend_agg
#from torch.utils.data import dataset



class myDataLoader(torch.utils.data.Dataset):
    def __init__(self):
        #self.x = 0
        self.images = []
        self.labels = []
        self.patient_id = []
        self.data_dir = './Data/XR_WRIST'
        patient_index = -1
        for dir1 in os.listdir(self.data_dir):
            #patient_index = patient_index + 1
            #print(dir1)


            for dir2 in os.listdir(os.path.join(self.data_dir, dir1)):
                for file in os.listdir(os.path.join(self.data_dir,dir1,dir2)):
                    file_path = os.path.join(self.data_dir,dir1,dir2,file)

                    #print(self.patient_id)
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
                            Ip = None
                            Ip = lab['imagePath']
                            #print(Ip)
                            Imagepth = os.path.join(self.data_dir,dir1,dir2,Ip)
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


    def __getitem__(self,i):
        dict = self.labels[i]
        t = torch.Tensor(dict)
        dict_out = {}
        t = torch.Tensor(dict).reshape((1, 1, 4))
        a = 'images'
        b = 'points'
        c = 'patient_id'
        dict_out[a] = self.images[i]
        dict_out[b] = t
        dict_out[c] = self.patient_id[i]



        return dict_out

    def __len__(self):
        return len(self.patient_id)


def Kfold(n, list):
    length = len(list)
    return [list[i * length // n: (i + 1) * length // n] for i in range(n)]






class GeometryAugmentation():
    def __init__(self):
        self.min_translation = -100
        self.max_translation = 100
        self.min_rotation = -45
        self.max_rotation = 45
        self.min_scaling = 0.8
        self.max_scaling = 1.5

    def get_transalation(self, trans):

        tf = transform.AffineTransform(translation=trans)

        #matrix = [[1,1,trans[0]],[1,1,trans[1]],[0,0,1]]


        return tf.params

    def get_rotation(self, angle):

        tf = transform.AffineTransform(rotation=angle)

        #matrix = [[np.cos(angle), np.sin(angle),0],[np.cos(angle), np.sin(angle),0],[0,0,1]]

        return tf.params

    def get_scaling(self,factor):

        tf = transform.AffineTransform(scale=factor)

        #matrix = [[factor, factor,0],[factor,factor,0],[0,0,1]]
        return tf.params

    def get_transformation(self, trans, angle, factor):

        tf = transform.AffineTransform(scale=factor,rotation=angle,translation=trans)

        #matrix = [[np.cos(angle)*factor, np.sin(angle)*factor, trans[0]], [np.cos(angle)*factor, np.sin(angle)*factor, trans[1]],[0,0,1]]

        return tf.params



    def transform(self,item,matrix):
        image = item['images']
        pts = item['points']
        pid = item['patient_id']
        #torch.Tensor.to('cpu')
        trx = transforms.ToPILImage()
        image = trx(image)
        image = img_as_float(image)
        pts = np.array(pts)
        matrix = np.array(matrix)
        tform = transform.AffineTransform(matrix)
        tf_img = transform.warp(image, tform.inverse)
        tx = transforms.ToTensor()
        #torch.Tensor.cuda()
        tf_img = tx(tf_img)
        tf_img = torch.reshape(tf_img,(1,512,512))
        pts = pts.reshape((2,2))

        #print(tf_img.size())
        p = np.empty((2,2))
        p[0][0] = pts[0][0]*matrix[0][0] + pts[1][0]*matrix[0][1] + matrix[0][2]
        p[1][0] = pts[0][0] * matrix[1][0] + pts[1][0] * matrix[1][1] + matrix[1][2]
        p[0][1] = pts[0][1] * matrix[0][0] + pts[1][1] * matrix[0][1] + matrix[0][2]
        p[1][1] = pts[0][1] * matrix[1][0] + pts[1][1] * matrix[1][1] + matrix[1][2]
        p = torch.tensor(p)
        p = torch.reshape(p,(1,1,4))
        it = {}

        it['images'] = tf_img
        it['points'] = p
        it['patient_id'] = pid


        return it

    def perform_augmentation(self,item):
            while True:

                trans = np.random.randint(self.min_translation,self.max_translation,2,dtype=int)
                angle = np.random.randint(self.min_rotation,self.max_rotation,dtype=int)
                factor = np.random.uniform(self.min_scaling,self.max_scaling)
                matrix = GeometryAugmentation.get_transformation(self,trans,angle,factor)
                tf_item = GeometryAugmentation.transform(self,item,matrix)
                pts = tf_item['points']
                pts = torch.reshape(pts, (1, 4))
                flag = 1
                for i in pts:
                    for j in i:

                        if j < 0 or j > 512:
                            flag = 0

                if flag == 1:
                    break



            return tf_item


class IntensityAugmentation():
    def __init__(self):
        self.min_contrast_change = 0.5
        self.max_contrast_change = 2
        self.prob_inversion = 0.5

    def contrast_change(self,factor, item):
        image = item['images']
        pts = item['points']
        pid = item['patient_id']
        image = image/255
        image = factor*(image - image.mean()) + image.mean()
        image = image *255
        it = {}
        it['images'] = image
        it['points'] = pts
        it['patient_id'] = pid

        #check pts


        return it

    def invert(self, item):

        image = item['images']
        pts= item['points']
        pid = item['patient_id']


        trx = transforms.ToPILImage()
        image = trx(image)
        image = ImageOps.invert(image)
        tx = transforms.ToTensor()

        image = tx(image)
        #image = iv(image)

        it = {}

        it['images'] = image
        it['points'] = pts
        it['patient_id'] = pid

        #check points

        return  it

    def normalise(self, item):

        image = item['images']
        pts =item['points']
        pid = item['patient_id']

        image = (image - image.mean())/image.std()

        it = {}

        it['images'] = image
        it['points']  =pts
        it['patient_id'] = pid
        return it

    def perform_augmentation(self, item):

            factor = np.random.uniform(self.min_contrast_change,self.max_contrast_change)
            item = IntensityAugmentation.contrast_change(self,factor,item)
            prob_iv = np.random.uniform()
            if prob_iv < self.prob_inversion:
                item = IntensityAugmentation.invert(self,item)

            tf_item = IntensityAugmentation.normalise(self,item)

            return tf_item




class testtool():
    def __init__(self):
        self.hundred = 0
        self.len = 0


    def test(self):
        data1 = myDataLoader()
        item = data1.__getitem__(0)
        self.len= data1.__len__()
        train,valid, test = torch.utils.data.random_split(data1, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42))
        print(train.__len__())
        print(valid.__len__())
        print(test.__len__())


    def test_kfold(self):
        patient_ids_list = []
        data2 = myDataLoader()
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

        print(train.__len__())
        print(valid.__len__())
        print(test.__len__())

        return train,valid,test







test1 = testtool()
test1.test_kfold()



#print(x.size())
#print(x)
#tn = transforms.ToPILImage()
#x= tn(x['images'])




from resnet_flex import myresnet18
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import io
#from model import testtool

class Res(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.m = myresnet18(**kwargs)
        self.ga = GeometryAugmentation()
        self.ia = IntensityAugmentation()
    def forward(self, x):

        return self.m.forward(x)

    def training_step(self, batch, batch_nb):
        x = batch['images']
        print(x.size())
        y = batch['points']

        #y_hat = self.m(x)
        print(y.size())
        loss = F.mse_loss(self(x).flatten(),y.flatten())
        return loss

    '''

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        for i,j,k in zip(batch['images'],batch['points'],batch['patient_id']) :
            item = {}
            item['images'] = i
            item['points'] = j
            item['patient_id']=k
            item = self.ga.perform_augmentation(item)
            item = self.ia.perform_augmentation(item)
            i = item['images']
            j = item['points']
            k = item['patient_id']

        return batch
        
        '''


    def validation_step(self, batch, batch_nb):
        x = batch['images']
        print(x.size())
        y = batch['points']
        y_hat = self.m(x)
        print(y.size())
        loss = F.mse_loss(y_hat.flatten(), y.flatten())
        return loss

    def test_step(self, batch, batch_nb):
        ten = []
        x = batch['images']
        y = batch['points']
        pid = batch['patient_id']
        y_hat = self.m(x)
        loss = F.mse_loss(y_hat.flatten(), y.flatten())
        for i, j, k,l in zip(x,y,y_hat,pid):
            t1 = torch.empty((2,2))
            t1 = j.reshape((2, 2)).numpy()
            t2 = torch.empty((2, 2))
            t2 = k.reshape((2, 2)).numpy()
            im = i.reshape((512,512)).numpy()
            a = t1[0]
            b = t1[1]
            print (a,b)
            plt.plot(a, b, color="white", linewidth=1)
            plt.plot(a, b, marker='x', color="white")
            im1 = plt.imshow(im)
            #im1 = im1.numpy()
            c = t2[0]
            d = t2[1]
            print(c,d)
            plt.plot(c, d, color="red", linewidth=1)
            plt.plot(c, d, marker='x', color="red")
            figure = plt.imshow(im)
            plt.title(l)
            '''
            with io.BytesIO() as buffer:
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                image = Image.open(buffer)
                ar = np.asarray(image)
                buffer.close()
            trx = transforms.ToTensor()
            #trx1 = transforms.ToPILImage
            #fig = trx(ar)
            ten = trx(ten)
            #ten = np.asarray(ten)
            plt.figure().clear()

        tensorboard = self.logger.experiment
        tensorboard.add_image('img_tensor',ten)
        '''

            plt.show()
            k = k.T
            point1 = k[0]
            point2 = k[1]
            dist = np.linalg.norm(point1 - point2)
            #print(dist)

        self.log('test_loss',loss, on_step=True,prog_bar=True,logger=True)




        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch.cuda

model = Res.load_from_checkpoint(checkpoint_path='./epoch=287-step=5183.ckpt')
#data1 = myDataLoader()
train_data, valid_data, test_data = test1.test_kfold()
train_loader = DataLoader(train_data, batch_size=8)
valid_loader = DataLoader(valid_data, batch_size=8)
test_loader = DataLoader(test_data, batch_size=8)
trainer = Trainer(
    accelerator="auto",
    devices=1
    if torch.cuda.is_available() else None,
    max_epochs = 5 )
#trainer.fit(model, train_loader,valid_loader)
trainer.test(model, test_loader, verbose=True)



