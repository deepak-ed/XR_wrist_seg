
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
import numpy
import torch
from torchvision import datasets, transforms
from PIL import Image
import json
from skimage import transform
from skimage import img_as_float
#from torch.utils.data import dataset



class myDataLoader(torch.utils.data.Dataset):
    def __init__(self):
        #self.x = 0
        self.images = []
        self.labels = []
        self.data_dir = './Data/XR_WRIST'
        for dir1 in os.listdir(self.data_dir):
            #print(dir1)
            for dir2 in os.listdir(os.path.join(self.data_dir, dir1)):
                for file in os.listdir(os.path.join(self.data_dir,dir1,dir2)):
                    file_path = os.path.join(self.data_dir,dir1,dir2,file)


                    if file.endswith('.json'):
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
        a = 'images'
        b = 'points'
        dict_out[a] = self.images[i]
        dict_out[b] = t



        return dict_out

    def __len__(self):
        return len(self.images)



class GeometryAugmentation():
    def __init__(self):
        self.min_translation = -100
        self.max_translation = 100
        self.min_rotation = -45
        self.max_rotation = 45
        self.min_scaling = 0.5
        self.max_scaling = 2.0

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
        hmap = item['heat_map']
        #radius_mask = item['radius']
        #radius_pts = item['radius_pts']
        #torch.Tensor.cpu(image)
        image = image.cpu()
        hmap = hmap.cpu()
        #radius_mask = radius_mask.cpu()
        trx = transforms.ToPILImage()
        image = trx(image)
        hmap_start = trx(hmap[0])
        hmap_end = trx(hmap[1])
        #radius_mask = trx(radius_mask)
        image = img_as_float(image)
        #radius_mask = img_as_float(radius_mask)
        hmap_start = img_as_float(hmap_start)
        hmap_end = img_as_float(hmap_end)
        pts = pts.cpu()
        pts = np.array(pts)
        matrix = np.array(matrix)
        tform = transform.AffineTransform(matrix)
        tf_img = transform.warp(image, tform.inverse)
        tf_hmap_start = transform.warp(hmap_start, tform.inverse)
        tf_hmap_end = transform.warp(hmap_end,tform.inverse)
        tx = transforms.ToTensor()
        tf_img = tx(tf_img)
        #tf_radius_mask = tx(tf_radius_mask)
        tf_hmap_start = tx(tf_hmap_start)
        tf_hmap_end = tx(tf_hmap_end)
        tf_img = torch.reshape(tf_img,(1,512,512))
        hmap[0]  =tf_hmap_start
        hmap[1] = tf_hmap_end
        #tf_radius_mask = torch.reshape(tf_radius_mask,(512,512))
        pts = pts.reshape((2,2))
        p = np.empty((2,2))
        #shape = radius_pts.size()
        #r_p = torch.empty(shape)
        p[0][0] = pts[0][0] * matrix[0][0] + pts[1][0] * matrix[0][1] + matrix[0][2]
        p[1][0] = pts[0][0] * matrix[1][0] + pts[1][0] * matrix[1][1] + matrix[1][2]
        p[0][1] = pts[0][1] * matrix[0][0] + pts[1][1] * matrix[0][1] + matrix[0][2]
        p[1][1] = pts[0][1] * matrix[1][0] + pts[1][1] * matrix[1][1] + matrix[1][2]
        #r_p[0] = radius_pts[0] * matrix[0][0] + radius_pts[1] * matrix[0][1] + matrix[0][2]
        #r_p[1] = radius_pts[0] * matrix[1][0] + radius_pts[1] * matrix[1][1] + matrix[1][2]

        p = torch.tensor(p)
        #p = p.cuda() 
        p = torch.reshape(p,(1,1,4))
        it = {}

        it['images'] = tf_img
        it['points'] = p
        it['patient_id'] = pid
        it['heat_map'] = hmap
        #it['radius_pts'] = r_p


        return it

    def perform_augmentation(self,item):
            while True:

                trans = np.random.randint(self.min_translation,self.max_translation,2,dtype=int)
                angle = np.random.randint(self.min_rotation,self.max_rotation,dtype=int)
                factor = np.random.uniform(self.min_scaling,self.max_scaling)
                matrix = GeometryAugmentation.get_transformation(self,trans,angle,(factor,factor))
                tf_item = GeometryAugmentation.transform(self,item,matrix)

                pts = tf_item['points']
                #rds_pts = tf_item['radius_pts']
                pts = torch.reshape(pts, (1, 4))

                flag = 1
                for i in pts:
                    for j in i:

                        if j < 0 or j > 512:
                            flag = 0

                if flag == 1:
                    break
                    
                '''

                flag2 = 1

                for i in rds_pts:
                    for j in i:

                        if j < 0 or j > 512:
                            flag2 = 0

                if flag2 == 1:
                    break
                    
                '''





            return tf_item


class IntensityAugmentation():
    def __init__(self):
        self.min_contrast_change = 0.5
        self.max_contrast_change = 2.0
        self.prob_inversion = 0.5

    def contrast_change(self,factor, item):
        image = item['images']
        pts = item['points']
        pid = item['patient_id']
        hmap = item['heat_map']
        #radius_pts = item['radius_pts']
        image = image/255
        image = factor*(image - image.mean()) + image.mean()
        image = image *255
        it = {}
        it['images'] = image
        it['points'] = pts
        it['patient_id'] = pid
        it['heat_map'] = hmap
        #it['radius_pts'] = radius_pts

        #check pts


        return it

    def invert(self, item):

        image = item['images']
        pts= item['points']
        pid = item['patient_id']
        hmap = item['heat_map']
        #radius_mask = item['radius']
        #radius_pts = item['radius_pts']
        image = 255-image
        it = {}
        it['images'] = image
        it['points'] = pts
        it['patient_id'] = pid
        it['heat_map'] = hmap
        #it['radius_pts'] = radius_pts

        #check points

        return it

    def normalise(self, item):

        image = item['images']
        pts =item['points']
        pid = item['patient_id']
        hmap = item['heat_map']
        #radius_mask = item['radius']
        #radius_pts = item['radius_pts']
        image = (image - image.mean())/image.std()

        it = {}

        it['images'] = image
        it['points']  =pts
        it['patient_id'] = pid
        it['heat_map'] = hmap
        #it['radius_pts'] = radius_pts
        return it

    def perform_augmentation(self, item):

            factor = np.random.uniform(self.min_contrast_change,self.max_contrast_change)
            item = IntensityAugmentation.contrast_change(self,factor,item)
            prob_iv = np.random.uniform()
            if prob_iv < self.prob_inversion:
                item = IntensityAugmentation.invert(self,item)

            tf_item = IntensityAugmentation.normalise(self,item)

            return tf_item



'''
class testtool():
    def __init__(self):
        self.hundred = 0
        self.len = 0


    def test(self):
        data1 = myDataLoader()
        item = data1.__getitem__(200)
        self.len= data1.__len__()
        #print(item)
        g_aug = GeometryAugmentation()
        i_aug = IntensityAugmentation()
        trans_mtx_t = g_aug.get_transalation((50,100))
        trans_mtx_r = g_aug.get_rotation(45)
        trans_mtx_s = g_aug.get_scaling(0.5)
        trans_mtx_c = g_aug.get_transformation((50,100),45,0.8)
        tf_item_s = g_aug.transform(item,trans_mtx_s)
        tf_item_r = g_aug.transform(item,trans_mtx_r)
        tf_item_t = g_aug.transform(item,trans_mtx_t)
        tf_item_c = g_aug.transform(item,trans_mtx_c)
        tf_item_p = g_aug.perform_augmentation(item)
        tf_item_cc = i_aug.contrast_change(1.5,item)
        tf_item_iv = i_aug.invert(item)
        tf_item_nm = i_aug.normalise(item)
        tf_item_cia = i_aug.perform_augmentation(item)
        trx = transforms.ToPILImage()
        pts_o = item['points']
        image_o = trx(item['images'])
        pts_s = tf_item_s['points']
        image_s = trx(tf_item_s['images'])
        pts_r = tf_item_r['points']
        image_r = trx(tf_item_r['images'])
        pts_t = tf_item_t['points']
        image_t = trx(tf_item_t['images'])
        pts_c = tf_item_c['points']
        image_c = trx(tf_item_c['images'])
        pts_p = tf_item_p['points']
        image_p = trx(tf_item_p['images'])
        pts_cc = tf_item_cc['points']
        image_cc = trx(tf_item_cc['images'])
        pts_iv = tf_item_iv['points']
        image_iv = trx(tf_item_iv['images'])
        pts_nm = tf_item_nm['points']
        image_nm = trx(tf_item_nm['images'])
        pts_cia = tf_item_cia['points']
        image_cia = trx(tf_item_cia['images'])
        fig, axs = plt.subplots(2,5)
        axs[0, 0].plot(pts_o[0], pts_o[1], color="white", linewidth=1)
        axs[0, 0].plot(pts_o[0], pts_o[1], marker='o', color="white")
        axs[0, 0].set_title("original")
        axs[0, 0].imshow(image_o)
        axs[0, 1].plot(pts_t[0], pts_t[1], color="white", linewidth=1)
        axs[0, 1].plot(pts_t[0], pts_t[1], marker='o', color="white")
        axs[0, 1].set_title("translated")
        axs[0, 1].imshow(image_t)
        axs[0, 2].plot(pts_r[0], pts_r[1], marker='o', color="white")
        axs[0, 2].plot(pts_r[0], pts_r[1], color="white", linewidth=1)
        axs[0, 2].set_title("rotated")
        axs[0, 2].imshow(image_r)
        axs[0, 3].plot(pts_s[0], pts_s[1], color="white", linewidth=1)
        axs[0, 3].plot(pts_s[0], pts_s[1], marker='o', color="white")
        axs[0, 3].set_title("scaled")
        axs[0, 3].imshow(image_s)
        axs[0, 4].plot(pts_c[0], pts_c[1], color="white", linewidth=1)
        axs[0, 4].plot(pts_c[0], pts_c[1], marker='o', color="white")
        axs[0, 4].set_title("Combined Geometric Aug")
        axs[0, 4].imshow(image_c)
        axs[1, 0].plot(pts_p[0], pts_p[1], color="white", linewidth=1)
        axs[1, 0].plot(pts_p[0], pts_p[1], marker='o', color="white")
        axs[1, 0].set_title("Random Geometric Aug")
        axs[1, 0].imshow(image_p)
        axs[1, 1].plot(pts_cc[0], pts_cc[1], color="white", linewidth=1)
        axs[1, 1].plot(pts_cc[0], pts_cc[1], marker='o', color="white")
        axs[1, 1].set_title("Contrast Change")
        axs[1, 1].imshow(image_cc)
        axs[1, 2].plot(pts_iv[0], pts_iv[1], color="white", linewidth=1)
        axs[1, 2].plot(pts_iv[0], pts_iv[1], marker='o', color="white")
        axs[1, 2].set_title("Inversion")
        axs[1, 2].imshow(image_iv)
        axs[1, 3].plot(pts_nm[0], pts_nm[1], color="white", linewidth=1)
        axs[1, 3].plot(pts_nm[0], pts_nm[1], marker='o', color="white")
        axs[1, 3].set_title("Normalised")
        axs[1, 3].imshow(image_nm)
        axs[1, 4].plot(pts_cia[0], pts_cia[1], color="white", linewidth=1)
        axs[1, 4].plot(pts_cia[0], pts_cia[1], marker='o', color="white")
        axs[1, 4].set_title("Random Intensity Aug")
        axs[1, 4].imshow(image_cia)

        fig.tight_layout()
        plt.show()

        return 0

test1 = testtool()
test1.test()
#print(x.size())
#print(x)
#tn = transforms.ToPILImage()
#x= tn(x['images'])
'''



