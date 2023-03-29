#Model for segmentation task
import matplotlib.pyplot as plt
import torch

from data_loader import data_kfold,myDataLoader
from my_unet import myUNet
from augmentation import GeometryAugmentation, IntensityAugmentation

from pytorch_lightning import LightningModule
import torch.nn.functional as F
import torch.optim


class unet(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.m = myUNet(**kwargs)
        self.ga = GeometryAugmentation()
        self.ia = IntensityAugmentation()
    def forward(self, x):

        return self.m.forward(x)

    def training_step(self, batch, batch_nb):
        x = batch['images']
        print(x.size())
        y = batch['radius']

        #y_hat = self.m(x)
        print(y.size())
        loss = F.binary_cross_entropy_with_logits(self(x).flatten(),y.flatten())
        return loss

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        for i,j,k,l,m in zip(batch['images'],batch['points'],batch['patient_id'],batch['radius'],batch['radius_pts']) :
            item = {}
            item['images'] = i
            item['points'] = j
            item['patient_id']=k
            item['radius'] = l
            item['radius_pts'] = m
            item = self.ga.perform_augmentation(item)
            item = self.ia.perform_augmentation(item)
            i = item['images']
            j = item['points']
            k = item['patient_id']
            l = item['radius']
            m= item['radius_pts']

        return batch


    def validation_step(self, batch, batch_nb):
        x = batch['images']
        print(x.size())
        y = batch['radius']
        y_hat = self.m(x)
        print(y.size())
        loss = F.binary_cross_entropy(y_hat.flatten(), y.flatten())
        return loss

    def test_step(self, batch, batch_nb):
        #ten = []
        x = batch['images']
        y = batch['radius']
        pid = batch['patient_id']
        y_hat = self.m(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.flatten(), y.flatten())
        for i, j, k,l in zip(x,y,y_hat,pid):
            im = i.reshape((512,512)).numpy()
            fig,axs = plt.subplots(1,2)
            axs[0].imshow(im)
            axs[0].imshow(j, alpha=0.5)
            axs[0].set_title("original")
            im1 = i.reshape((512, 512)).numpy()
            k=k.reshape((512,512))
            axs[1].imshow(im1)
            axs[1].imshow(k, alpha=0.5)
            axs[1].set_title("pred")
            plt.title(l)
            fig.tight_layout()
            plt.show()


        self.log('test_loss',loss, on_step=True,prog_bar=True,logger=True)




        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch.cuda

#model = Res()
#model = Res.load_from_checkpoint(checkpoint_path='./epoch=2-step=5183.ckpt')
#data1 = myDataLoader()
model = unet.load_from_checkpoint(checkpoint_path='./epoch=247-step=11655.ckpt')
data = myDataLoader()
kfold_data = data_kfold()
train_data, valid_data, test_data = kfold_data.kfold_split(data)
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


