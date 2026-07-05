from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import numpy as np
import torch
import random
from glob import glob
import itertools

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SequentialSampler, \
                                     RandomSampler
from torchvision import transforms
from PIL import Image
from scipy import io, misc
class DatasetPair(Dataset):
    def __init__(self, cover_dir, stego_dir, transform):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.cover_list = os.listdir(cover_dir)#[x.split('/')[-1] for x in glob(cover_dir + '/*')]
        #print(self.cover_list)
        self.transform = transform
        assert len(os.listdir(stego_dir)) == len(self.cover_list), "Cover和Stego文件数量不一致！"


    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        idx = int(idx)
        labels = np.array([0,1], dtype='int32')
        cover_path = os.path.join(self.cover_dir,
                                  self.cover_list[idx])
        #print(cover_path)
        cover = Image.open(cover_path)
        images = np.empty((2, cover.size[0], cover.size[1], 1),
                          dtype='uint8')
        images[0,:,:,0] = np.array(cover)

        #print(self.cover_list[idx])
        stego_path = os.path.join(self.stego_dir,
                                      self.cover_list[idx])
        #print(stego_path)
        stego = Image.open(stego_path)
        images[1,:,:,0] = np.array(stego)
        # print(cover_path)
        # print(stego_path)
        samples = {'images': images, 'labels': labels}
        if self.transform:
            samples = self.transform(samples)
        return samples
class ToTensor(object):
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']
        #images = images.transpose((0,3,1,2)).astype('float32')
        images = (images.transpose((0,3,1,2)).astype('float32')/ 255)
        return {'images': torch.from_numpy(images),
                'labels': torch.from_numpy(labels).long()}

class AugData():#需要在totensor之前使用
    def __call__(self, samples):
        images, labels = samples['images'], samples['labels']

        # Rotation
        rot = random.randint(0, 3)
        images = np.rot90(images, rot, axes=[1, 2]).copy()
        # Mirroring
        if random.random() < 0.5:
            images = np.flip(images, axis=2).copy()

        new_sample = {'images': images, 'labels': labels}

        return new_sample


'''



#"E:\CMF_image_set\\256×256\\bossbase\wow-0.5\\train\\0\\"
train_cover_path = "E:\复现\\7个深度隐写分析模型\\2017Ye-Net_quanxuexi\\1\dataset\\train\\cover\\"
train_stego_path = "E:\复现\\7个深度隐写分析模型\\2017Ye-Net_quanxuexi\\1\dataset\\train\\stego\\"

train_transform = transforms.Compose([ToTensor()])
train_data = DatasetPair(train_cover_path,train_stego_path,train_transform)
train_batch_size = 3
train_loader = torch.utils.data.DataLoader(train_data,batch_size=train_batch_size, shuffle=True)
print(len(train_loader))
for batch_idx, data in enumerate(train_loader):
    print(batch_idx)
    print('data[\'images\'] =',data['images'])
    print('data[\'images\'].size() = ',data['images'].size())
    print('data[\'labels\'] = ',data['labels'])
    print('data[\'labels\'].size()',data['labels'].size())
    a = data['labels'].size()
    print('a = ',a)
    if batch_idx == len(train_loader)-1:
        last_batch_size =len(os.listdir(train_cover_path)) - train_batch_size*(len(train_loader)-1)
        datas = data['images'].view(last_batch_size * 2, 1, 256, 256)
        labels = data['labels'].view(last_batch_size * 2)
        print('datas = ', datas)
        print('datas.size() = ', datas.size())
        print('labels', labels)
        print('labels.size() = ', labels.size())
    else:
        datas = data['images'].view(train_batch_size*2,1,256,256)
        labels = data['labels'].view(train_batch_size*2)
        print('datas = ',datas)
        print('datas.size() = ', datas.size())
        print('labels',labels)
        print('labels.size() = ',labels.size())


'''
