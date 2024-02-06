import torch
import torchvision
import os
from PIL import Image
from torchvision.transforms import v2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path1, path2, size = (224, 224)):
        internal_image_path_1 = os.path.join(r'Scene01\15-deg-left\frames', path1.split('_')[-1], 'Camera_0')
        internal_image_path_2 = os.path.join(r'Scene01\15-deg-left\frames', path2.split('_')[-1], 'Camera_0')
        self.path1 = os.path.join(path1, internal_image_path_1)
        self.path2 = os.path.join(path2, internal_image_path_2)
        
        file_map1 = {}
        for f in os.listdir(self.path1):
            try:
                index = int(f.split('_')[1].split('.')[0])
            except:
                print(f)
            file_map1[index] = (os.path.join(self.path1, f))
        self.file_map1 = file_map1

        file_map2 = {}
        for f in os.listdir(self.path2):
            try:
                index = int(f.split('_')[1].split('.')[0])
            except:
                print(f)
            file_map2[index] = (os.path.join(self.path2, f))
        self.file_map2 = file_map2

        self.convert_tensor = v2.Compose(
            [
                v2.Resize(size),
                v2.ToImage()
            ]
        )
        self.convert_image = torchvision.transforms.ToPILImage()

    def __getitem__(self, i):
        rgb = Image.open(self.file_map1[i])
        rgb = self.convert_tensor(rgb)        

        depth = Image.open(self.file_map2[i])
        depth = self.convert_tensor(depth)
        
        return rgb, depth

    def view_raw(self, i):
        # display raw image before any transformations are applied
        display(Image.open(self.file_map1[i]))
        display(Image.open(self.file_map2[i]))
    
    def view(self, i):
        # display the image after we have processed it for training
        rgb, depth = self[i]
        display(self.convert_image(rgb))
        display(self.convert_image(depth))

    def __len__(self):
        return len(os.listdir(self.path1))