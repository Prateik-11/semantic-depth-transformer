import torch
import torchvision
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from tqdm.autonotebook import tqdm
from PIL import Image
import os
import wandb
import random


#==============================================#
#                 HYPERPARAMETERS
#==============================================#
device = "cuda:2"
batch_size = 32
WANDB = True
epochs = 10
save_every = 1
model_ID = "openai/clip-vit-large-patch14"
#==============================================#


model = CLIPModel.from_pretrained(model_ID)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)
preprocess.do_center_crop = False # replaced with random crop in the dataloader
preprocess.do_resize = False # handled in the dataloader

# preprocess1 = CLIPImageProcessor.from_pretrained(model_ID)
## turn off pre-processing to diagnose issues with the images looking wrong
# preprocess1.do_center_crop = False
# preprocess1.do_normalize = False
# preprocess1.do_rescale = False
# preprocess1.do_resize = False
# preprocess1.do_rample = False
# preprocess1.do_convert_rgb = True
# preprocess2 = CLIPImageProcessor.from_pretrained(model_ID)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, depth_path, segmentation_path):
        depth_paths = []
        seg_paths = []

        for root, dirs, files in os.walk(depth_path):
            if files != []:    
                depth_paths += [f'{root}/{file}' for file in files]
                seg_paths += [f"{root.replace('depth', 'classSegmentation')}/{file.replace('depth', 'classgt')}" for file in files]
        
        ## since it searches alphabetically, these paths will not align with the paths of the other images by default
        # segmentation_paths = []
        # for root, dirs, files in os.walk(segmentation_path):
        #     if files != []:    
        #         segmentation_paths += [f'{root}/{file}' for file in files]
        
        self.depths = depth_paths
        self.segs = seg_paths
    
    def convert_to_rgb(self, y, show = False):
        y = torchvision.transforms.PILToTensor()(y)
        y2 = (y - torch.min(y))
        y3 = ((y2 / torch.max(y2)) * 255).type(torch.uint8)
        y4 = torch.cat([y3, y3, y3], dim=0)
        y5 = torchvision.transforms.ToPILImage()(y4)
        if show:
            display(y5)
        return y5
    
    # def load_and_preprocess_image_depth(self, image_path):
    #     # the PIL convert to rgb function does not work for the depth PNGs
    #     # they will turn into one flat color
    #     image = convert_to_rgb(Image.open(image_path))
    #     image = preprocess(image, return_tensors="pt")["pixel_values"]        
    #     return image
    
    # def load_and_preprocess_image_seg(self, image_path):
    #     image = Image.open(image_path)
    #     image = preprocess(image, return_tensors="pt")["pixel_values"]
    #     return image
    
    def load_and_preprocess(self, depth_path, seg_path):
        depth = self.convert_to_rgb(Image.open(depth_path))
        seg = Image.open(seg_path)

        # random cropping
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(depth, output_size=(224, 224))
        depth = torchvision.transforms.functional.crop(depth, i, j, h, w)
        seg = torchvision.transforms.functional.crop(seg, i, j, h, w)
        
        # random horizontal flipping
        if random.random() > 0.5:
            depth = torchvision.transforms.functional.hflip(depth)
            seg = torchvision.transforms.functional.hflip(seg)

        # Random vertical flipping
        if random.random() > 0.5:
            depth = torchvision.transforms.functional.vflip(depth)
            seg = torchvision.transforms.functional.vflip(seg)
        
        depth = preprocess(depth, return_tensors="pt")["pixel_values"]
        seg = preprocess(seg, return_tensors="pt")["pixel_values"]

        return depth, seg

    def __getitem__(self, i):
        return self.load_and_preprocess(self.depths[i], self.segs[i])

    def __len__(self):
        return len(self.depths)

dataset = Dataset(
    depth_path = '/localhome/prateiksinha/kitti/depth',
    segmentation_path = '/localhome/prateiksinha/kitti/classSegmentation'
)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    drop_last=True, 
    num_workers=8, 
    pin_memory=True
)


# sanity checking dataset:

# i = 0
# j = 446
# x, y = dataset[i]
# a, b = dataset[j]
# import torchvision.transforms as T
# transform = T.ToPILImage()
# # print(torch.all(x == a))
# # print(torch.all(y == b))
# display(transform(a[0,:,:,:]))
# display(transform(x[0,:,:,:]))
# display(transform(b[0,:,:,:]))
# display(transform(y[0,:,:,:]))


class ImageModel(torch.nn.Module):
    def __init__(self, model, projection, logit_scale):
        super().__init__()
        self.model = model
        self.projection = projection
        self.logit_scale = logit_scale
        pass

    def forward(self, x, y):
        x = self.projection(self.model(x)['pooler_output'])
        y = self.projection(self.model(y)['pooler_output'])

        # normalize features
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        y = y / y.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_y = torch.matmul(y, x.t()) * logit_scale
        logits_per_x = logits_per_y.t()

        return {
            'logits per x': logits_per_x,
            'logits per y': logits_per_y,
            'x_features' : x,
            'y_features' : y,
        }

image_model = ImageModel(model.vision_model, model.visual_projection, model.logit_scale)
image_model = image_model.to(device)

if WANDB:
    wandb.init(
        project="depth-seg-CLIP",
        config={
        "learning_rate": 5e-5,
        "architecture": "CLIP",
        "dataset": "Virtual Kitti",
        "epochs": 10,
        }
    )

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

optimizer = torch.optim.Adam(image_model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

for epoch in range(epochs):
    for depth, seg in iter(tqdm(dataloader)):
        depth, seg = depth.squeeze(dim=1).to(device), seg.squeeze(dim=1).to(device)
        logits_per_x = image_model(depth, seg)['logits per x']
        loss = clip_loss(logits_per_x)
        tqdm.write(str(loss.item()), end='\r')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if WANDB:
            wandb.log({
                'loss':loss
            })
    if epoch % save_every == 0:
        torch.save({
            'model' : image_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, '/localhome/prateiksinha/clip/checkpoints/image_model.pt')
if WANDB:
    wandb.finish()


