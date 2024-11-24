from torchvision import transforms
from skimage.color import rgb2gray
import numpy as np
import dnnlib
import torch
import copy
from tqdm import tqdm

def compute_SSIM(opts, all_images, all_brain_recons):
    batch_size = 64
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, **data_loader_kwargs):
        print(_labels)
    
    
    
    # Setup generator and load labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Image generation func.
    def run_generator(z, c):
        img = G(z=z, c=c, **opts.G_kwargs)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img


        z = torch.zeros([batch_size, G.z_dim], device=opts.device)
        c = torch.zeros([batch_size, G.c_dim], device=opts.device)
        run_generator = torch.jit.trace(run_generator, [z, c], check_trace=False)
    
  
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])
    # convert image to grayscale with rgb2grey
    img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0,2,3,1)).cpu())
    print("converted, now calculating ssim...")

    ssim_score=[]
    for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
        ssim_score.append(ssim(rec, im, multichannel=True, 
                               gaussian_weights=True, sigma=1.5, 
                               use_sample_covariance=False, data_range=1.0))

    ssim = np.mean(ssim_score)
    print(ssim)