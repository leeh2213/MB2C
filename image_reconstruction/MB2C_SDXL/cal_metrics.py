from torchvision import transforms
import numpy as np
from skimage.color import rgb2gray
from tqdm import tqdm



def PixCorr(all_images, all_brain_recons):
    preprocess = transforms.Compose([transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),])

    # Flatten images while keeping the batch dimension
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_brain_recons_flattened = preprocess(all_brain_recons).view(len(all_brain_recons), -1).cpu()

    print(all_images_flattened.shape)
    print(all_brain_recons_flattened.shape)

    corrsum = 0
    for i in tqdm(range(982)):
        corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
    corrmean = corrsum / 982
    pixcorr = corrmean
    print(pixcorr)
    
def SSIM(all_images, all_brain_recons):
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
