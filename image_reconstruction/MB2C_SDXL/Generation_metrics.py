from diffusion_prior import *
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.color import rgb2gray
import numpy as np
import torch

def fintune_diff_subs():
    subs = 10
    for sub in range(1, subs+1):
        emb_img_train = torch.load(f'image_reconstruction/MB2C_SDXL/features/clip/train/image/sub{sub}_train_img_features.pt')
        emb_img_train = emb_img_train.view(1654,10,1,768).view(-1,768)
        emb_eeg = torch.load(f'image_reconstruction/MB2C_SDXL/features/clip/train/eeg/sub{sub}_train_eeg_features.pt')
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        print(device)

        config = {
        "data_path": "Data/Things-EEG2/Preprocessed_data_250Hz",
        "project": "train_pos_img_text_rep",
        "entity": "sustech_rethinkingbci",
        "name": "lr=3e-4_img_pos_pro_eeg",
        "lr": 3e-4,
        "epochs": 50,
        "batch_size": 1024,
        "logger": True,
        "encoder_type":'MB2C',
        }

        dataset = EmbeddingDataset(
            c_embeddings=emb_eeg, h_embeddings=emb_img_train, 
        )

        dl = DataLoader(dataset, batch_size=1024, shuffle=True)

        diffusion_prior = DiffusionPriorUNet(cond_dim=768, dropout=0.1)
        # number of parameters
        print(sum(p.numel() for p in diffusion_prior.parameters() if p.requires_grad))
        pipe = Pipe(diffusion_prior,device=device)

        # load pretrained model
        model_name = 'diffusion_prior' # 'diffusion_prior_vice_pre_imagenet' or 'diffusion_prior_vice_pre'
        pipe.train(dl, num_epochs=150, learning_rate=1e-3) # to 0.142 
        print('training successfully')

        save_path = f'Generation/fintune_ckpts/{config["encoder_type"]}/sub{sub}/{model_name}.pt'
        directory = os.path.dirname(save_path)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        torch.save(pipe.diffusion_prior.state_dict(), save_path)
        print('save successfully')
      
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

if __name__ == "__main__":
    gpus = [5]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
    fintune_diff_subs()