import torch
import torch.nn.functional as F
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from data import test_dataset
from utils import print_network
from tqdm import tqdm
import cv2
import numpy as np
import time


# ----------------------------------------------------------------------------------------------------------------------
from models.puenet_hybrid_vit import BCVAEModule, PUAModule

# alternative models
#from models.puenet_res2net50_352_352 import BCVAEModule, PUAModule
#from models.puenet_resnet50 import BCVAEModule, PUAModule

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')
opt = parser.parse_args()

SegMod = BCVAEModule(latent_dim=opt.latent_dim)
print_network(SegMod, 'BCVAE')
PUAMod = PUAModule(ndf=64)
print_network(PUAMod, 'PUA')

# load pretrained models
SegMod.load_state_dict(torch.load('./pretrains/BCVAE_hybrid_vit_final.pth'))
PUAMod.load_state_dict(torch.load('./pretrains/PUA_final.pth'))

SegMod.cuda()
SegMod.eval()

PUAMod.cuda()
PUAMod.eval()

dataset_path = '/home/yzhang1/PythonProjects/DATASET/Obect_Seg/TestDataset/Imgs/'
test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']

save_root = './results/preds/'
save_root_2 = './results/uncertainties/'

TIME_BCVAE, TIME_PUA = [], []
for dataset in test_datasets:
    save_path = save_root + dataset + '/'
    save_path_2 = save_root_2 + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_2):
        os.makedirs(save_path_2)
    print(dataset)
    image_root = dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in tqdm(range(test_loader.size)):
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()

        # generate predictions & uncertainty maps
        time_bcvae_start = time.time()
        pred, _ = SegMod.forward(image)
        time_bcvae_end = time.time()
        TIME_BCVAE.append(time_bcvae_end - time_bcvae_start)

        pua_in = torch.cat((image, torch.sigmoid(pred)), 1)
        time_pua_start = time.time()
        sigma = PUAMod.forward(pua_in)
        time_pua_end = time.time()
        TIME_PUA.append(time_pua_end - time_pua_start)

        pred = F.upsample(pred, size=[WW, HH], mode='bilinear', align_corners=False)
        pred = torch.sigmoid(pred)
        sigma = F.upsample(sigma, size=[WW, HH], mode='bilinear', align_corners=False)
        sigma = torch.sigmoid(sigma)
        
        # save predictions
        pred = pred.data.cpu().numpy().squeeze()
        pred = 255 * (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        cv2.imwrite(save_path + name, pred)

        # save uncertainty maps
        sigma = sigma.data.cpu().numpy().squeeze()
        sigma = 255 * (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-8)
        heat = sigma.astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_2 + name, heat)
print('BCVAE: %f FPS' % (6473 / np.sum(TIME_BCVAE)))  # Sum of 'CAMO', 'CHAMELEON', 'COD10K', 'NC4K'
print('PUA: %f FPS' % (6473 / np.sum(TIME_PUA)))  # Sum of 'CAMO', 'CHAMELEON', 'COD10K', 'NC4K'
print('Test Done!')