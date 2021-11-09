import torch
import torch.nn.functional as F
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from data import test_dataset
from tqdm import tqdm
import cv2
import numpy as np

# select testing model
from models.puenet_hybrid_vit import BCVAEModule, PUAModule

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')
opt = parser.parse_args()

dataset_path = '/home/yzhang1/PythonProjects/COD_dataset/PARTIAL/TestDataset/Imgs/'

# -----------------------------------------

SegMod = BCVAEModule(latent_dim=opt.latent_dim)

# load pretrained model
SegMod.load_state_dict(torch.load('./model_test/BCVAE_hybrid_vit_final.pth'))

SegMod.cuda()
SegMod.eval()

# -----------------------------------------

PUAMod = PUAModule(ndf=64)

# load pretrained model
PUAMod.load_state_dict(torch.load('./model_test/PUA_final.pth'))

PUAMod.cuda()
PUAMod.eval()

test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']

save_root = './results/PUA_pred/'
save_root_2 = './results/PUA_heatmap/'

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

        # generate prediction
        pred, _ = SegMod.forward(image)
        res = PUAMod.forward(torch.cat((image, torch.sigmoid(pred)), 1))
        res = F.upsample(res, size=[WW, HH], mode='bilinear', align_corners=False)
        res = torch.sigmoid(res)

        # save prediction
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res)

        # save heatmap
        heat = res.astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_2 + name, heat)
    pass
