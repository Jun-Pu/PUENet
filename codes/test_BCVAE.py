import torch
import torch.nn.functional as F
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from data import test_dataset
from tqdm import tqdm
import cv2
import numpy as np
from utils import enable_dropout

from models.puenet_hybrid_vit import BCVAEModule

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')
parser.add_argument('--forward_iter', type=int, default=5, help='number of iterations of generator forward')
opt = parser.parse_args()

dataset_path = '/home/yzhang1/PythonProjects/COD_dataset/PARTIAL/TestDataset/Imgs/'

generator = BCVAEModule(latent_dim=opt.latent_dim)

generator.load_state_dict(torch.load('./model_test/BCVAE_hybrid_vit_final.pth'))

generator.cuda()
generator.eval()
enable_dropout(generator)  # uncertainty estimation for MC-dropout

test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']

save_root = './results/pred/'
save_root_2 = './results/entrophy/'
save_root_3 = './results/heatmap/'

for dataset in test_datasets:
    save_path = save_root + dataset + '/'
    save_path_2 = save_root_2 + dataset + '/'
    save_path_3 = save_root_3 + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_2):
        os.makedirs(save_path_2)
    if not os.path.exists(save_path_3):
        os.makedirs(save_path_3)
    print(dataset)
    image_root = dataset_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in tqdm(range(test_loader.size)):
        preds = []
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()

        # generate prediction and entropy
        with torch.no_grad():
            for uu in range(opt.forward_iter):
                preds.append(torch.sigmoid(generator.forward(image)[0]))

        preds = torch.cat(preds, dim=1)
        mean_preds = torch.mean(preds, 1, keepdim=True)
        var_map = -1 * mean_preds * torch.log(mean_preds + 1e-8)

        res = F.upsample(mean_preds, size=[WW, HH], mode='bilinear', align_corners=False)
        var = F.upsample(var_map, size=[WW, HH], mode='bilinear', align_corners=False)

        # save prediction
        res = res.data.cpu().numpy().squeeze()
        res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res)

        # save entropy
        var = var.data.cpu().numpy().squeeze()
        var = 255 * (var - var.min()) / (var.max() - var.min() + 1e-8)
        cv2.imwrite(save_path_2 + name, var)

        # save heatmap
        heat = var.astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        cv2.imwrite(save_path_3 + name, heat)
    pass
