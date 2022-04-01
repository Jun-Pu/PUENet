import torch
#torch.cuda.current_device()
import torch.nn.functional as F
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import os
from datetime import datetime

from data import get_loader, test_in_train
from utils import adjust_lr, AvgMeter, print_network, structure_loss
import torchvision.transforms as transforms

import cv2
import argparse
import logging

from models.puenet_hybrid_vit import BCVAEModule, PUAModule


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_bcvae', type=float, default=2.5e-5, help='learning rate')
parser.add_argument('--lr_pua', type=float, default=1e-5, help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--batchsize', type=int, default=7, help='training batch size')
parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')
parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')
parser.add_argument('--forward_iter', type=int, default=5, help='number of iterations of BCVAE forward')
parser.add_argument('--lat_weight', type=float, default=1.0, help='weight for latent loss')
parser.add_argument('--vae_loss_weight', type=float, default=0.4, help='weight for vae loss')
parser.add_argument('--reg_weight', type=float, default=1e-4, help='weight for regularization term')
opt = parser.parse_args()
print('BCVAE Learning Rate: {}'.format(opt.lr_bcvae))
print('PUA Learning Rate: {}'.format(opt.lr_pua))

# build models
BCVAE = BCVAEModule(latent_dim=opt.latent_dim)
print_network(BCVAE, 'BCVAE')
BCVAE.cuda()
BCVAE_params = BCVAE.parameters()
BCVAE_optimizer = torch.optim.Adam(BCVAE_params, opt.lr_bcvae)

PUA = PUAModule(ndf=64)
print_network(PUA, 'PUA')
PUA.cuda()
PUA_params = PUA.parameters()
PUA_optimizer = torch.optim.Adam(PUA_params, opt.lr_pua)

# set path
image_root = '/home/yzhang1/PythonProjects/COD_dataset/PARTIAL/TrainDataset/Imgs/'
gt_root = '/home/yzhang1/PythonProjects/COD_dataset/PARTIAL/TrainDataset/GT/'
image_root_te = '/home/yzhang1/PythonProjects/COD_dataset/PARTIAL/Test_in_train/Imgs/'
gt_root_te = '/home/yzhang1/PythonProjects/COD_dataset/PARTIAL/Test_in_train/GT/'

save_path = 'checkpoints/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_in_train(image_root_te, gt_root_te, testsize=opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("PUENet-Train")
logging.info("Config")
logging.info('epoch:{}; lr_bcvae:{}; lr_pua:{}; batchsize:{}; trainsize:{}; save_path:{}'.
             format(opt.epoch, opt.lr_bcvae, opt.lr_pua, opt.batchsize, opt.trainsize, save_path))

# loss function
CE_loss = torch.nn.BCEWithLogitsLoss()

# linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)

    return annealed

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)

    return l2_reg


# ----------------------------------------------------------------------------------------------------------------------
best_mae = 1
best_epoch = 0

def TRAIN(train_loader, BcvaeMod, PuaMod, bcvae_optimizer, pua_optimizer, epoch, save_path):
    BcvaeMod.train()
    PuaMod.train()
    loss_record_bcvae = AvgMeter()
    loss_record_pua = AvgMeter()
    print('BCVAE Learning Rate: {}'.format(bcvae_optimizer.param_groups[0]['lr']))
    print('PUA Learning Rate: {}'.format(pua_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        bcvae_optimizer.zero_grad()
        pua_optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        # check input size
        images = F.upsample(images, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=True)
        gts = F.upsample(gts, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=True)

        # train BCVAE
        p_m_prior, p_m_post, p_a, latent_loss = BcvaeMod(images, gts)
        reg_loss = l2_regularisation(BcvaeMod.enc_x) + l2_regularisation(BcvaeMod.enc_xy) + \
                   l2_regularisation(BcvaeMod.decoder_prior) + l2_regularisation(BcvaeMod.decoder_post)
        reg_loss = opt.reg_weight * reg_loss
        anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
        latent_loss = opt.lat_weight * anneal_reg * latent_loss
        loss_cvae_post = opt.vae_loss_weight * (structure_loss(p_m_post, gts) + latent_loss)
        loss_cvae_prior = (1 - opt.vae_loss_weight) * structure_loss(p_m_prior, gts)
        seg_loss = loss_cvae_post + loss_cvae_prior + structure_loss(p_a, gts) + reg_loss
        seg_loss.backward()
        bcvae_optimizer.step()

        # get variance map (entropy)
        preds = [torch.sigmoid(p_m_post)]
        with torch.no_grad():
            for ff in range(opt.forward_iter - 1):
                _, ff_m, _, _ = BcvaeMod(images, gts)
                preds.append(torch.sigmoid(ff_m))

        preds = torch.cat(preds, dim=1)
        mean_preds = torch.mean(preds, 1, keepdim=True)
        var_map = -1 * mean_preds * torch.log(mean_preds + 1e-8)
        var_map = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
        var_map = Variable(var_map.data, requires_grad=True)

        # train PUA
        output_D = PuaMod(torch.cat((images, torch.sigmoid(p_m_post.detach())), 1))
        output_D = F.upsample(output_D, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=True)
        approximation_loss = CE_loss(output_D, var_map)
        approximation_loss.backward()
        pua_optimizer.step()


        loss_record_bcvae.update(seg_loss.data, opt.batchsize)
        loss_record_pua.update(approximation_loss.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Seg Loss: {:.4f}, Consist Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record_bcvae.show(), loss_record_pua.show()))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Seg Loss: {:.4f}, Consist Loss: {:.4f}'.
                         format(epoch, opt.epoch, i, total_step, loss_record_bcvae.show(), loss_record_pua.show()))

    if epoch % 5 == 0:
        torch.save(BcvaeMod.state_dict(), save_path + 'Model' + '_%d' % epoch + '_bcvae.pth')
        torch.save(PuaMod.state_dict(), save_path + 'Model' + '_%d' % epoch + '_pua.pth')

def TEST(test_loader, BcvaeMod, PuaMod, epoch, save_path):
    global best_mae, best_epoch
    BcvaeMod.eval()
    PuaMod.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, HH, WW = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, _ = BcvaeMod(image)
            res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae = mae_sum / test_loader.size
        # writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(BcvaeMod.state_dict(), save_path + 'BCVAE_epoch_best.pth')
                torch.save(PuaMod.state_dict(), save_path + 'PUA_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Let's go!")
    for epoch in range(1, (opt.epoch+1)):
        adjust_lr(BCVAE_optimizer, opt.lr_bcvae, epoch, opt.decay_rate, opt.decay_epoch)
        adjust_lr(PUA_optimizer, opt.lr_pua, epoch, opt.decay_rate, opt.decay_epoch)
        TRAIN(train_loader, BCVAE, PUA, BCVAE_optimizer, PUA_optimizer, epoch, save_path)
        TEST(test_loader, BCVAE, PUA, epoch, save_path)
