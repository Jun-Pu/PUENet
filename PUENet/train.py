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

from models.puenet_res2net50 import BCVAEModule, PUAModule

# ablation studies
#from models_ab.puenet_hybrid_vit_wo_sam import BCVAEModule, PUAModule


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=70, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-6, help='learning rate')
parser.add_argument('--lr_dis', type=float, default=1e-6, help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')
parser.add_argument('--forward_iter', type=int, default=2, help='number of iterations of generator forward')
parser.add_argument('--lat_weight', type=float, default=1.0, help='weight for latent loss')
parser.add_argument('--vae_loss_weight', type=float, default=0.4, help='weight for vae loss')
parser.add_argument('--reg_weight', type=float, default=1e-4, help='weight for regularization term')
opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
print('Discriminator Learning Rate: {}'.format(opt.lr_dis))

# build models
generator = BCVAEModule(latent_dim=opt.latent_dim)
print_network(generator, 'generator')

# load pretrain
generator.load_state_dict(torch.load('./pretrain/GEN_epoch_checkpoint.pth'))

generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

discriminator = PUAModule(ndf=64)
print_network(discriminator, 'discriminator')

# load pretrain
discriminator.load_state_dict(torch.load('./pretrain/DIS_epoch_checkpoint.pth'))

discriminator.cuda()
discriminator_params = discriminator.parameters()
discriminator_optimizer = torch.optim.Adam(discriminator_params, opt.lr_dis)

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
logging.info("GenTransCOD-Train")
logging.info("Config")
logging.info('epoch:{}; lr_gen:{}; lr_dis:{}; batchsize:{}; trainsize:{}; save_path:{}'.
             format(opt.epoch, opt.lr_gen, opt.lr_dis, opt.batchsize, opt.trainsize, save_path))

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

# visualization
def visualize_prediction_init(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_init.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_dis_out(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_dis_output.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_dis_target(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_dis_target.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_prediction_ref(pred):
    for kk in range(pred.shape[0]):
        pred_edge_kk = pred[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_ref.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_gt(var_map):
    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_original_img(rec_img):
    img_transform = transforms.Compose([
        transforms.Normalize(mean = [-0.4850/.229, -0.456/0.224, -0.406/0.225], std =[1/0.229, 1/0.224, 1/0.225])])
    for kk in range(rec_img.shape[0]):
        current_img = rec_img[kk,:,:,:]
        current_img = img_transform(current_img)
        current_img = current_img.detach().cpu().numpy().squeeze()
        current_img = current_img * 255
        current_img = current_img.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_img.png'.format(kk)
        current_img = current_img.transpose((1,2,0))
        current_b = current_img[:, :, 0]
        current_b = np.expand_dims(current_b, 2)
        current_g = current_img[:, :, 1]
        current_g = np.expand_dims(current_g, 2)
        current_r = current_img[:, :, 2]
        current_r = np.expand_dims(current_r, 2)
        new_img = np.concatenate((current_r, current_g, current_b), axis=2)
        cv2.imwrite(save_path+name, new_img)


# ----------------------------------------------------------------------------------------------------------------------
best_mae = 1
best_epoch = 0

def TRAIN(train_loader, gen_model, dis_model, gen_optimizer, dis_optimizer, epoch, save_path):
    gen_model.train()
    dis_model.train()
    loss_record_gen = AvgMeter()
    loss_record_dis = AvgMeter()
    print('Generator Learning Rate: {}'.format(gen_optimizer.param_groups[0]['lr']))
    print('Discriminator Learning Rate: {}'.format(dis_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        # check input size
        images = F.upsample(images, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=True)
        gts = F.upsample(gts, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=True)

        # train generator
        p_m_prior, p_m_post, p_a, latent_loss = gen_model(images, gts)
        reg_loss = l2_regularisation(gen_model.enc_x) + l2_regularisation(gen_model.enc_xy) + \
                   l2_regularisation(gen_model.decoder_prior) + l2_regularisation(gen_model.decoder_post)
        reg_loss = opt.reg_weight * reg_loss
        anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
        latent_loss = opt.lat_weight * anneal_reg * latent_loss
        gen_loss_cvae = opt.vae_loss_weight * (structure_loss(p_m_post, gts) + latent_loss)
        gen_loss_gsnn = (1 - opt.vae_loss_weight) * structure_loss(p_m_prior, gts)
        seg_loss = gen_loss_cvae + gen_loss_gsnn + structure_loss(p_a, gts) + reg_loss
        seg_loss.backward()
        gen_optimizer.step()

        # get variance map
        preds = [torch.sigmoid(p_m_post)]
        with torch.no_grad():
            for ff in range(opt.forward_iter - 1):
                _, ff_m, _, _ = gen_model(images, gts)
                preds.append(torch.sigmoid(ff_m))

        preds = torch.cat(preds, dim=1)
        mean_preds = torch.mean(preds, 1, keepdim=True)
        var_map = -1 * mean_preds * torch.log(mean_preds + 1e-8)
        #var_map = torch.mean(var_map, 1, keepdim=True)
        var_map = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
        var_map = Variable(var_map.data, requires_grad=True)

        # train discriminator
        output_D = dis_model(torch.cat((images, torch.sigmoid(p_m_post.detach())), 1))
        output_D = F.upsample(output_D, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=True)
        consist_loss = CE_loss(output_D, var_map)
        consist_loss.backward()
        dis_optimizer.step()

        #visualize_prediction_init(torch.sigmoid(p_m))
        #visualize_dis_out(torch.sigmoid(output_D))
        #visualize_gt(gts)

        loss_record_gen.update(seg_loss.data, opt.batchsize)
        loss_record_dis.update(consist_loss.data, opt.batchsize)

        if i % 50 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Seg Loss: {:.4f}, Consist Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record_gen.show(), loss_record_dis.show()))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Seg Loss: {:.4f}, Consist Loss: {:.4f}'.
                         format(epoch, opt.epoch, i, total_step, loss_record_gen.show(), loss_record_dis.show()))

    if epoch % 5 == 0:
        torch.save(gen_model.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
        torch.save(dis_model.state_dict(), save_path + 'Model' + '_%d' % epoch + '_dis.pth')

def TEST(test_loader, gen_model, dis_model, epoch, save_path):
    global best_mae, best_epoch
    gen_model.eval()
    dis_model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, HH, WW = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res, _ = gen_model(image)
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
                torch.save(gen_model.state_dict(), save_path + 'GEN_epoch_best.pth')
                torch.save(dis_model.state_dict(), save_path + 'DIS_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Let's go!")
    for epoch in range(1, (opt.epoch+1)):
        adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
        adjust_lr(discriminator_optimizer, opt.lr_dis, epoch, opt.decay_rate, opt.decay_epoch)
        #print(generator_optimizer.param_groups[0]['lr'])
        #print(discriminator_optimizer.param_groups[0]['lr'])
        TRAIN(train_loader, generator, discriminator, generator_optimizer, discriminator_optimizer, epoch, save_path)
        TEST(test_loader, generator, discriminator, epoch, save_path)