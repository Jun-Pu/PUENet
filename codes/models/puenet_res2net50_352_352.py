import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from models.base_model import BaseModel
from models.Res2Net import res2net50_v1b_26w_4s
from torch.distributions import Normal, Independent, kl

size_hyperpara = 11


class InferenceModel_x(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(InferenceModel_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * size_hyperpara * size_hyperpara, latent_size)
        self.fc2 = nn.Linear(channels * 8 * size_hyperpara * size_hyperpara, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * size_hyperpara * size_hyperpara)
        # print(output.size())
        # output = self.tanh(output)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return mu, logvar, dist


class InferenceModel_xy(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(InferenceModel_xy, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2 * channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2 * channels, 4 * channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8 * channels, 8 * channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * size_hyperpara * size_hyperpara, latent_size)
        self.fc2 = nn.Linear(channels * 8 * size_hyperpara * size_hyperpara, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn5(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * size_hyperpara * size_hyperpara)
        # print(output.size())
        # output = self.tanh(output)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

        return mu, logvar, dist


class PUAModule(nn.Module):
    def __init__(self, ndf):
        super(PUAModule, self).__init__()
        self.conv1 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


class BCVAEModule(BaseModel):
    def __init__(self,
        latent_dim
    ):
        super(BCVAEModule, self).__init__()
        channel = 128

        self.encoderRes2_50 = res2net50_v1b_26w_4s(pretrained=True)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.down4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.conv_aux = nn.Conv2d(4 * channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.out_conv_aux = nn.Conv2d(channel, 1, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(0.3)

        self.RFB_L4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.RFB_L3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.RFB_L2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.RFB_L1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.enc_x = InferenceModel_x(3, int(channel / 8), latent_dim)
        self.enc_xy = InferenceModel_xy(4, int(channel / 8), latent_dim)

        self.decoder_prior = BCVAEModule_decoder(latent_dim)
        self.decoder_post = BCVAEModule_decoder(latent_dim)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)

        return kl_div

    def forward(self, x, y=None):
        raw_x = x
        EnFeat_4, EnFeat_3, EnFeat_2, EnFeat_1 = self.encoderRes2_50.forward(x)

        EnFeat_1, EnFeat_2, EnFeat_3, EnFeat_4 = self.dropout(EnFeat_1), self.dropout(EnFeat_2),\
                                                 self.dropout(EnFeat_3), self.dropout(EnFeat_4)

        D4, D3, D2, D1 = self.RFB_L4(EnFeat_4), self.RFB_L3(EnFeat_3), self.RFB_L2(EnFeat_2), self.RFB_L1(EnFeat_1)

        A_out = torch.cat((self.upsample8(D4), self.upsample4(D3), self.upsample2(D2), D1), 1)
        A_out = self.conv_aux(A_out)
        Guidance = self.out_conv_aux(A_out)
        A_out = self.upsample4(Guidance)

        Guidance_P = self.sigmoid(Guidance)
        Guidance_N = self.sigmoid(Guidance) * (-1) + 1

        if y == None:
            mu_prior, logvar_prior, _ = self.enc_x(raw_x)
            z_prior = self.reparametrize(mu_prior, logvar_prior)
            D_out_prior = self.decoder_prior(Guidance_P, Guidance_N, D4, D3, D2, D1, z_prior)

            return D_out_prior, A_out
        else:
            mu_prior, logvar_prior, dist_prior = self.enc_x(raw_x)
            z_prior = self.reparametrize(mu_prior, logvar_prior)
            mu_post, logvar_post, dist_post = self.enc_xy(torch.cat((raw_x, y), 1))
            z_post = self.reparametrize(mu_post, logvar_post)
            kld = torch.mean(self.kl_divergence(dist_post, dist_prior))

            D_out_prior = self.decoder_prior(Guidance_P, Guidance_N, D4, D3, D2, D1, z_prior)
            D_out_post = self.decoder_post(Guidance_P, Guidance_N, D4, D3, D2, D1, z_post)

            return D_out_prior, D_out_post, A_out, kld


class BCVAEModule_decoder(BaseModel):
    def __init__(self,
                 latent_dim
                 ):
        super(BCVAEModule_decoder, self).__init__()
        channel = 128

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.down4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.noise_conv = nn.Conv2d(channel + latent_dim, channel, kernel_size=1, padding=0)
        self.spatial_axes = [2, 3]

        self.CAtten4 = SAMLayer(channel)
        self.CAtten3 = SAMLayer(channel)
        self.CAtten2 = SAMLayer(channel)
        self.CAtten1 = SAMLayer(channel)

        self.path4 = FeatureFusionBlock(channel)
        self.path3 = FeatureFusionBlock(channel)
        self.path2 = FeatureFusionBlock(channel)
        self.path1 = FeatureFusionBlock(channel)

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)

    def forward(self, Guidance_P, Guidance_N, D4, D3, D2, D1, z):
        z_noise = torch.unsqueeze(z, 2)
        z_noise = self.tile(z_noise, 2, D4.shape[self.spatial_axes[0]])
        z_noise = torch.unsqueeze(z_noise, 3)
        z_noise = self.tile(z_noise, 3, D4.shape[self.spatial_axes[1]])

        D4 = torch.cat((D4, z_noise), 1)
        D4 = self.noise_conv(D4)

        D4 = self.CAtten4(D4, self.down8(Guidance_P), self.down8(Guidance_N))
        D3 = self.CAtten3(D3, self.down4(Guidance_P), self.down4(Guidance_N))
        D2 = self.CAtten2(D2, self.down2(Guidance_P), self.down2(Guidance_N))
        D1 = self.CAtten1(D1, Guidance_P, Guidance_N)

        D_out = self.path4(D4)
        D_out = self.path3(D_out, D3)
        D_out = self.path2(D_out, D2)
        D_out = self.path1(D_out, D1)
        D_out = self.out_conv(D_out)

        return D_out


class SAMLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SAMLayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool_p = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_n = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du_p = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.conv_du_n = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x, w_p, w_n):
        x_p = x * w_p
        y_p = self.avg_pool_p(x_p)
        y_p = self.conv_du_p(y_p)

        x_n = x * w_n
        y_n = self.avg_pool_n(x_n)
        y_n = self.conv_du_n(y_n)
        return x + x_p * y_p + x_n * y_n


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_multi(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel, kb):
        super(RFB_multi, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, kb), padding=(0, int((kb-1)/2))),
            BasicConv2d(out_channel, out_channel, kernel_size=(kb, 1), padding=(int((kb-1)/2), 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=kb, dilation=kb)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, kb+2), padding=(0, int((kb+1)/2))),
            BasicConv2d(out_channel, out_channel, kernel_size=(kb+2, 1), padding=(int((kb+1)/2), 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=kb+2, dilation=kb+2)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, kb+4), padding=(0, int((kb+3)/2))),
            BasicConv2d(out_channel, out_channel, kernel_size=(kb+4, 1), padding=(int((kb+3)/2), 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=kb+4, dilation=kb+4)
        )
        self.conv_cat = BasicConv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))

        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = F.interpolate( output, scale_factor=2, mode="bilinear", align_corners=True)

        return output


class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out
