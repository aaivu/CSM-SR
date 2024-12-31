import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from torch.autograd import Variable
import torchvision
import numpy as np




class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()
        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue

class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue

# class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(num_filter),  # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(num_filter)  # Added BatchNorm
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        return self.act(out)


class discriminator(nn.Module):
    def __init__(self, num_channels, base_filter, image_size):
        super(discriminator, self).__init__()
        self.image_size = image_size

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1)
        self.conv_blocks = nn.Sequential(
            nn.MaxPool2d(4, 4, 0),
            nn.BatchNorm2d(base_filter),
            ConvBlock(base_filter, base_filter, 3, 1, 1),
            nn.MaxPool2d(4, 4, 0),
            nn.BatchNorm2d(base_filter),
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1),
            ConvBlock(base_filter * 2, base_filter * 2, 4, 2, 1),
            nn.BatchNorm2d(base_filter * 2),
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1),
            ConvBlock(base_filter * 4, base_filter * 4, 4, 2, 1),
            nn.BatchNorm2d(base_filter * 4),
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1),
            ConvBlock(base_filter * 8, base_filter * 8, 4, 2, 1),
            nn.BatchNorm2d(base_filter * 8),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 100),  # Adjust input dimension based on the printed shape
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        # print(out.shape)  # Add this line to check the shape
        out = out.view(out.size()[0], -1)
        out = self.classifier(out).view(-1)
        return out



'''class discriminator(nn.Module):
    def __init__(self, num_channels, base_filter, image_size):
        super(discriminator, self).__init__()
        self.image_size = image_size

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1)  # 512
        self.conv_blocks = nn.Sequential(
            nn.MaxPool2d(4, 4, 0),
            nn.BatchNorm2d(base_filter),
            ConvBlock(base_filter, base_filter, 3, 1, 1),  # 128
            nn.MaxPool2d(4, 4, 0),
            nn.BatchNorm2d(base_filter),
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1),  # 32
            ConvBlock(base_filter * 2, base_filter * 2, 4, 2, 1),  # 16
            nn.BatchNorm2d(base_filter * 2),
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1),
            ConvBlock(base_filter * 4, base_filter * 4, 4, 2, 1),  # 8
            nn.BatchNorm2d(base_filter * 4),
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1),
            ConvBlock(base_filter * 8, base_filter * 8, 4, 2, 1),  # 4
            nn.BatchNorm2d(base_filter * 8),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 100),
            nn.LeakyReLU(0.2, inplace=True),  # Changed to LeakyReLU
            nn.Linear(100, 1),
        )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = out.view(out.size()[0], -1)
        out = self.classifier(out).view(-1)
        return out
'''

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.deconv(x))

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super(ResnetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(in_channels)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))

class VAE_denoise(nn.Module):
    def __init__(self, input_dim, dim, feat_size, z_dim, prior):
        super(VAE_denoise, self).__init__()

        self.LR_feat = nn.Sequential(
            ConvBlock(input_dim, 2*dim, 3, 1, 1),
            ConvBlock(2*dim, 2*dim, 3, 1, 1),
            ConvBlock(2*dim, dim, 3, 1, 1),
        )

        self.decoder = nn.Sequential(
            ConvBlock(4 * dim, 4 * dim, 1, 1, 0),
            DeconvBlock(4 * dim, 4 * dim, 6, 4, 1),
            DeconvBlock(4 * dim, 2 * dim, 6, 4, 1),
            ConvBlock(2 * dim, dim, 3, 1, 1),
            nn.Conv2d(dim, dim, 3, 2, 1),  # Adjust dimensions
        )   

        self.SR_recon = nn.Sequential(
            ResnetBlock(dim, 3, 1, 1),
            ResnetBlock(dim, 3, 1, 1),
            ResnetBlock(dim, 3, 1, 1),
        )

        self.SR_mu = nn.Sequential(
            nn.Conv2d(dim, input_dim, 3, 1, 1),
        )
        self.prior = prior
        self.feat_size = feat_size
        self.dim = dim

        self.VAE_encoder = nn.Sequential(
            nn.Linear(2048, 4096),  # Adjusted input dimension
            nn.Sigmoid()
        )

        self.q_z_mu = nn.Linear(4096, z_dim)
        self.q_z_logvar = nn.Sequential(
            nn.Linear(4096, z_dim),
            nn.Hardtanh(min_val=-6., max_val=2.),
        )

        self.VAE_decoder = nn.Sequential(
            nn.Linear(z_dim, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 8192),
            nn.Sigmoid(),
            nn.Linear(8192, 4 * dim * feat_size * feat_size),  # Adjust this layer to match the dimensions
            nn.Sigmoid(),
        )

        for m in self.modules():
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('Linear') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def log_p_z(self, z, prior):
        if prior == 'standard':
            log_prior = log_Normal_standard(z, dim=1)
        else:
            raise Exception('Wrong name of the prior!')

        return log_prior

    def reparameterize(self, mu, logvar, flag=0):
        if flag == 0:
            std = logvar.mul(0.5).exp_()
            eps = torch.cuda.FloatTensor(std.size()).normal_()
            eps = Variable(eps)
            z = eps.mul(std).add_(mu)
        else:
            std = logvar.mul(0.5).exp_()
            eps = torch.from_numpy(np.random.normal(0, 0.05, size=(std.size(0), 1, std.size(2), std.size(3)))).float()
            eps = Variable(eps).cuda()
            eps = eps.repeat(1, 3, 1, 1)
            z = eps.mul(std).add_(mu)

        return z

    def encode(self, HR_feat):
        x = self.VAE_encoder(HR_feat.view(HR_feat.size(0), -1))
        z_q_mu = self.q_z_mu(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mu, z_q_logvar

    def decode(self, LR, z_q):
        up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        LR_feat = self.LR_feat(LR)
        dec_feat = self.VAE_decoder(z_q)
        dec_feat = dec_feat.view(dec_feat.size(0), 4 * self.dim, self.feat_size, self.feat_size)

        mu_feat = self.decoder(dec_feat)

        com_feat = LR_feat - mu_feat
        SR_feat = self.SR_recon(com_feat)
        Denoise_LR = LR - self.SR_mu(SR_feat)

        return Denoise_LR

    def forward(self, HR_feat, LR):
        z_q_mu, z_q_logvar = self.encode(HR_feat)
        z_q = self.reparameterize(z_q_mu, z_q_logvar, flag=0)
        log_p_z = self.log_p_z(z_q, self.prior)
        log_q_z = log_Normal_diag(z_q, z_q_mu, z_q_logvar, dim=1)
        KL = -(log_p_z - log_q_z)
        KL = torch.sum(KL)

        Denoise_LR = self.decode(LR, z_q)

        return Denoise_LR, KL

def log_Normal_standard(x, dim=None):
    log_norm = -0.5 * torch.sum(x ** 2, dim=dim)
    return log_norm

def log_Normal_diag(x, mu, log_var, dim=None):
    log_norm = -0.5 * torch.sum(log_var + (x - mu) ** 2 / torch.exp(log_var), dim=dim)
    return log_norm

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.deconv(x))

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(in_channels)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))

class VAE_denoise(nn.Module):
    def __init__(self, input_dim, dim, feat_size, z_dim, prior):
        super(VAE_denoise, self).__init__()

        self.LR_feat = nn.Sequential(
            ConvBlock(input_dim, 2*dim, 3, 1, 1),
            ConvBlock(2*dim, 2*dim, 3, 1, 1),
            ConvBlock(2*dim, dim, 3, 1, 1),
        )

        self.decoder = nn.Sequential(
            ConvBlock(4 * dim, 4 * dim, 1, 1, 0),
            DeconvBlock(4 * dim, 4 * dim, 6, 4, 1),
            DeconvBlock(4 * dim, 2 * dim, 6, 4, 1),
            ConvBlock(2 * dim, dim, 3, 1, 1),
            nn.Conv2d(dim, dim, 3, 2, 1),  # Adjust dimensions
        )   

        self.SR_recon = nn.Sequential(
            ResnetBlock(dim, 3, 1, 1),
            ResnetBlock(dim, 3, 1, 1),
            ResnetBlock(dim, 3, 1, 1),
        )

        self.SR_mu = nn.Sequential(
            nn.Conv2d(dim, input_dim, 3, 1, 1),
        )
        self.prior = prior
        self.feat_size = feat_size
        self.dim = dim

        self.VAE_encoder = nn.Sequential(
            nn.Linear(2048, 4096),  # Adjusted input dimension
            nn.Sigmoid()
        )

        self.q_z_mu = nn.Linear(4096, z_dim)
        self.q_z_logvar = nn.Sequential(
            nn.Linear(4096, z_dim),
            nn.Hardtanh(min_val=-6., max_val=2.),
        )

        self.VAE_decoder = nn.Sequential(
            nn.Linear(z_dim, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 8192),
            nn.Sigmoid(),
            nn.Linear(8192, 4 * dim * feat_size * feat_size),  # Adjust this layer to match the dimensions
            nn.Sigmoid(),
        )

        for m in self.modules():
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('Linear') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def log_p_z(self, z, prior):
        if prior == 'standard':
            log_prior = log_Normal_standard(z, dim=1)
        else:
            raise Exception('Wrong name of the prior!')

        return log_prior

    def reparameterize(self, mu, logvar, flag=0):
        if flag == 0:
            std = logvar.mul(0.5).exp_()
            eps = torch.cuda.FloatTensor(std.size()).normal_()
            eps = Variable(eps)
            z = eps.mul(std).add_(mu)
        else:
            std = logvar.mul(0.5).exp_()
            eps = torch.from_numpy(np.random.normal(0, 0.05, size=(std.size(0), 1, std.size(2), std.size(3)))).float()
            eps = Variable(eps).cuda()
            eps = eps.repeat(1, 3, 1, 1)
            z = eps.mul(std).add_(mu)

        return z

    def encode(self, HR_feat):
        x = self.VAE_encoder(HR_feat.view(HR_feat.size(0), -1))
        z_q_mu = self.q_z_mu(x)
        z_q_logvar = self.q_z_logvar(x)
        return z_q_mu, z_q_logvar

    def decode(self, LR, z_q):
        up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        LR_feat = self.LR_feat(LR)
        dec_feat = self.VAE_decoder(z_q)
        dec_feat = dec_feat.view(dec_feat.size(0), 4 * self.dim, self.feat_size, self.feat_size)

        mu_feat = self.decoder(dec_feat)

        com_feat = LR_feat - mu_feat
        SR_feat = self.SR_recon(com_feat)
        Denoise_LR = LR - self.SR_mu(SR_feat)

        return Denoise_LR

    def forward(self, HR_feat, LR):
        z_q_mu, z_q_logvar = self.encode(HR_feat)
        z_q = self.reparameterize(z_q_mu, z_q_logvar, flag=0)
        log_p_z = self.log_p_z(z_q, self.prior)
        log_q_z = log_Normal_diag(z_q, z_q_mu, z_q_logvar, dim=1)
        KL = -(log_p_z - log_q_z)
        KL = torch.sum(KL)

        Denoise_LR = self.decode(LR, z_q)

        return Denoise_LR, KL

def log_Normal_standard(x, dim=None):
    log_norm = -0.5 * torch.sum(x ** 2, dim=dim)
    return log_norm

def log_Normal_diag(x, mu, log_var, dim=None):
    log_norm = -0.5 * torch.sum(log_var + (x - mu) ** 2 / torch.exp(log_var), dim=dim)
    return log_norm


class VAE_denoise_vali(nn.Module):
    def __init__(self, input_dim, dim, feat_size, z_dim, prior):
        super(VAE_denoise_vali, self).__init__()

        self.LR_feat = nn.Sequential(
            ConvBlock(input_dim, 2*dim, 3, 1, 1),
            ConvBlock(2*dim, 2*dim, 3, 1, 1),
            ConvBlock(2*dim, dim, 3, 1, 1),
        )

        self.decoder = nn.Sequential(
            ConvBlock(4 * dim, 4 * dim, 1, 1, 0),
            DeconvBlock(4 * dim, 4 * dim, 6, 4, 1),
            DeconvBlock(4 * dim, 2 * dim, 6, 4, 1),
            ConvBlock(2 * dim, dim, 3, 1, 1),
            nn.Conv2d(dim, dim, 3, 2, 1),  # Adjust dimensions
        )   

        self.SR_recon = nn.Sequential(
            ResnetBlock(dim, 3, 1, 1),
            ResnetBlock(dim, 3, 1, 1),
            ResnetBlock(dim, 3, 1, 1),
        )

        self.SR_mu = nn.Sequential(
            nn.Conv2d(dim, input_dim, 3, 1, 1),
        )
        self.prior = prior
        self.feat_size = feat_size
        self.dim = dim

        self.VAE_encoder = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.Sigmoid()
        )

        self.q_z_mu = nn.Linear(4096, z_dim)
        self.q_z_logvar = nn.Sequential(
            nn.Linear(4096, z_dim),
            nn.Hardtanh(min_val=-6., max_val=2.),
        )

        self.VAE_decoder = nn.Sequential(
            nn.Linear(z_dim, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, 8192),
            nn.Sigmoid(),
            nn.Linear(8192, 4 * dim * feat_size * feat_size),  # Adjust this layer to match the dimensions
            nn.Sigmoid(),
        )

        for m in self.modules():
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('Linear') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def decode(self, LR, z_q):
        up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        LR_feat = self.LR_feat(LR)
        dec_feat = self.VAE_decoder(z_q)
        dec_feat = dec_feat.view(dec_feat.size(0), 4 * self.dim, self.feat_size, self.feat_size)  # Adjusted the view to match the dimensions

        mu_feat = self.decoder(dec_feat)

        com_feat = LR_feat - mu_feat
        SR_feat = self.SR_recon(com_feat)
        Denoise_LR = LR - self.SR_mu(SR_feat)

        return Denoise_LR

    def forward(self, LR, z_q):
        Denoise_LR = self.decode(LR, z_q)
        return Denoise_LR


class VAE_SR(nn.Module):
    def __init__(self, input_dim, dim, scale_factor):
        super(VAE_SR, self).__init__()
        self.up = torch.nn.Upsample(scale_factor=4, mode='bicubic')
        self.LR_feat = ConvBlock(input_dim, dim, 3, 1, 1)
        self.feat = nn.Sequential(
            ResnetBlock(dim, 3, 1, 1, bias=True),
            ResnetBlock(dim, 3, 1, 1, bias=True),
            ResnetBlock(dim, 3, 1, 1, bias=True),
            ResnetBlock(dim, 3, 1, 1, bias=True),
        )
        self.recon = nn.Sequential(
            nn.Conv2d(dim, input_dim, 3, 1, 1)
        )

        for m in self.modules():
            class_name = m.__class__.__name__
            if class_name.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif class_name.find('Linear') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, LR):
        LR_feat = self.LR_feat(self.up(LR))
        LR_feat = self.feat(LR_feat)
        SR = self.recon(LR_feat)
        return SR


class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 use_bn=False,
                 use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.device = device
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        self.features.to(self.device)
        for param in self.parameters():
            param.requires_grad = False

    def set_feature_layer(self, feature_layer):
        model = torchvision.models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        self.features.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output
