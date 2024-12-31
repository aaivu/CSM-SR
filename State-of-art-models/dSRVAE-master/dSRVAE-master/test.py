from __future__ import print_function
import argparse

import os
import torch
from modules import VAE_SR, VAE_denoise_vali
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from os.path import join
import time
from collections import OrderedDict
import math
from datasets import is_image_file
from image_utils import *
from PIL import Image, ImageOps
from os import listdir
import torch.utils.data as utils
from torch.autograd import Variable
import os
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=64, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=128, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=8, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input', type=str, default='/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/data/DIV2K/DIV2K_test_LR_bicubic/X4')
parser.add_argument('--model_type', type=str, default='VAE')
parser.add_argument('--output', default='/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/dSRVAE-master/dSRVAE-master/Result', help='Location to save SR results')
parser.add_argument('--model_denoiser', default='/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/dSRVAE-master/dSRVAE-master/models/VAE_epoch_500.pth', help='pretrained denoising model')
# parser.add_argument('--model_SR', default='models/VAE_SR.pth', help='pretrained SR model')
parser.add_argument('--model_SR', default='/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/dSRVAE-master/dSRVAE-master/models/GAN_generator_500.pth', help='pretrained SR model')

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Building model ', opt.model_type)


denoiser = VAE_denoise_vali(input_dim=3, dim=32, feat_size=8, z_dim=512, prior='standard')
model = VAE_SR(input_dim=3, dim=64, scale_factor=opt.upscale_factor)

denoiser = torch.nn.DataParallel(denoiser, device_ids=gpus_list)
model = torch.nn.DataParallel(model, device_ids=gpus_list)
if cuda:
    denoiser = denoiser.cuda(gpus_list[0])
    model = model.cuda(gpus_list[0])


print('===> Loading datasets')
"""
if os.path.exists(opt.model_denoiser):
    # denoiser.load_state_dict(torch.load(opt.model_denoiser, map_location=lambda storage, loc: storage)) 
    pretrained_dict = torch.load(opt.model_denoiser, map_location=lambda storage, loc: storage)
    model_dict = denoiser.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    denoiser.load_state_dict(model_dict)
    print('Pre-trained Denoiser model is loaded.')

if os.path.exists(opt.model_SR):
    model.load_state_dict(torch.load(opt.model_SR, map_location=lambda storage, loc: storage))
    print('Pre-trained SR model is loaded.')

"""

'''if os.path.exists(opt.model_denoiser):
    # Load the checkpoint
    pretrained_dict = torch.load(opt.model_denoiser, map_location=lambda storage, loc: storage)
    
    # Get the current model's state dictionary
    model_dict = denoiser.state_dict()
    
    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    
    # Overwrite entries in the existing state dictionary
    model_dict.update(pretrained_dict)
    
    # Load the new state dictionary
    denoiser.load_state_dict(model_dict)
    print('Pre-trained Denoiser model is loaded.')'''


def eval():

    denoiser.eval()
    model.eval()

    LR_image = [join(opt.input, x) for x in listdir(opt.input) if is_image_file(x)]
    SR_image = [join(opt.output, x) for x in listdir(opt.input) if is_image_file(x)]


    for i in range(LR_image.__len__()):
        t0 = time.time()

        LR = Image.open(LR_image[i]).convert('RGB')
        with torch.no_grad():
            prediction = chop_forward(LR)

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(i), (t1 - t0)))

        prediction = prediction * 255.0
        prediction = prediction.clamp(0, 255)

        Image.fromarray(np.uint8(prediction)).save(SR_image[i])



transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)


def chop_forward(img):


    img = transform(img).unsqueeze(0)

    testset = utils.TensorDataset(img)
    test_dataloader = utils.DataLoader(testset, num_workers=opt.threads,
                                       drop_last=False, batch_size=opt.testBatchSize, shuffle=False)

    for iteration, batch in enumerate(test_dataloader, 1):
        input = Variable(batch[0]).cuda(gpus_list[0])
        batch_size, channels, img_height, img_width = input.size()

        lowres_patches = patchify_tensor(input, patch_size=opt.patch_size, overlap=opt.stride)

        n_patches = lowres_patches.size(0)
        out_box = []
        with torch.no_grad():
            for p in range(n_patches):
                LR_input = lowres_patches[p:p + 1]
                std_z = torch.from_numpy(np.random.normal(0, 1, (input.shape[0], 512))).float()
                z = Variable(std_z, requires_grad=False).cuda(gpus_list[0])
                Denoise_LR = denoiser(LR_input, z)
                SR = model(Denoise_LR)
                out_box.append(SR)

            out_box = torch.cat(out_box, 0)
            SR = recompose_tensor(out_box, opt.upscale_factor * img_height, opt.upscale_factor * img_width,
                                              overlap=opt.upscale_factor * opt.stride)


            SR = SR.data[0].cpu().permute(1, 2, 0)

    return SR




##Eval Start!!!!
eval()
