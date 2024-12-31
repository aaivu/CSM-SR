import os
os.environ['TL_BACKEND'] = 'tensorflow' # Just modify this line, easily switch to any framework! PyTorch will coming soon!
# os.environ['TL_BACKEND'] = 'mindspore'
# os.environ['TL_BACKEND'] = 'paddle'
# os.environ['TL_BACKEND'] = 'torch'
import time
import numpy as np
import sys
import tensorflow as tf
import torch

import tensorlayerx as tlx
from tensorlayerx.dataflow import Dataset, DataLoader
from srgan import SRGAN_g, SRGAN_d
from config import config
from tensorlayerx.vision.transforms import Compose, RandomCrop, Normalize, RandomFlipHorizontal, Resize, HWC2CHW
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, Resize, ToPILImage, ToTensor
import vgg
from tensorlayerx.model import TrainOneStep
from tensorlayerx.nn import Module
import cv2
tlx.set_device('GPU')

from image_quality_metrics import calculate_psnr, calculate_ssim, calculate_lpips
import matplotlib.pyplot as plt


from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, Resize##
## from tensorflow.keras.mixed_precision import experimental as mixed_precision      ##--Version incompatible

### Set up mixed precision policy
## policy = mixed_precision.Policy('mixed_float16')
## mixed_precision.set_policy(policy)


###====================== HYPER-PARAMETERS ===========================###
batch_size = 16 # 16
n_epoch_init = config.TRAIN.n_epoch_init
n_epoch = config.TRAIN.n_epoch
# create folders to save result images and trained models
save_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/SRGAN-master/samples"
tlx.files.exists_or_mkdir(save_dir)
checkpoint_dir = "/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/State-of-art-models/SRGAN-master/models"
tlx.files.exists_or_mkdir(checkpoint_dir)
import cv2
import os
import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Normalize, Resize, ToPILImage, ToTensor

# Set up mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set up MirroredStrategy
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Define the transformations
hr_transform = Compose([
    ToPILImage(),
    RandomCrop(size=(384, 384)),  # Adjusted crop size
    RandomHorizontalFlip(),
    ToTensor()
])

nor = Compose([
    Normalize(mean=(127.5,), std=(127.5,)),
    ToTensor()
])

lr_transform = Compose([
    ToPILImage(),
    Resize(size=(96, 96)),  # Adjusted resize size
    ToTensor()
])

# Load high-resolution images
train_hr_imgs = tlx.vision.load_images(path=config.TRAIN.hr_img_path, n_threads=32)

# Define the dataset class
class TrainData(Dataset):

    def __init__(self, hr_trans=hr_transform, lr_trans=lr_transform):
        self.train_hr_imgs = train_hr_imgs
        self.hr_trans = hr_trans
        self.lr_trans = lr_trans

    def __getitem__(self, index):
        img = self.train_hr_imgs[index]
        
        # Ensure the image is in the correct format
        if isinstance(img, str):
            img = cv2.imread(img)
        if img is None:
            raise ValueError(f"Image at index {index} is None. Check the image path or format.")
        
        # Convert image to numpy array if it's not already
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # Ensure the image array is of type uint8
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        hr_patch = self.hr_trans(img)
        lr_patch = self.lr_trans(hr_patch)
        return nor(lr_patch).permute(2, 0, 1), nor(hr_patch).permute(2, 0, 1)

    def __len__(self):
        return len(self.train_hr_imgs)

# Define the loss classes
class WithLoss_init(Module):
    def __init__(self, G_net, loss_fn):
        super(WithLoss_init, self).__init__()
        self.net = G_net
        self.loss_fn = loss_fn

    def forward(self, lr, hr):
        out = self.net(lr)
        loss = self.loss_fn(out, hr)
        return loss

class WithLoss_D(Module):
    def __init__(self, D_net, G_net, loss_fn):
        super(WithLoss_D, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.loss_fn = loss_fn

    def forward(self, lr, hr):
        fake_patchs = self.G_net(lr)
        logits_fake = self.D_net(fake_patchs)
        logits_real = self.D_net(hr)
        d_loss1 = self.loss_fn(logits_real, tlx.ones_like(logits_real))
        d_loss1 = tlx.ops.reduce_mean(d_loss1)
        d_loss2 = self.loss_fn(logits_fake, tlx.zeros_like(logits_fake))
        d_loss2 = tlx.ops.reduce_mean(d_loss2)
        d_loss = d_loss1 + d_loss2
        return d_loss

class WithLoss_G(Module):
    def __init__(self, D_net, G_net, vgg, loss_fn1, loss_fn2):
        super(WithLoss_G, self).__init__()
        self.D_net = D_net
        self.G_net = G_net
        self.vgg = vgg
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2

    def forward(self, lr, hr):
        fake_patchs = self.G_net(lr)
        logits_fake = self.D_net(fake_patchs)
        feature_fake = self.vgg((fake_patchs + 1) / 2.)
        feature_real = self.vgg((hr + 1) / 2.)
        g_gan_loss = 1e-3 * self.loss_fn1(logits_fake, tlx.ones_like(logits_fake))
        g_gan_loss = tlx.ops.reduce_mean(g_gan_loss)
        mse_loss = self.loss_fn2(fake_patchs, hr)
        vgg_loss = 2e-6 * self.loss_fn2(feature_fake, feature_real)
        g_loss = mse_loss + vgg_loss + g_gan_loss
        return g_loss

# Define the models
with strategy.scope():
    G = SRGAN_g()
    D = SRGAN_d()
    VGG = vgg.VGG19(pretrained=True, end_with='pool4', mode='dynamic')
    G.init_build(tlx.nn.Input(shape=(8, 3, 96, 96)))  # Define above should be matched with this 
    D.init_build(tlx.nn.Input(shape=(8, 3, 384, 384)))

    # Define the optimizers and loss functions
    lr_v = tlx.optimizers.lr.StepDecay(learning_rate=0.05, step_size=1000, gamma=0.1, last_epoch=-1, verbose=True)
    g_optimizer_init = tlx.optimizers.Momentum(lr_v, 0.9)
    g_optimizer = tlx.optimizers.Momentum(lr_v, 0.9)
    d_optimizer = tlx.optimizers.Momentum(lr_v, 0.9)
    g_weights = G.trainable_weights
    d_weights = D.trainable_weights
    net_with_loss_init = WithLoss_init(G, loss_fn=tlx.losses.mean_squared_error)
    net_with_loss_D = WithLoss_D(D_net=D, G_net=G, loss_fn=tlx.losses.sigmoid_cross_entropy)
    net_with_loss_G = WithLoss_G(D_net=D, G_net=G, vgg=VGG, loss_fn1=tlx.losses.sigmoid_cross_entropy,
                                 loss_fn2=tlx.losses.mean_squared_error)

    trainforinit = TrainOneStep(net_with_loss_init, optimizer=g_optimizer_init, train_weights=g_weights)
    trainforG = TrainOneStep(net_with_loss_G, optimizer=g_optimizer, train_weights=g_weights)
    trainforD = TrainOneStep(net_with_loss_D, optimizer=d_optimizer, train_weights=d_weights)

def train():
    with strategy.scope():
        G.set_train()
        D.set_train()
        VGG.set_eval()
        train_ds = TrainData()
        train_ds_img_nums = len(train_ds)
        train_ds = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        # Initialize learning (G)
        n_step_epoch = round(train_ds_img_nums // batch_size)
        for epoch in range(n_epoch_init):
            for step, (lr_patch, hr_patch) in enumerate(train_ds):
                step_time = time.time()
                loss = trainforinit(lr_patch, hr_patch)
                print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                    epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, float(loss)))

        # Adversarial learning (G, D)
        n_step_epoch = round(train_ds_img_nums // batch_size)
        for epoch in range(n_epoch):
            for step, (lr_patch, hr_patch) in enumerate(train_ds):
                step_time = time.time()
                loss_g = trainforG(lr_patch, hr_patch)
                loss_d = trainforD(lr_patch, hr_patch)
                print(
                    "Epoch: [{}/{}] step: [{}/{}] time: {:.4f}s, g_loss:{:.4f}, d_loss: {:.4f}".format(
                        epoch, n_epoch, step, n_step_epoch, time.time() - step_time, float(loss_g), float(loss_d)))
            # Dynamic learning rate update
            lr_v.step()

            if (epoch != 0) and (epoch % 10 == 0):
                G.save_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
                D.save_weights(os.path.join(checkpoint_dir, 'd.npz'), format='npz_dict')


def evaluate():
    ###====================== PRE-LOAD DATA ===========================###
    valid_hr_imgs = tlx.vision.load_images(path=config.VALID.hr_img_path )
    ###========================LOAD WEIGHTS ============================###
    G.load_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
    G.set_eval()
    imid = 0  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_hr_img = valid_hr_imgs[imid]
    valid_lr_img = np.asarray(valid_hr_img)
    hr_size1 = [valid_lr_img.shape[0], valid_lr_img.shape[1]]
    valid_lr_img = cv2.resize(valid_lr_img, dsize=(hr_size1[1] // 4, hr_size1[0] // 4))
    valid_lr_img_tensor = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]


    valid_lr_img_tensor = np.asarray(valid_lr_img_tensor, dtype=np.float32)
    valid_lr_img_tensor = np.transpose(valid_lr_img_tensor,axes=[2, 0, 1])
    valid_lr_img_tensor = valid_lr_img_tensor[np.newaxis, :, :, :]
    valid_lr_img_tensor= tlx.ops.convert_to_tensor(valid_lr_img_tensor)
    size = [valid_lr_img.shape[0], valid_lr_img.shape[1]]

    out = tlx.ops.convert_to_numpy(G(valid_lr_img_tensor))
    out = np.asarray((out + 1) * 127.5, dtype=np.uint8)
    out = np.transpose(out[0], axes=[1, 2, 0])
    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")

    tlx.vision.save_image(out, file_name='valid_gen.jpg', path=save_dir)
    tlx.vision.save_image(valid_lr_img, file_name='valid_lr.jpg', path=save_dir)
    tlx.vision.save_image(valid_hr_img, file_name='valid_hr.jpg', path=save_dir)
    out_bicu = cv2.resize(valid_lr_img, dsize = [size[1] * 4, size[0] * 4], interpolation = cv2.INTER_CUBIC)
    tlx.vision.save_image(out_bicu, file_name='valid_hr_cubic.jpg', path=save_dir)
    
    # Calculate PSNR
    psnr_value = calculate_psnr(valid_hr_img, out)
    ssim_value = calculate_ssim(valid_hr_img, out)
    # lpips_value = calculate_lpips(valid_hr_img, out)

    print(f'PSNR value: {psnr_value} dB')
    print(f'SSIM value: {ssim_value}')
    # print(f'LPIPS value: {lpips_value}')
        
    # Display images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Low-Resolution Image')
    plt.imshow(cv2.cvtColor(valid_lr_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Generated High-Resolution Image')
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Original High-Resolution Image')
    plt.imshow(cv2.cvtColor(valid_hr_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, eval')

    args = parser.parse_args()

    tlx.global_flag['mode'] = args.mode

    if tlx.global_flag['mode'] == 'train':
        train()
    elif tlx.global_flag['mode'] == 'eval':
        evaluate()
    else:
        raise Exception("Unknow --mode")
