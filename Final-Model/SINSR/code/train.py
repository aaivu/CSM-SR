import os
import tensorflow as tf
from utils.config_loader import load_config, load_loss_weights
from utils.utilities import interpolate_weights, update_loss_weights, get_lr, clear_gpu_memory
from utils.model_setup import setup_model_and_optimizers
from utils.training_functions import train
from torch.utils.data import DataLoader
import importlib
from utils.data_loader import PairedImageDataset

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

model_config = load_config('/Final-Model/SINSR/code/config/model_config.json')
train_config = load_config('/Final-Model/SINSR/code/config/train.json')
loss_weights_config = load_loss_weights('/Final-Model/SINSR/code/config/loss_config.json')

architecture_module = importlib.import_module(f'architectures.{model_config["model_architecture"]}')
strategy = tf.distribute.MirroredStrategy()

train_hr_path = train_config['train_hr_path']
train_lr_path = train_config['train_lr_path']
valid_hr_path = train_config['valid_hr_path']
valid_lr_path = train_config['valid_lr_path']
target_size_hr = tuple(train_config['target_size_hr'])
target_size_lr = tuple(train_config['target_size_lr'])
batch_size = train_config['batch_size']

train_dataset = PairedImageDataset(train_lr_path, train_hr_path, target_size_lr, target_size_hr, train_config)
valid_dataset = PairedImageDataset(valid_lr_path, valid_hr_path, target_size_lr, target_size_hr, train_config)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

clear_gpu_memory()
train(train_config, model_config, loss_weights_config, architecture_module, strategy, train_dataloader, valid_dataloader)
