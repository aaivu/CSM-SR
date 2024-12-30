import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.applications import VGG19 # type: ignore
from torch.utils.data import DataLoader

# Import necessary functions and models
from utils.data_loader import ImageDataset
from utils.train_util import pretrain_step, distributed_pretrain_step, train_step, distributed_train_step
from loss_functions.loss_functions import total_loss

import tensorflow as tf

# # Check GPU availability
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# tf.debugging.set_log_device_placement(True)

# Load configurations
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

model_config = load_config('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/model_config.json')
train_config = load_config('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/train.json')
loss_config = load_config('/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/loss_config.json')

# Dynamic import of the selected architecture
import importlib
architecture_module = importlib.import_module(f'architectures.{model_config["model_architecture"]}')

def train():

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        generator = architecture_module.generator()
        discriminator = architecture_module.discriminator()
        vgg = VGG19(include_top=False, weights='imagenet', input_shape=tuple(train_config['vgg_input_shape']))
        vgg.trainable = False

        generator_optimizer = Adam(learning_rate=train_config['learning_rates']['generator'])
        discriminator_optimizer = Adam(learning_rate=train_config['learning_rates']['discriminator'])

        os.makedirs(train_config['save_path'], exist_ok=True)

        for epoch in range(train_config['epochs']):
            step = 0
            for hr_images, lr_images in zip(train_hr_dataloader, train_lr_dataloader):
                hr_images = hr_images.numpy().copy().reshape(batch_size, 768, 1024, 3).astype('float32')
                lr_images = lr_images.numpy().copy().reshape(batch_size, 192, 256, 3).astype('float32')
                hr_images = tf.convert_to_tensor(hr_images)
                lr_images = tf.convert_to_tensor(lr_images)

                gen_loss, disc_loss, generated_images = distributed_train_step(
                    lr_images, hr_images, generator, discriminator, vgg,
                    generator_optimizer, discriminator_optimizer, strategy,
                    lambda_adv=loss_config['lambda_adv'],
                    lambda_perceptual=loss_config['lambda_perceptual'],
                    lambda_grad=loss_config['lambda_grad'],
                    lambda_second=loss_config['lambda_second'],
                    lambda_struct=loss_config['lambda_struct']
                )

                gathered_generated_images = strategy.gather(generated_images, axis=0)

                print(f"Epoch {epoch + 1}, Step {step}, Generator Loss: {gen_loss.numpy()}", end="")
                print(f", Discriminator Loss: {disc_loss.numpy()}")
                step += 1

            # Save generated images for inspection
            for i in range(gathered_generated_images.shape[0]):
                generated_image_np = (gathered_generated_images[i].numpy() * 255).astype(np.uint8)
                generated_image_bgr = cv2.cvtColor(generated_image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(train_config['save_path'], f'epoch_{epoch}_step_{step}_replica_{i}.jpg'), generated_image_bgr)

            # Validation
            val_loss = 0
            for hr_val_images, lr_val_images in zip(valid_hr_dataloader, valid_lr_dataloader):
                hr_val_images = hr_val_images.numpy().copy().reshape(batch_size, 768, 1024, 3).astype('float32')
                lr_val_images = lr_val_images.numpy().copy().reshape(batch_size, 192, 256, 3).astype('float32')
                hr_val_images = tf.convert_to_tensor(hr_val_images)
                lr_val_images = tf.convert_to_tensor(lr_val_images)
                generated_val_images = generator(lr_val_images, training=False)

                real_output = discriminator(hr_val_images, training=False)
                fake_output = discriminator(generated_val_images, training=False)

                val_loss += total_loss(vgg, hr_val_images, generated_val_images, real_output, fake_output, 
                                       lambda_adv=loss_config['lambda_adv'],
                                       lambda_perceptual=loss_config['lambda_perceptual'],
                                       lambda_grad=loss_config['lambda_grad'],
                                       lambda_second=loss_config['lambda_second'],
                                       lambda_struct=loss_config['lambda_struct'])

            print(f"Epoch {epoch + 1}/{train_config['epochs']}, Validation Loss: {val_loss / len(valid_hr_dataloader)}")

            if epoch % train_config['save_interval'] == 0: 
                generator_save_path = os.path.join(train_config['model_save_path'], f'generator_epoch_{epoch}.keras') 
                generator.save(generator_save_path)
        print("Training completed successfully!")

if __name__ == "__main__":
        # Load datasets
    train_hr_path = train_config['train_hr_path']
    train_lr_path = train_config['train_lr_path']
    valid_hr_path = train_config['valid_hr_path']
    valid_lr_path = train_config['valid_lr_path']
    target_size_hr = tuple(train_config['target_size_hr'])
    target_size_lr = tuple(train_config['target_size_lr'])
    batch_size = train_config['batch_size']

    train_hr_dataset = ImageDataset(train_hr_path, target_size_hr)
    train_lr_dataset = ImageDataset(train_lr_path, target_size_lr)
    valid_hr_dataset = ImageDataset(valid_hr_path, target_size_hr)
    valid_lr_dataset = ImageDataset(valid_lr_path, target_size_lr)

    train_hr_dataloader = DataLoader(train_hr_dataset, batch_size=batch_size, shuffle=True)
    train_lr_dataloader = DataLoader(train_lr_dataset, batch_size=batch_size, shuffle=True)
    valid_hr_dataloader = DataLoader(valid_hr_dataset, batch_size=batch_size, shuffle=False)
    valid_lr_dataloader = DataLoader(valid_lr_dataset, batch_size=batch_size, shuffle=False)

    train()


# import os
# import json
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.optimizers import Adam # type: ignore
# from tensorflow.keras.applications import VGG19 # type: ignore
# from torch.utils.data import DataLoader


# # Load configuration
# config_path = '/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SINSR/code/config/train.json'  # Update this path accordingly
# with open(config_path, 'r') as f:
#     config = json.load(f)

# # Training Loop
# def train():
#     strategy = tf.distribute.MirroredStrategy()

#     with strategy.scope():
#         generator = build_super_resolution_generator()
#         discriminator = build_hybrid_discriminator()
#         vgg = VGG19(include_top=False, weights='imagenet', input_shape=(1024, 768, 3))
#         vgg.trainable = False

#         generator_optimizer = Adam(learning_rate=config['generator_learning_rate'])
#         discriminator_optimizer = Adam(learning_rate=config['discriminator_learning_rate'])

#         os.makedirs(config['save_path'], exist_ok=True)

#         for epoch in range(config['epochs']):
#             step = 0
#             for hr_images, lr_images in zip(train_hr_dataloader, train_lr_dataloader):
#                 hr_images = hr_images.numpy().copy().reshape(config['batch_size'], 768, 1024, 3).astype('float32')
#                 lr_images = lr_images.numpy().copy().reshape(config['batch_size'], 192, 256, 3).astype('float32')
#                 hr_images = tf.convert_to_tensor(hr_images)
#                 lr_images = tf.convert_to_tensor(lr_images)
                
#                 gen_loss, disc_loss, generated_images = distributed_train_step(lr_images, hr_images, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, strategy)
#                 gathered_generated_images = strategy.gather(generated_images, axis=0)

#                 print(f"Epoch {epoch + 1}, Step {step}, Generator Loss: {gen_loss.numpy()}", end="")
#                 print(f", Discriminator Loss: {disc_loss.numpy()}")
#                 step += 1

#             # Save generated images for inspection
#             for i in range(gathered_generated_images.shape[0]):
#                 generated_image_np = (gathered_generated_images[i].numpy() * 255).astype(np.uint8)
#                 generated_image_bgr = cv2.cvtColor(generated_image_np, cv2.COLOR_RGB2BGR)
#                 cv2.imwrite(os.path.join(config['save_path'], f'epoch_{epoch}_step_{step}_replica_{i}.jpg'), generated_image_bgr)

#             # Validation
#             val_loss = 0
#             for hr_val_images, lr_val_images in zip(valid_hr_dataloader, valid_lr_dataloader):
#                 hr_val_images = hr_val_images.numpy().copy().reshape(config['batch_size'], 768, 1024, 3).astype('float32')
#                 lr_val_images = lr_val_images.numpy().copy().reshape(config['batch_size'], 192, 256, 3).astype('float32')
#                 hr_val_images = tf.convert_to_tensor(hr_val_images)
#                 lr_val_images = tf.convert_to_tensor(lr_val_images)
#                 generated_val_images = generator(lr_val_images, training=False)

#                 real_output = discriminator(hr_val_images, training=False)
#                 fake_output = discriminator(generated_val_images, training=False)

#                 val_loss += total_loss(vgg, hr_val_images, generated_val_images, real_output, fake_output, 
#                                        lambda_adv=1, lambda_perceptual=0.5, lambda_grad=0.1, 
#                                        lambda_second=0.1, lambda_struct=0.1)

#             print(f"Epoch {epoch + 1}/{config['epochs']}, Validation Loss: {val_loss / len(valid_hr_dataloader)}")

#             if epoch % config['generator_save_interval'] == 0:
#                 generator.save(f'/home/oshadi/SISR-Final_Year_Project/envs/SISR-Project/Final-Model/SInSR/models2/generator_epoch_{epoch}.keras')

#         print("Training completed successfully!")

# if __name__ == "__main__":
#     train()
