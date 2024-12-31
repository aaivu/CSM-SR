import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.applications import VGG19 # type: ignore
from torch.utils.data import DataLoader
from loss_functions.loss_functions import adversarial_loss, total_loss

@tf.function 
def pretrain_step(LR_batch, HR_batch, generator, generator_optimizer, mse_loss):
    """
    Performs a single pre-training step for the generator using MSE loss.

    Args:
    - LR_batch (Tensor): Low-resolution images batch.
    - HR_batch (Tensor): High-resolution images batch.
    - generator (Model): The generator model.
    - generator_optimizer (Optimizer): The optimizer for the generator.
    - mse_loss (Loss): Mean Squared Error loss function.

    Returns:
    - Tensor: The loss value for the current training step.
    """
    with tf.GradientTape() as tape:
        generated_images = generator(LR_batch, training=True)
        loss = mse_loss(HR_batch, generated_images)
    
    gradients = tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    
    return loss

@tf.function 
def distributed_pretrain_step(LR_batch, HR_batch, generator, generator_optimizer, mse_loss, strategy):
    """
    Performs a distributed pre-training step for the generator.

    Args:
    - LR_batch (Tensor): Low-resolution images batch.
    - HR_batch (Tensor): High-resolution images batch.
    - generator (Model): The generator model.
    - generator_optimizer (Optimizer): The optimizer for the generator.
    - mse_loss (Loss): Mean Squared Error loss function.
    - strategy (Strategy): The distributed training strategy.

    Returns:
    - Tensor: The loss value for the current training step.
    """
    per_replica_losses = strategy.run(pretrain_step, args=(LR_batch, HR_batch, generator, generator_optimizer, mse_loss))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function 
def train_step(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, **kwargs):    
    """
    Performs a single training step for both the generator and the discriminator.

    Args:
    - LR_batch (Tensor): Low-resolution images batch.
    - HR_batch (Tensor): High-resolution images batch.
    - generator (Model): The generator model.
    - discriminator (Model): The discriminator model.
    - vgg (Model): Pre-trained VGG19 model for perceptual loss.
    - generator_optimizer (Optimizer): The optimizer for the generator.
    - discriminator_optimizer (Optimizer): The optimizer for the discriminator.
    - lambda_adv (float): Weight for adversarial loss.
    - lambda_perceptual (float): Weight for perceptual loss.
    - lambda_grad (float): Weight for gradient loss.
    - lambda_second (float): Weight for second-order gradient loss.
    - lambda_struct (float): Weight for structure-aware loss.

    Returns:
    - Tuple[Tensor, Tensor, Tensor]: Generator loss, discriminator loss, and generated images.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(LR_batch, training=True)

        real_output = discriminator(HR_batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = total_loss(vgg, HR_batch, generated_images, real_output, fake_output, 
                              **kwargs)
        disc_loss = adversarial_loss(tf.ones_like(real_output, dtype=tf.float32), real_output) + \
                        adversarial_loss(tf.zeros_like(fake_output, dtype=tf.float32), fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # Gradient clipping for discriminator
    ## gradients_of_discriminator, _ = tf.clip_by_global_norm(gradients_of_discriminator, 10.0)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss, generated_images

@tf.function 
def distributed_train_step(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, strategy, **kwargs): 
    """
    Performs a distributed training step for both the generator and the discriminator.

    Args:
    - LR_batch (Tensor): Low-resolution images batch.
    - HR_batch (Tensor): High-resolution images batch.
    - generator (Model): The generator model.
    - discriminator (Model): The discriminator model.
    - vgg (Model): Pre-trained VGG19 model for perceptual loss.
    - generator_optimizer (Optimizer): The optimizer for the generator.
    - discriminator_optimizer (Optimizer): The optimizer for the discriminator.
    - strategy (Strategy): The distributed training strategy.
    - lambda_adv (float): Weight for adversarial loss.
    - lambda_perceptual (float): Weight for perceptual loss.
    - lambda_grad (float): Weight for gradient loss.
    - lambda_second (float): Weight for second-order gradient loss.
    - lambda_struct (float): Weight for structure-aware loss.

    Returns:
    - Tuple[Tensor, Tensor, Tensor]: Generator loss, discriminator loss, and generated images.
    """
    # per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images = strategy.run(
    #     train_step, 
    #     args=(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer, 
    #           lambda_adv, lambda_perceptual, lambda_grad, lambda_second, lambda_struct)
    # )
    per_replica_gen_losses, per_replica_disc_losses, per_replica_generated_images = strategy.run( train_step, args=(LR_batch, HR_batch, generator, discriminator, vgg, generator_optimizer, discriminator_optimizer), kwargs=kwargs) 
    gen_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_gen_losses, axis=None) 
    disc_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_disc_losses, axis=None) 
    return gen_loss, disc_loss, per_replica_generated_images

