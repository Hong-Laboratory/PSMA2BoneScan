import tensorflow as tf
import numpy as np
import os
from glob import glob
import loss
import models
import matplotlib.pyplot as plt
import pandas as pd
import data_loader
import argparse
from tqdm import tqdm
import utils
import json
import tensorflow_probability as tfp


def get_args_parser():
    parser = argparse.ArgumentParser('PSMA2BS', add_help=False)
    
    parser.add_argument('--some_notes_about_experim', default='''mask bladder and kidney in PET, reduce lr ''')

    # Set Paths
    parser.add_argument('--data_dir', default="/path/to/trainset")
    parser.add_argument('--val_dir', default="/path/to/valset")
    parser.add_argument('--sample_image_dir', default="/path/to/trainset/sample", help="""A sample patient dir from the training set to visualize training progress""")

    # Checkpoint related parameters
    parser.add_argument('--output_path', default='/path/to/results/dir', type=str, help="""Path to save checkpoints""")
    parser.add_argument('--save_every', default=100, type=int, help="""Save checkpoint after every n epochs""")
    parser.add_argument('--max_to_keep', default=5, type=int, help="Max number of checkpoints to keep be CheckpointManager")
    
    # Model parameters
    parser.add_argument('--gan_mode', default='vanilla', choices=['vanilla', 'lsgan'], help="""What kind of adverserial loss function?""")
    parser.add_argument('--gen_arch', default='pix2pix', choices=['pix2pix', 'unet'], help="""Architiectire for Generators. Paper uses ResNet.""")
    parser.add_argument('--disc_arch', default='pixel', choices=['patch', 'pixel', 'MedGAN'], help="""Architiectire for Discriminators. Paper uses PatchGAN.""")
    parser.add_argument('--num_downsample_layers', default=2, type=int, help="""Number of downsampling layers for ResNet generator. Number of upsampling layers will automatically match.""")
    parser.add_argument('--num_downs', default=7, type=int, help="""Number of downsampling in Unet. For 256x256 input, 8""")
    parser.add_argument('--num_filters', default=64, type=int, help="""Number of filters in initial CNN layer of Generator & Discriminator""")
    parser.add_argument('--norm_type', default='instance', choices=['batch', 'instance', None], help="""Noramlization layer type in model""")
    parser.add_argument('--use_dropout', default=True)
    parser.add_argument('--patch_size', default=(256, 256, 1))

    # Training parameters
    parser.add_argument('--batch_size', default=1, type=int, help="""Batch size in training. Original paper uses 1 or 2""" )
    parser.add_argument('--epochs', default=501, type=int, help="""Total training epochs. Paper recommends 200 or more""")
    parser.add_argument('--epoch_decay', default=300, type=int, help="""Start linear decay of learning rate at this epoch""")
    parser.add_argument('--buffer_size', default=40, type=int, help="""Size of buffer used in shuffling training data""")
    parser.add_argument('--gen_initial_lr', default=5e-05, type=float, help="""Initial learning rate of optimizer""")
    parser.add_argument('--disc_initial_lr', default=5e-05, type=float, help="""Initial learning rate of optimizer""")
    parser.add_argument('--beta', default=0.5, type=float )
    parser.add_argument('--random_seed', default=42, help="""Sets the global random seed""")
    
    # Loss parameters
    parser.add_argument('--hist_loss_alpha', default=[1, 1, 1, 1, 0], help="""Absolute loss weight in histogram loss""")
    parser.add_argument('--hist_loss_beta', default=[0, 1, 1, 1, 1], help="""Relative loss weight in histogram loss""")
    parser.add_argument('--perceptual_weight', default=10, type=int, help="""Perceptial loss weight""")
    parser.add_argument('--style_weight', default=0.01, type=float, help="""Style loss weight""")
    parser.add_argument('--content_weight', default=0.01, type=float, help="""Content loss weight""")

    return parser
  

@tf.function
def train_step(real_x, real_y, d_fake_cond):
    with tf.GradientTape(persistent=True) as tape:
        # Generate fake image
        fake_y = gen(real_x) 
        
        ### GET FEATURES FROM REAL IMAGE ###
        # Feed real image to Discrimiator
        d_real_outputs = disc(real_y) # Run it through the Discriminator and get the features
        d_real_intermediates = d_real_outputs['intermediate']
        d_real_final = d_real_outputs['final']

        # Get the style, content features from the Feature Extractor
        real_feature_outputs = feature_extractor(tf.repeat(real_y, [3, 0, 0], axis=-1)) # Reshape input channel dim to 3 for VGG19
        real_style = real_feature_outputs['style']  
        real_content = real_feature_outputs['content']


        ### GET FEATURES FROM FAKE IMAGE ###
        # Input fake image to Discriminator
        d_fake_outputs = disc(tf.concat([fake_y, d_fake_cond], axis=-1))
        d_fake_intermediates = d_fake_outputs['intermediate']
        d_fake_final = d_fake_outputs['final']
        
        fake_feature_outputs = feature_extractor(tf.repeat(fake_y, 3, axis=-1))
        fake_style = fake_feature_outputs['style']  
        fake_content = fake_feature_outputs['content']
    

        ### COMPUTE LOSSES ###
        logger = {}
        # Adversarial loss 
        logger["gen_loss"] = loss.generator_loss(loss_obj, d_fake_final)
        logger["disc_loss"] = loss.discriminator_loss(loss_obj, d_real_final, d_fake_final)
        # Intensity histogram loss
        logger["hist_loss"] = loss.histogram_loss(real_y, fake_y, num_bins=5, alpha=args.hist_loss_alpha, beta=args.hist_loss_beta) # Write the code
        # Feature losses
        logger["perceptual_loss"] = tf.reduce_sum([loss.perceptualLoss(real_intermediate, fake_intermediate) for real_intermediate, fake_intermediate in zip(d_real_intermediates, d_fake_intermediates)]) * args.perceptual_weight
        logger["style_loss"] = tf.reduce_sum([loss.styleLoss(fake_style[k], real_style[k]) for k in real_style.keys()]) * args.style_weight
        logger["content_loss"] = tf.reduce_sum([loss.styleLoss(fake_content[k], real_content[k]) for k in real_content.keys()]) * args.content_weight
        # Total Generator loss
        logger["total_gen_loss"] = logger["gen_loss"] + logger["hist_loss"] + logger["perceptual_loss"] + logger["content_loss"] + logger["content_loss"]

    # Backprop
    g_gradients = tape.gradient(logger["total_gen_loss"], gen.trainable_variables)
    d_gradients = tape.gradient(logger["disc_loss"], disc.trainable_variables)
    generator_optimizer.apply_gradients(zip(g_gradients, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(d_gradients, disc.trainable_variables))
    
    return logger


@tf.function
def val_step(real_x, real_y):
    fake_y = gen(real_x, training=False)

    # Compute validation metrics
    logger = {}
    logger['hist_loss'] = tf.reduce_mean(loss.histogram_loss(real_y, fake_y, num_bins=5, alpha=args.hist_loss_alpha, beta=args.hist_loss_beta)) # Write the code
    logger["psnr"] = tf.reduce_mean(tf.image.psnr(real_y, fake_y, max_val=1.0))
    logger["ssim"] = tf.reduce_mean(tf.image.ssim(real_y, fake_y, max_val=1.0))
    
  return fake_y, logger




if __name__ == "__main__":
    
    # Configuring GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 1:
        print("This script uses only 1 GPU")

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser('PSMA2BS', parents=[get_args_parser()])
    args = parser.parse_args()

    # Save command line arguments
    args_dict = vars(args)
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, 'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
    
    # Metrics logger setup
    metric_logger = utils.MetricLoggerCallback()

    # Set seed
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Loss functions and configurations
    if args.gan_mode == 'vanilla':
        loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    elif args.gan_mode == 'lsgan':
        loss_obj = tf.keras.losses.MeanSquaredError()

    # Define Dataset
    train_dataset = data_loader.create_dataset(args.data_dir, batch_size=args.batch_size, buffer_size=args.buffer_size, train=True, input_size=(256,256, 256))
    val_dataset = data_loader.create_dataset(args.val_dir, batch_size=args.batch_size, train=False, input_size=(256, 256, 256))
    n_train = len(glob(f"{args.data_dir}/*"))
    n_val = len(glob(f"{args.val_dir}/*"))
    
    # Build model
    gen = models.build_generator(arch=args.gen_arch, num_blocks=args.num_downs, output_channels=1, base_filters=args.num_filters, use_dropout=args.use_dropout)
    disc = models.build_discriminator(arch=args.disc_arch, base_filters=args.num_filters, norm_type=args.norm_type, return_intermediate=True)
    feature_extractor = models.build_extractor(arch='vgg19', input_size=(args.patch_size[0], args.patch_size[1], 3))
    feature_extractor.trainable = False
    
    # Optimizers setup
    generator_lineardecay = utils.LinearDecay(initial_learning_rate=args.gen_initial_lr,
                                              total_steps=args.epochs * n_train,
                                              step_decay=args.epoch_decay * n_train)
    
    discriminator_lineardecay = utils.LinearDecay(initial_learning_rate=args.disc_initial_lr,
                                              total_steps=args.epochs * n_train,
                                              step_decay=args.epoch_decay * n_train)
    
    generator_optimizer = tf.keras.optimizers.Adam(generator_lineardecay, beta_1=args.beta)
    discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_lineardecay, beta_1=args.beta) 

    # Checkpoint management
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), 
                              generator=gen, 
                              discriminator=disc,
                              generator_optimizer=generator_optimizer,
                              discriminator_optimzer=discriminator_optimizer)

    manager = tf.train.CheckpointManager(ckpt, args.output_path, max_to_keep=args.max_to_keep, )   

    # Get sample image from training set to visualize training process
    sample = data_loader.sample_image(args.sample_image_dir)

    # Start training loop
    for epoch in tqdm(range(args.epochs)):
        metric_logger.on_epoch_begin()
        # Training step
        for step, data in enumerate(train_dataset):
       
            if tf.random.uniform(()) < 0.5: # Conditionally generate based on random choice (front or back)
                g_input, d_real_input, d_fake_cond = utils.create_conditional_gan_pairs(data, front=True)
            else: # Back
                g_input, d_real_input, d_fake_cond = utils.create_conditional_gan_pairs(data, front=False)
            
            train_losses = train_step(g_input, d_real_input, d_fake_cond)
            
            # Update metrics
            logs = {key: float(value.numpy()) for key, value in train_losses.items()}
            metric_logger.on_batch_end(logs)

        metric_logger.on_epoch_end()

        # Validation step - validate both front and back
        metric_logger.on_validation_begin()
        for test_step, val_data in enumerate(val_dataset):
            target_front = val_data["bs_front"]
            target_back = val_data["bs_back"]
            g_input_front, _, _ = utils.create_conditional_gan_pairs(val_data, front=True)
            g_input_back, _, _ = utils.create_conditional_gan_pairs(val_data, front=False)
            translated_front, logs = val_step(g_input_front, target_front)
            translated_back, _ = val_step(g_input_back, target_back)
            logs = {key: float(value.numpy()) for key, value in logs.items()}
            metric_logger.on_validation_batch_end(logs, args.save_every, 
                                                  translated_image=np.concatenate([translated_front.numpy(), translated_back.numpy()], axis=0),
                                                  ground_truth=np.concatenate([target_front.numpy(), target_back.numpy()], axis=0)
                                                  )
        metric_logger.on_validation_end()

        if (epoch+1) % args.save_every == 0:
            # Save validation images 
            utils.save_images(args.output_path, epoch=epoch+1, preds=metric_logger.translated_images, ground_truths=metric_logger.ground_truths, title='val')
            manager.save()
            
            # Pss sample image through model to visualize trainging progress
            translated, _ = val_step(sample['inputs'], sample['targets'])
            utils.save_images(args.output_path, epoch=epoch+1, preds=[translated.numpy()], ground_truths=[sample['targets'].numpy()], title='sample')
            
            # Log losses
            metric_logger.on_checkpoint(save_path=args.output_path)


