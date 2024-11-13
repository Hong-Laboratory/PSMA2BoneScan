import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np
import os
from glob import glob
import math
import utils


def load_3d(file_path: str):
    """
    Load a 3D PET or CT scan from a numpy file.
    """
    image = np.load(file_path)
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    return image


def load_bonescans(file_path: str): 
    """
    Load a 2D bone scan (front and back) from a numpy file.
    """
    bs_front = np.load(os.path.join(file_path, "frontNM.npy"))
    bs_back = np.load(os.path.join(file_path, "backNM.npy"))
    bs_front, bs_back = tf.convert_to_tensor(bs_front), tf.convert_to_tensor(bs_back)
    bs_front, bs_back = tf.cast(bs_front, tf.float32), tf.cast(bs_back, tf.float32)
    
    return bs_front, bs_back  


def random_crop(ct, pt, bs_front, bs_back, crop_size=(128, 128, 128)):
    """
    Randomly crop the 3D PET/CT and the corresponding 2D bone scans
    to the same width and depth coordinates.
    """
    h, d, w = pt.shape
    crop_h, crop_d, crop_w = crop_size
    
    # Randomly select the cropping start points
    start_h = tf.random.uniform(shape=(), minval=0, maxval=h-crop_h+1, dtype=tf.int32)
    start_w = tf.random.uniform(shape=(), minval=20, maxval=(w-20) - crop_w + 1, dtype=tf.int32)
    
    ct_cropped = ct[start_h:start_h + crop_h, :, start_w:start_w + crop_w]
    pt_cropped = pt[start_h:start_h + crop_h, :, start_w:start_w + crop_w]
    bs_front_cropped = bs_front[start_h:start_h + crop_h, start_w:start_w + crop_w, ...]
    bs_back_cropped = bs_back[start_h:start_h + crop_h, start_w:start_w + crop_w, ...]
    
    return ct_cropped, pt_cropped, bs_front_cropped, bs_back_cropped


def augmentations(ct, pt, bs_front, bs_back):
    """Apply Augmentations to each modalities"""
    ct, pt, bs_front, bs_back = random_rotation(ct, pt, bs_front, bs_back, p=0.5)
    ct, pt = random_noise(ct, pt, p=0.25)
    bs_front = gaussian_filter(bs_front, p=1)
    bs_back = gaussian_filter(bs_back, p=1)
    
    return ct, pt, bs_front, bs_back


def random_rotation(ct, pt, bs_front, bs_back, p=0.25):
    """
    Apply random 3D rotation to PET/CT and corresponding 2D rotation to the bone scan
    in a certain percentage `p`. 3D rotation only applied in the xy plane
    """
    if tf.random.uniform(()) < p:
        # Random rotation angles
        angle = tf.random.uniform((), minval=-0.2, maxval=0.2) * tf.constant(math.pi)
        # Transpose 3D image axis to use off-the-shelf 2D rotation func
        pt, ct = tf.transpose(pt, [0, 2, 1, 3]), tf.transpose(ct, [0, 2, 1, 3])
        # Apply rotation and re-transpose
        pt, ct = tfa.image.rotate(pt, angle, fill_mode='constant', fill_value=tf.reduce_min(pt)), tfa.image.rotate(ct, angle, fill_mode='constant', fill_value=tf.reduce_min(ct))
        bs_front, bs_back = tfa.image.rotate(bs_front, angle, fill_mode='constant', fill_value=tf.reduce_min(bs_front)), tfa.image.rotate(bs_back, angle, fill_mode='constant', fill_value=tf.reduce_min(bs_back))
        pt, ct = tf.transpose(pt, [0, 2, 1, 3]), tf.transpose(ct, [0, 2, 1, 3])
        
    return ct, pt, bs_front, bs_back
    

def random_noise(ct, pt, p=0.2):
    """
    Apply random Gaussian noise to PET/CT with probability `p`.
    """
    if tf.random.uniform(()) < p:
        ct_noise = tf.random.normal(shape=tf.shape(ct), mean=0.0, stddev=0.15, seed=42, dtype=tf.float32)
        pt_noise = tf.random.normal(shape=tf.shape(pt), mean=0.0, stddev=0.15, seed=42, dtype=tf.float32)
        ct = tf.clip_by_value(ct+ct_noise, -1, 1)
        pt = tf.clip_by_value(pt+pt_noise, -1, 1)
    
    return ct, pt
    

def crop_and_resize(image, resize_to):
    image = image[:430, ...]
    image = tf.image.resize(image, resize_to)
    
    return image


def gaussian_filter(image, p=1):
    """
    Apply Gaussian smoothing to image with probability p.
    """
    if tf.random.uniform(()) < p:
        image = tfa.image.gaussian_filter2d(image, sigma=0.7)
    
    return image


def window_image(image, low, high):
    """
    Apply windowing to the PET/CT image based on `low` and `high` intensity thresholds.
    """
    image = tf.cast(image, tf.float32)
    windowed = tf.clip_by_value(image, low, high)
    windowed = (windowed - low) / (high - low)
    
    return windowed


def percentile_normalization(image, percentile=99.5):
    """
    Normalize image intensities based on a given percentile.
    """
    image = tf.cast(image, tf.float32)
    upper = tfp.stats.percentile(image, percentile)
    
    return tf.clip_by_value(image / upper, 0, 1)


def min_max_normalization(image):
    """
    Apply min-max normalization to scale image to [0, 1].
    """
    image = tf.cast(image, tf.float32)
    min_val = tf.reduce_min(image)
    max_val = tf.reduce_max(image)
    
    return (image - min_val) / (max_val - min_val)


def standardization(image):
    """
    Apply standardization to scale image to have mean 0 and standard deviation 1.
    """
    image = tf.cast(image, tf.float32)
    mean = tf.reduce_mean(image)
    std = tf.math.reduce_std(image)
   
    return (image - mean) / std


def DataGenerator(dir_path, input_size, train=True):
    """Wraps the preprocessing steps to one function.
    Load image -> random crop -> window CT -> augmentation 
    -> normalization -> match dim shape
    """
    for path in glob(f"{dir_path}/*/(2, 2, 4)"):

        # Load images as tf tensors
        ct = load_3d(f"{path}/CT.npy")
        pt = load_3d(f"{path}/PT_SUV.npy")
        bs_front, bs_back = load_bonescans(path)
        data = preprocess_pipeline(ct, pt, bs_front, bs_back, input_size=input_size, train=train)

        yield data


def preprocess_pipeline(ct, pt, bs_front, bs_back, input_size, train):
        # Slice or pad 3D images at the coronal axis
        ct, pt = utils.resize_with_crop_or_pad(ct, axis=[0, 1, 2], target_size=input_size), utils.resize_with_crop_or_pad(pt, axis=[0, 1, 2], target_size=input_size)
 
        # Window CT
        ct = window_image(ct, 0, 1000)
        
        # Normalization
        ct = min_max_normalization(ct)
        pt = min_max_normalization(pt)
        bs_front = percentile_normalization(bs_front, percentile=99.5)
        bs_back = percentile_normalization(bs_back, percentile=99.5)
       
        # Add a channel dimension
        ct = tf.expand_dims(ct, axis=-1)
        pt = tf.expand_dims(pt, axis=-1)
        bs_front = tf.expand_dims(bs_front, axis=-1)
        bs_back = tf.expand_dims(bs_back, axis=-1)
       
        # Crop and resize bone scans
        bs_front = crop_and_resize(bs_front, resize_to=(256, 256))
        bs_back = crop_and_resize(bs_back, resize_to=(256, 256))
        
        # Augmentations
        if train:
            ct, pt, bs_front, bs_back = augmentations(ct, pt, bs_front, bs_back)

        return {"ct": ct, "pt": pt, "bs_front": bs_front, "bs_back": bs_back}


def sample_image(image_path, crop_size=(256, 256, 256)):
    """
    Samples and preprocess a single image

    Args:
        image_path (str): The image path of sample image.
        crop_size (tuple): The size of each crop (height, width).
    
    Returns:
    dict: A dictionary containing:
            - "inputs": A tensor of shape (6, crop_H, crop_W, 2), overlapping input PET/CT
            - "targets": A list of six corresponding target crops
    """
    
    bs_front = np.load(f"{image_path}/frontNM.npy")
    bs_back = np.load(f"{image_path}/backNM.npy")
    ct = np.load(f"{image_path}/CT.npy")
    pt = np.load(f"{image_path}/PT_SUV.npy")

    data = preprocess_pipeline(ct, pt, bs_front, bs_back, input_size=crop_size, train=False)
    g_input_front, _, _ = utils.create_conditional_gan_pairs(data, front=True)
    g_input_back, _, _ = utils.create_conditional_gan_pairs(data, front=False)

    inputs = tf.concat([g_input_front[tf.newaxis, ...], g_input_back[tf.newaxis, ...]], axis=0)
    targets = tf.concat([data['bs_front'][tf.newaxis, ...], data['bs_back'][tf.newaxis, ...]], axis=0)
    
    return {"inputs": inputs, "targets": targets}


def create_dataset(data_dir, buffer_size=1, batch_size=1, input_size=(128, 256, 128), train=True):
    if train:
        dataset = tf.data.Dataset.from_generator(
            lambda: DataGenerator(data_dir, input_size=input_size, train=train), 
            output_signature={
                "ct": tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
                "pt": tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
                "bs_front": tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                "bs_back": tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)
            }
        ).shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        

    else:
        dataset = tf.data.Dataset.from_generator(
            lambda: DataGenerator(data_dir, input_size=input_size, train=train), 
            output_signature={
                "ct": tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
                "pt": tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
                "bs_front": tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                "bs_back": tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)
            }
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE) 
    
    return dataset

