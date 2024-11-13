import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero
    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        a = self._steps - self._step_decay
        b = step - self._step_decay
        self.current_learning_rate.assign(tf.cond(step >= self._step_decay,
                                                  true_fn=lambda: tf.cast(self._initial_learning_rate * (1 - (b+1)/ a), tf.float32),
                                                  false_fn=lambda: self._initial_learning_rate))
        return self.current_learning_rate


def resize_with_crop_or_pad(image, target_size, axis=None):
    """
    Crop or pad the image along the specified axes to ensure dimensions match `target_sizes`.
    Assumes `image` is a TensorFlow tensor of shape (W, H, D, C) or similar.
    
    Args:
        image (tf.Tensor): The input tensor.
        target_sizes (list or tuple): The desired sizes for each specified axis. 
                                      Should match the length of `axes`.
        axes (list or tuple): The axes along which to crop or pad. 
                              If None, it will adjust all axes except the last one (assumed to be the channel dimension).
        
    Returns:
        tf.Tensor: The tensor with specified axes cropped or padded to `target_sizes`.
    """
    if axis is None:
        # If no specific axes are provided, adjust all axes
        axis = list(range(len(image.shape)))
    
    if len(target_size) != len(axis):
        raise ValueError("Length of `target_sizes` must match length of `axes`.")
    
    for axis, target_size in zip(axis, target_size):
        # Get the current size along the specified axis
        current_size = tf.shape(image)[axis]
        
        # Calculate the difference
        size_diff = target_size - current_size
        
        # Crop if the current size is larger than the target size
        if size_diff < 0:
            crop_start = (current_size - target_size) // 2
            slices = [slice(None)] * len(image.shape)
            slices[axis] = slice(crop_start, crop_start + target_size)
            image = image[tuple(slices)]
        
        # Pad if the current size is smaller than the target size
        elif size_diff > 0:
            pad_before = size_diff // 2
            pad_after = size_diff - pad_before
            paddings = [[0, 0]] * len(image.shape)
            paddings[axis] = [pad_before, pad_after]
            image = tf.pad(image, paddings, mode='CONSTANT', constant_values=0)
    
    return image


def create_conditional_gan_pairs(data, front=True):
    """
    Generates conditional input-output pairs for a GAN model with specified view condition.

    Creates inputs for both the Generator and Discriminator in a GAN,
    where the input consists of CT and PT scan data concatenated 
    with a conditional channel (indicating front or back view). The target output is a
    front or back view of a bone scan, based on the specified condition.

    Parameters:
    data (dict): A dictionary containing CT, PT, and bone scan (front and back) data.
                 Expected keys are 'ct', 'pt', 'bs_front', and 'bs_back'.
    front (bool): A flag to indicate if the front (True) or back (False) view should be used 
                  for conditioning.

    Returns:
    tuple: A tuple containing:
        - g_input (Tensor): Input for the Generator, with CT, PT, and condition channel.
        - d_real_input (Tensor): Real input for the Discriminator, with target, PT projection,
          and condition channel.
        - d_fake_cond (Tensor): Condition for the fake image that goes through Discriminator, with PT projection and 
          condition channel.
    """
    if front: # Front
        g_cond = tf.ones_like(data["ct"])
        d_cond = tf.ones_like(data["bs_front"])
        target = data["bs_front"]
    
    else: # Back
        g_cond = tf.zeros_like(data["ct"])
        d_cond = tf.zeros_like(data["bs_back"])
        target = data["bs_back"]
    
    pt_projection = tf.reduce_max(data["pt"], axis=2)

    # Generate inputs for Generator and Discriminator

    g_input = tf.concat([data["ct"], data["pt"], g_cond], axis=-1)
    d_real_input = tf.concat([target, pt_projection, d_cond], axis=-1)
    d_fake_cond = tf.concat([pt_projection, d_cond], axis=-1)

    return g_input, d_real_input, d_fake_cond


class MetricLoggerCallback:
    def __init__(self):
        self.epoch = 0
        self.batch = 0
        self.logs = {}
        self.train_epoch_logs = []
        self.batch_logs = []
        self.validation_epoch_logs=[]
        self.translated_images = []
        self.ground_truths = []

    def on_train_begin(self):
        """Called at the start of training."""
        self.epoch = 0
        self.train_epoch_logs = []

    def on_epoch_begin(self):
        """Called at the start of each epoch."""
        self.batch = 0
        self.batch_logs = []

    def on_batch_end(self, logs):
        """Called at the end of each batch."""
        self.batch_logs.append(logs.copy())
        self.batch += 1

    def on_epoch_end(self):
        """Called at the end of each epoch."""
        epoch_log = {}
        for key in self.batch_logs[0]:
            epoch_log[key] = sum(d[key] for d in self.batch_logs) / len(self.batch_logs)
        self.train_epoch_logs.append(epoch_log)
        self.epoch += 1

        print(f"Epoch {self.epoch} metrics:")
        for key, value in epoch_log.items():
            print(f"    {key}: {value:.4f}")
    
    def on_checkpoint(self, save_path):
        """Called at checkpoint epochs.
        Save metrics line plot"""
        train_metric_names = self.train_epoch_logs[0].keys()
        val_metric_names = self.validation_epoch_logs[0].keys()
        plt.figure(figsize=(16,6))

        for metric_name in train_metric_names:
            metric_values = [train_log[metric_name] for train_log in self.train_epoch_logs]
            plt.plot(range(self.epoch), metric_values, label=f'Train {metric_name}')

        for metric_name in val_metric_names:
            metric_values = [val_log[metric_name] for val_log in self.validation_epoch_logs]
            plt.plot(range(self.epoch), metric_values, label=f'Validation {metric_name}', linestyle='--')

        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training and Validation Metrics by Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/epoch_{self.epoch}/metrics.png")
        plt.close()
        
    def on_validation_begin(self):
        self.validation_logs = []
        self.translated_images = []
        self.ground_truths = []

    def on_validation_batch_end(self, logs, save_every, translated_image, ground_truth):
        self.validation_logs.append(logs.copy())
        
        if self.epoch % save_every == 0:
            self.translated_images.append(translated_image)
            self.ground_truths.append(ground_truth)

    def on_validation_end(self):
        val_log = {}
        for key in self.validation_logs[0]:
            val_log[key] = sum(d[key] for d in self.validation_logs) / len(self.validation_logs)
        self.validation_epoch_logs.append(val_log)

        print(f"Validation metrics:")
        for key, value in val_log.items():
            print(f"    {key}: {value:.4f}")



def save_images(save_path, epoch, preds, ground_truths, title):
    """
    Plots the batch of ground truth and predicted images in separate files
    
    Parameters:
    - save_path: Designated directory to save
    - epoch: Current epoch number
    - preds: Batch of translated images, each cropped into 18 crops
    - ground_truths: Batch of ground truth images, each cropped into 18 crops
    
    """
    os.makedirs(f"{save_path}/epoch_{epoch}", exist_ok=True)
    for step, (translated, ground_truth) in enumerate(zip(preds, ground_truths)):
        fig, axs = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle(title, fontsize=16)

        for i in range(2):# Original images (top row)
            axs[0, 2 * i].imshow(ground_truth[i], cmap='gray')
            axs[0, 2 * i].set_title(f"Ground Truth {i+1} (Original)")
            axs[0, 2 * i].axis('off')

            axs[0, 2 * i + 1].imshow(translated[i], cmap='gray')
            axs[0, 2 * i + 1].set_title(f"Prediction {i+1} (Original)")
            axs[0, 2 * i + 1].axis('off')

            # Clipped images (bottom row)
            clipped_gt = np.clip(ground_truth[i], a_min=ground_truth[i].min(), a_max=np.percentile(ground_truth[i], 95))  
            clipped_pred = np.clip(translated[i], a_min=translated[i].min(), a_max=np.percentile(translated[i], 95))  # Modify this to your clipping range

            axs[1, 2 * i].imshow(clipped_gt, cmap='gray')
            axs[1, 2 * i].set_title(f"Ground Truth {i+1} (Clipped)")
            axs[1, 2 * i].axis('off')

            axs[1, 2 * i + 1].imshow(clipped_pred, cmap='gray')
            axs[1, 2 * i + 1].set_title(f"Prediction {i+1} (Clipped)")
            axs[1, 2 * i + 1].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust top to make room for the main title
        plt.savefig(f"{save_path}/epoch_{epoch}/{title}_{step}.png")
        plt.close()
        



