import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Conv3D, Activation, LeakyReLU, Conv2DTranspose, Dropout, ZeroPadding2D, Input
from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow_addons.layers import SpectralNormalization



def build_generator(arch='unet-3d', base_filters=64, num_blocks=7, output_channels=2, use_dropout=False):
    if arch == 'unet-3d':
        return UNet3D(num_blocks=num_blocks, base_filters=base_filters, output_channels=output_channels)
    
    elif arch == 'pix2pix':
        return Pix2PixGenerator(input_nc=2, output_nc=output_channels, num_downs=num_blocks, ngf=base_filters, use_dropout=use_dropout, norm_type='instance')

    else:
        raise ValueError("Unsupported architecture specified. Use 'unet-3d' or 'pix2pix'.")

 
def build_discriminator(arch='pixel', input_shape=(128, 128, 3), base_filters=64, norm_type='instance', return_intermediate=False):
    if arch == 'pixel':
        return PixelDiscriminator(dim=base_filters, norm_type=norm_type, return_intermediate=return_intermediate, feature_names=['conv1', 'activation2'])
    
    elif arch == 'patch':
        return PatchDiscriminator(dim=base_filters, norm_type=norm_type, input_shape=input_shape, return_intermediate=True, feature_names=['conv1', 'conv2'])
    
    elif arch == 'MedGAN':
        return MedGANDiscriminator(dim=base_filters, return_intermediate=return_intermediate, feature_name=['conv1', 'conv2'])

    else:
      raise ValueError("Unsupported architecture specified. Use 'pixel', 'patch', or 'MedGAN'.")


def build_extractor(arch='vgg19', input_size=(128, 128, 3)):
    if arch == 'vgg19':
        model = StyleContentVGG(style_layers=['block1_conv1', 'block5_conv1'],
                                content_layers=['block'+str(i)+'_conv1' for i in range(1, 5)],
                                input_size=input_size)
    return model

# =============================================================================================
#                                    FEATURE EXTRACTOR
# =============================================================================================

def vgg_layers(layer_names, input_size):
    """ Creates a VGG model that returns a list of intermediate output values."""
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(input_size[0], input_size[1], 3))
    vgg.trainable = False

    outputs = [vgg.get_layer(layer.name).output for layer in vgg.layers if layer.name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


class StyleContentVGG(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers, input_size):
        super(StyleContentVGG, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers, input_size)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = (inputs+1)*127.5
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                        outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                        for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# =============================================================================================
#                                       GENERATOR
# =============================================================================================
class ConvBlock3D(layers.Layer):
    def __init__(self, num_filters):
        super(ConvBlock3D, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv1 = Conv3D(num_filters, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(0.2), kernel_initializer=initializer)
        self.normalization = norm_layer('instance')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

class ConvBlock2D(layers.Layer):
    def __init__(self, num_filters):
        super(ConvBlock3D, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv1 = Conv2D(num_filters, 3, padding='same', activation=LeakyReLU(0.2), kernel_initializer=initializer)
        self.conv2 = Conv2D(num_filters, 3, padding='same', activation=LeakyReLU(0.2), kernel_initializer=initializer)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class DownsampleBlock(layers.Layer):
    def __init__(self, num_filters):
        super(DownsampleBlock, self).__init__()
        self.conv_block = ConvBlock3D(num_filters)

    def call(self, inputs):
        x = self.conv_block(inputs)
        p = self.pool(x)
        return x, p

class UpsampleBlock(layers.Layer):
    def __init__(self, num_filters):
        super(UpsampleBlock, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.upconv = Conv2DTranspose(num_filters // 2, 2, strides=2, padding='same', kernel_initializer=initializer)
        self.conv_block = ConvBlock3D(num_filters // 2)

    def call(self, inputs, skip_features):
        x = self.upconv(inputs)
        x = layers.concatenate([x, skip_features], axis=-1)
        x = self.conv_block(x)
        return x

class UNet3D(tf.keras.Model):
    def __init__(self, base_filters=64, num_blocks=6, output_channels=1):
        super(UNet3D, self).__init__()
        self.down_blocks = []
        self.up_blocks = []
        self.num_blocks = num_blocks

        # Contracting Path (Encoder)
        for i in range(num_blocks):
            num_filters = base_filters * min((2 ** i), 8)
            down_block = DownsampleBlock(num_filters)
            self.down_blocks.append(down_block)

        # Expanding Path (Decoder)
        for i in range(num_blocks - 1, -1, -1):
            num_filters = base_filters * (2 ** i)
            up_block = UpsampleBlock(num_filters * 2)
            self.up_blocks.append(up_block)

        # Bottleneck
        self.bottleneck = ConvBlock3D(base_filters * (2 ** num_blocks))

        # Final Convolution
        self.final_conv = Conv2D(output_channels, 1, activation='sigmoid')

    def call(self, inputs):
        skip_connections = []

        x = inputs
        # Contracting Path
        for down_block in self.down_blocks:
            x, p = down_block(x)
            skip_connections.append(x)
            x = p

        # Bottleneck
        x = self.bottleneck(x)

        # Expanding Path
        skip_connections = skip_connections[::-1]
        for up_block, skip_features in zip(self.up_blocks, skip_connections):
            x = up_block(x, skip_features)

        # Reduce volume to a 2D image
        x = tf.reduce_mean(x, axis=-2)  # Assuming 'channels_last', reduce the 3rd dimension

        # Final Convolution to get to the desired output shape
        x = self.final_conv(x)

        return x


#%% Copy and Paste from official code
class Pix2PixGenerator(tf.keras.Model):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf, norm_type='batch', use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Pix2PixGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_type=norm_type, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_type=norm_type, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_type=norm_type)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_type=norm_type)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_type=norm_type)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_type=norm_type)  # add the outermost layer

    def call(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(tf.keras.Model):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_type='batch', use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        initializer = tf.random_normal_initializer(0., 0.02)

        use_bias = norm_type != 'batch'
            
        if input_nc is None:
            input_nc = outer_nc
            
        downconv = Conv3D(inner_nc, kernel_size=4,
                             strides=2, padding='same', kernel_initializer=initializer, use_bias=use_bias)
        
        downrelu = LeakyReLU(0.2)
        downnorm = norm_layer(norm_type)
        uprelu = LeakyReLU(0.2)
        upnorm = norm_layer(norm_type)

        if outermost:
            upconv = Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding='same',  kernel_initializer=initializer)
            down = [downconv]
            up = [uprelu, upconv, Activation('tanh')]
            model = down + [submodule] + up
            
        elif innermost:
            upconv =Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, layers.Lambda(lambda x: tf.math.reduce_max(x, axis=2)), upconv, upnorm]
            model = down + up
        
        else:
            upconv = Conv2DTranspose(outer_nc, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [Dropout(0.3)]
            else:
                model = down + [submodule] + up

        self.model = Sequential(model)

    def call(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return tf.concat([tf.math.reduce_max(x, axis=2), self.model(x)], -1)        

# =============================================================================================
#                                        DISCRIMINATOR
# =============================================================================================

class PixelDiscriminator(tf.keras.Model):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    
    def __init__(self, dim, norm_type='instance', return_intermediate=True, feature_names=['conv1']):
        super(PixelDiscriminator, self).__init__()
        self.use_bias = norm_type != 'batch'
        self.return_intermediate = return_intermediate
        self.feature_names = feature_names
    
        # Define layers
        self.conv1 = Conv2D(filters=dim, kernel_size=1, use_bias=self.use_bias, name='conv1')
        self.leaky_relu1 = LeakyReLU(0.2, name='leaky_relu1')
        self.conv2 = Conv2D(filters=dim * 2, kernel_size=1, use_bias=self.use_bias, name='conv2')
        self.norm = norm_layer(norm_type=norm_type)
        self.leaky_relu2 = LeakyReLU(0.2, name='leaky_relu2')
        self.conv3 = Conv2D(filters=1, kernel_size=1, use_bias=self.use_bias, name='conv3')

    def call(self, x):
        outputs = {'intermediate': []}

        x = self.conv1(x)
        x = self.leaky_relu1(x)
        if self.return_intermediate and 'conv1' in self.feature_names:
            outputs['intermediate'].append(x)

        x = self.conv2(x)
        x = self.norm(x)
        x = self.leaky_relu2(x)
        if self.return_intermediate and 'conv2' in self.feature_names:
            outputs['intermediate'].append(x)

        x = self.conv3(x)
        outputs['final'] = x

        return outputs
 



class PatchDiscriminator(tf.keras.Model):
    """ Try to tell real or fake in 70x70 patches. """
    
    def __init__(self, input_shape, dim=64, n_conv_layers=3, norm_type='instance', return_intermediate=True, feature_names=['conv1', 'conv2']):
        super().__init__()
        self.n_filters = dim
        self.n_conv_layers = n_conv_layers
        self.norm_type = norm_type
        self.return_intermediate = return_intermediate
        self.feature_names = feature_names
        self.patchgan = self._build_patchgan(dim, input_shape)

    def _build_patchgan(self, dim, input_shape):
        patchgan = Sequential(name='patch_discrminator')
        patchgan.add(Input(input_shape))
        patchgan.add(Conv2D(filters=dim, kernel_size=4, strides=2, padding='same', name='conv1'))
        patchgan.add(LeakyReLU(0.2))
        for i in range(1, self.n_conv_layers):
            expand_rate = min(2 ** i, 8)
            patchgan.add(Conv2D(filters=dim*expand_rate, kernel_size=4, strides=2, padding='same', name=f'conv{i+1}'))
            patchgan.add(norm_layer(self.norm_type))
            patchgan.add(LeakyReLU(alpha=0.2))

        # second last ouput layer
        patchgan.add(ZeroPadding2D())
        patchgan.add(Conv2D(filters=dim*expand_rate, kernel_size=4, strides=1, padding='valid', name=f'conv{i+2}'))
        patchgan.add(norm_layer(self.norm_type))
        
        patchgan.add(LeakyReLU(alpha=0.2))

        # last (patch) output
        patchgan.add(ZeroPadding2D())
        patchgan.add(Conv2D(filters=1, kernel_size=4, padding='valid', name=f'conv{i+3}'))

        return patchgan



    def call(self, x):
        outputs = {"intermediate": []}
        for layer in self.patchgan.layers:
            x = layer(x)
            if self.return_intermediate and layer.name in self.feature_names:
                outputs["intermediate"].append(x)
        outputs["final"] = x
        
        return outputs



class MedGANDiscriminator(tf.keras.Model):
    def __init__(self, dim, return_intermediate=False, feature_names=['conv1'], norm_type='batch'):
        """
        Parameters:
            dim (int) -- the number of filters in the first layer
            return_intermediate (bool) -- whether to return intermediate feature maps
            featuer_name (int)  -- the name of the layer to return the ouput of. Only relative when return_intermediate=True
            norm_type(str) -- which kind of normalization layer to use
        """
        super(MedGANDiscriminator, self).__init__()
        self.model = Sequential([
            #Input(input_size),
            SpectralNormalization(Conv2D(filters=dim, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(0.2)), power_iterations=1, name='conv1'),
            norm_layer(norm_type),
            SpectralNormalization(Conv2D(filters=dim*2, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(0.2)), power_iterations=1, name='conv2'),
            norm_layer(norm_type),
            SpectralNormalization(Conv2D(filters=1, kernel_size=4, strides=1, padding='valid', name='final_conv'), power_iterations=1, name='conv3')
        ])
        
        self.feature_names = feature_names
        self.return_intermediate = return_intermediate


   

    def call(self, x):
        outputs = {'intermediate': []}
        for layer in self.model.layers:
            x = layer(x)

            if self.return_intermediate and layer.name in self.feature_name:
                outputs['intermediate'].append(x)
            elif not self.return_intermediate:
                outputs['intermediate'] = None

        outputs['final'] = x

        return outputs


    

    
# =============================================================================================
#                                       Normalization Layers
# =============================================================================================
   
def norm_layer(norm_type='instance'):
    if norm_type == 'instance':
        return InstanceNormalization()
    elif norm_type == 'batch':
        return BatchNormalization()
    else: 
        print('Unreconginzed normalization type. Implementing BatchNormaliation.')
        return BatchNormalization()



class InstanceNormalization(tf.keras.layers.Layer): 
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
