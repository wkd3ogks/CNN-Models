""" Simple implementation of ResNet for CIFAR-10 dataset

PlainNet refers to the same architecture as ResNet but without shortcut
"""
import tensorflow as tf

def residual_block(
        name, x, filters, kernel_size=3, stride=1, use_shortcut=True
):
    """A residual block(2 conv layer) for ResNet
    
    Args:
        x: input tensor
        filters: number of filters
        kernel_size: kernel size
        stride: stride
        name: name of the block
        use_shortcut: if True, shortcut is used

    Returns:
        Output tensor for the residual block
    """

    shortcut = x
    
    # --- First convolutional layer ---
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=(stride, stride), padding='same',
        kernel_initializer=tf.keras.initializers.HeNormal(seed=42), name=name + '_1_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_1_bn',)(x)
    x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)


    # --- Second convolutional layer ---
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, padding='same',
        kernel_initializer=tf.keras.initializers.HeNormal(seed=42), name=name + '_2_conv'
    )(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name=name + '_2_bn')(x)
    
    # if dimension increases, apply option A(refers to original paper) which is using zero-padding
    if x.shape[-1] != shortcut.shape[-1]: 
        # make same spatial dimensions / work similar to 1x1 convolution with stride=2(throw away half of the pixels)
        shortcut = tf.keras.layers.MaxPooling2D(
            pool_size=(1, 1), strides=(2, 2), name=name+'_spatial_pool'
        )(shortcut) 

        # zero-padding for increased channel dimension
        zero_pad = tf.keras.ops.zeros_like(shortcut)
        shortcut = tf.keras.ops.concatenate([shortcut, zero_pad], axis=-1)
    
    # add shortcut
    if use_shortcut:
        x = tf.keras.layers.Add(name=name + '_add_shortcut')([shortcut, x]) 

    x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)
    return x

def stack_residual_blocks(
    name, n, x, filters, is_plain, stride1 = 1 
):
    """stacks residual blocks for ResNet

    Args:
        name: name of the stacked blocks
        n: number of residual blocks(2n)
        x: input tensor
        filters: number of filters
        is_plain: if True, residual blocks does not have shortcut
        stride1: stride for the first residual block
    return:
        Output tensors for the stacked residual blocks
    """
    x = residual_block(f'{name}_{1}', x, filters, stride=stride1, use_shortcut=not is_plain)
    for i in range(2, n + 1):
        x = residual_block(f'{name}_{i}', x, filters, use_shortcut= not is_plain)
    return x

def ResNet(n = 3, is_plain=False):
    """Instantiates the ResNet or PlainNet architecture for CIFAR-10 dataset

    n = { 3, 5, 7, 9, ... } leading to 20, 32, 44, 56-layer network

    Args:
        n: number of residual blocks per stack(2n)
        is_plain: if True, PlainNet is created. Otherwise, ResNet is created
    Returns:
        A (6n + 2)-layer ResNet or PlainNet model 
    """

    input_layer = tf.keras.Input(shape=(32, 32, 3), name='input')

    # output map size: 32x32, layers: 2n + 1 = 7, filters: 16
    x = tf.keras.layers.Conv2D(
        16, 3, padding='same',
        kernel_initializer=tf.keras.initializers.HeNormal(seed=42), name='conv1_conv'
    )(input_layer)
    x = tf.keras.layers.BatchNormalization(epsilon=1.001e-5, name='conv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

    x = stack_residual_blocks('conv2', n, x, 16, is_plain, stride1=1)
    x = stack_residual_blocks('conv3', n, x, 32, is_plain, stride1=2)
    x = stack_residual_blocks('conv4', n, x, 64, is_plain, stride1=2)

    # global average pooling and 10-way fully-connected layer and softmax
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(
        10, kernel_initializer=tf.keras.initializers.HeNormal(seed=42), name='fc10'
    )(x)
    output_layer = tf.keras.layers.Activation('softmax', name='softmax')(x)

    if not is_plain:
        print(f'Successfully created {6 * n + 2}-layer ResNet')
    else:
        print(f'Successfully created {6 * n + 2}-layer PlainNet')
    
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)