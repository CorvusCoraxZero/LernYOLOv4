from functools import wraps
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Layer
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


#   单次卷积
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}  # 相比于其他的卷积 这里引入了l2正则化
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


#   卷积块  使更方便的构建整个网络
#   DarknetConv2D + BatchNormalization + Mish(激活函数)
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


#   CSPdarknet的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    # 进行长和宽的压缩
    preconv1 = ZeroPadding2D(((1, 0), (1, 0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(preconv1)

    # 生成一个大的残差边 
    shortconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)

    # 主干部分的卷积
    mainconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)
    # 1x1卷积对通道数进行整合->3x3卷积提取特征，使用残差结构
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Mish(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (3, 3)))(mainconv)
        mainconv = Add()([mainconv, y])
    # 1x1卷积后和残差边堆叠
    postconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(mainconv)
    route = Concatenate()([postconv, shortconv])

    # 最后对通道数进行整合 (利用1*1的卷积)
    return DarknetConv2D_BN_Mish(num_filters, (1, 1))(route)


#   darknet53 的主体部分
def darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3, 3))(x)  # 进行了一个普通的卷积块 通道数是32
    # 进行了5次 resblock_body 最后一个参数 1 2 8 8 4 指的是resblock_body 内部残差结构堆积的次数
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x

    # feat1,feat2,feat3是要取出的三个有效特征层 分别对应(52,52,256), (26,26,512), (13,13,1024) 前两个是长宽后面是通道数
    return feat1, feat2, feat3
