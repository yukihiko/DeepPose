import tensorflow as tf

__weights_dict = dict()

is_train = False

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    input           = tf.placeholder(tf.float32, shape = (None, 224, 224, 3), name = 'input')
    MobileNetn2nSequentialnmodelnnSequentialn0nnConv2dn0n138_pad = tf.pad(input, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn0nnConv2dn0n138 = convolution(MobileNetn2nSequentialnmodelnnSequentialn0nnConv2dn0n138_pad, group=1, strides=[2, 2], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn0nnConv2dn0n138')
    MobileNetn2nSequentialnmodelnnSequentialn0nnBatchNorm2dn1n139 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn0nnConv2dn0n138, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn0nnBatchNorm2dn1n139')
    MobileNetn2nSequentialnmodelnnSequentialn0nnReLUn2n140 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn0nnBatchNorm2dn1n139, name = 'MobileNetn2nSequentialnmodelnnSequentialn0nnReLUn2n140')
    MobileNetn2nSequentialnmodelnnSequentialn1nnConv2dn0n141_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn0nnReLUn2n140, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn1nnConv2dn0n141 = convolution(MobileNetn2nSequentialnmodelnnSequentialn1nnConv2dn0n141_pad, group=32, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn1nnConv2dn0n141')
    MobileNetn2nSequentialnmodelnnSequentialn1nnBatchNorm2dn1n142 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn1nnConv2dn0n141, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn1nnBatchNorm2dn1n142')
    MobileNetn2nSequentialnmodelnnSequentialn1nnReLUn2n143 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn1nnBatchNorm2dn1n142, name = 'MobileNetn2nSequentialnmodelnnSequentialn1nnReLUn2n143')
    MobileNetn2nSequentialnmodelnnSequentialn1nnConv2dn3n144 = convolution(MobileNetn2nSequentialnmodelnnSequentialn1nnReLUn2n143, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn1nnConv2dn3n144')
    MobileNetn2nSequentialnmodelnnSequentialn1nnBatchNorm2dn4n145 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn1nnConv2dn3n144, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn1nnBatchNorm2dn4n145')
    MobileNetn2nSequentialnmodelnnSequentialn1nnReLUn5n146 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn1nnBatchNorm2dn4n145, name = 'MobileNetn2nSequentialnmodelnnSequentialn1nnReLUn5n146')
    MobileNetn2nSequentialnmodelnnSequentialn2nnConv2dn0n147_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn1nnReLUn5n146, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn2nnConv2dn0n147 = convolution(MobileNetn2nSequentialnmodelnnSequentialn2nnConv2dn0n147_pad, group=64, strides=[2, 2], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn2nnConv2dn0n147')
    MobileNetn2nSequentialnmodelnnSequentialn2nnBatchNorm2dn1n148 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn2nnConv2dn0n147, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn2nnBatchNorm2dn1n148')
    MobileNetn2nSequentialnmodelnnSequentialn2nnReLUn2n149 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn2nnBatchNorm2dn1n148, name = 'MobileNetn2nSequentialnmodelnnSequentialn2nnReLUn2n149')
    MobileNetn2nSequentialnmodelnnSequentialn2nnConv2dn3n150 = convolution(MobileNetn2nSequentialnmodelnnSequentialn2nnReLUn2n149, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn2nnConv2dn3n150')
    MobileNetn2nSequentialnmodelnnSequentialn2nnBatchNorm2dn4n151 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn2nnConv2dn3n150, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn2nnBatchNorm2dn4n151')
    MobileNetn2nSequentialnmodelnnSequentialn2nnReLUn5n152 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn2nnBatchNorm2dn4n151, name = 'MobileNetn2nSequentialnmodelnnSequentialn2nnReLUn5n152')
    MobileNetn2nSequentialnmodelnnSequentialn3nnConv2dn0n153_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn2nnReLUn5n152, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn3nnConv2dn0n153 = convolution(MobileNetn2nSequentialnmodelnnSequentialn3nnConv2dn0n153_pad, group=128, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn3nnConv2dn0n153')
    MobileNetn2nSequentialnmodelnnSequentialn3nnBatchNorm2dn1n154 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn3nnConv2dn0n153, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn3nnBatchNorm2dn1n154')
    MobileNetn2nSequentialnmodelnnSequentialn3nnReLUn2n155 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn3nnBatchNorm2dn1n154, name = 'MobileNetn2nSequentialnmodelnnSequentialn3nnReLUn2n155')
    MobileNetn2nSequentialnmodelnnSequentialn3nnConv2dn3n156 = convolution(MobileNetn2nSequentialnmodelnnSequentialn3nnReLUn2n155, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn3nnConv2dn3n156')
    MobileNetn2nSequentialnmodelnnSequentialn3nnBatchNorm2dn4n157 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn3nnConv2dn3n156, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn3nnBatchNorm2dn4n157')
    MobileNetn2nSequentialnmodelnnSequentialn3nnReLUn5n158 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn3nnBatchNorm2dn4n157, name = 'MobileNetn2nSequentialnmodelnnSequentialn3nnReLUn5n158')
    MobileNetn2nSequentialnmodelnnSequentialn4nnConv2dn0n159_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn3nnReLUn5n158, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn4nnConv2dn0n159 = convolution(MobileNetn2nSequentialnmodelnnSequentialn4nnConv2dn0n159_pad, group=128, strides=[2, 2], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn4nnConv2dn0n159')
    MobileNetn2nSequentialnmodelnnSequentialn4nnBatchNorm2dn1n160 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn4nnConv2dn0n159, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn4nnBatchNorm2dn1n160')
    MobileNetn2nSequentialnmodelnnSequentialn4nnReLUn2n161 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn4nnBatchNorm2dn1n160, name = 'MobileNetn2nSequentialnmodelnnSequentialn4nnReLUn2n161')
    MobileNetn2nSequentialnmodelnnSequentialn4nnConv2dn3n162 = convolution(MobileNetn2nSequentialnmodelnnSequentialn4nnReLUn2n161, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn4nnConv2dn3n162')
    MobileNetn2nSequentialnmodelnnSequentialn4nnBatchNorm2dn4n163 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn4nnConv2dn3n162, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn4nnBatchNorm2dn4n163')
    MobileNetn2nSequentialnmodelnnSequentialn4nnReLUn5n164 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn4nnBatchNorm2dn4n163, name = 'MobileNetn2nSequentialnmodelnnSequentialn4nnReLUn5n164')
    MobileNetn2nSequentialnmodelnnSequentialn5nnConv2dn0n165_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn4nnReLUn5n164, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn5nnConv2dn0n165 = convolution(MobileNetn2nSequentialnmodelnnSequentialn5nnConv2dn0n165_pad, group=256, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn5nnConv2dn0n165')
    MobileNetn2nSequentialnmodelnnSequentialn5nnBatchNorm2dn1n166 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn5nnConv2dn0n165, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn5nnBatchNorm2dn1n166')
    MobileNetn2nSequentialnmodelnnSequentialn5nnReLUn2n167 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn5nnBatchNorm2dn1n166, name = 'MobileNetn2nSequentialnmodelnnSequentialn5nnReLUn2n167')
    MobileNetn2nSequentialnmodelnnSequentialn5nnConv2dn3n168 = convolution(MobileNetn2nSequentialnmodelnnSequentialn5nnReLUn2n167, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn5nnConv2dn3n168')
    MobileNetn2nSequentialnmodelnnSequentialn5nnBatchNorm2dn4n169 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn5nnConv2dn3n168, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn5nnBatchNorm2dn4n169')
    MobileNetn2nSequentialnmodelnnSequentialn5nnReLUn5n170 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn5nnBatchNorm2dn4n169, name = 'MobileNetn2nSequentialnmodelnnSequentialn5nnReLUn5n170')
    MobileNetn2nSequentialnmodelnnSequentialn6nnConv2dn0n171_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn5nnReLUn5n170, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn6nnConv2dn0n171 = convolution(MobileNetn2nSequentialnmodelnnSequentialn6nnConv2dn0n171_pad, group=256, strides=[2, 2], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn6nnConv2dn0n171')
    MobileNetn2nSequentialnmodelnnSequentialn6nnBatchNorm2dn1n172 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn6nnConv2dn0n171, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn6nnBatchNorm2dn1n172')
    MobileNetn2nSequentialnmodelnnSequentialn6nnReLUn2n173 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn6nnBatchNorm2dn1n172, name = 'MobileNetn2nSequentialnmodelnnSequentialn6nnReLUn2n173')
    MobileNetn2nSequentialnmodelnnSequentialn6nnConv2dn3n174 = convolution(MobileNetn2nSequentialnmodelnnSequentialn6nnReLUn2n173, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn6nnConv2dn3n174')
    MobileNetn2nSequentialnmodelnnSequentialn6nnBatchNorm2dn4n175 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn6nnConv2dn3n174, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn6nnBatchNorm2dn4n175')
    MobileNetn2nSequentialnmodelnnSequentialn6nnReLUn5n176 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn6nnBatchNorm2dn4n175, name = 'MobileNetn2nSequentialnmodelnnSequentialn6nnReLUn5n176')
    MobileNetn2nSequentialnmodelnnSequentialn7nnConv2dn0n177_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn6nnReLUn5n176, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn7nnConv2dn0n177 = convolution(MobileNetn2nSequentialnmodelnnSequentialn7nnConv2dn0n177_pad, group=512, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn7nnConv2dn0n177')
    MobileNetn2nSequentialnmodelnnSequentialn7nnBatchNorm2dn1n178 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn7nnConv2dn0n177, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn7nnBatchNorm2dn1n178')
    MobileNetn2nSequentialnmodelnnSequentialn7nnReLUn2n179 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn7nnBatchNorm2dn1n178, name = 'MobileNetn2nSequentialnmodelnnSequentialn7nnReLUn2n179')
    MobileNetn2nSequentialnmodelnnSequentialn7nnConv2dn3n180 = convolution(MobileNetn2nSequentialnmodelnnSequentialn7nnReLUn2n179, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn7nnConv2dn3n180')
    MobileNetn2nSequentialnmodelnnSequentialn7nnBatchNorm2dn4n181 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn7nnConv2dn3n180, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn7nnBatchNorm2dn4n181')
    MobileNetn2nSequentialnmodelnnSequentialn7nnReLUn5n182 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn7nnBatchNorm2dn4n181, name = 'MobileNetn2nSequentialnmodelnnSequentialn7nnReLUn5n182')
    MobileNetn2nSequentialnmodelnnSequentialn8nnConv2dn0n183_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn7nnReLUn5n182, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn8nnConv2dn0n183 = convolution(MobileNetn2nSequentialnmodelnnSequentialn8nnConv2dn0n183_pad, group=512, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn8nnConv2dn0n183')
    MobileNetn2nSequentialnmodelnnSequentialn8nnBatchNorm2dn1n184 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn8nnConv2dn0n183, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn8nnBatchNorm2dn1n184')
    MobileNetn2nSequentialnmodelnnSequentialn8nnReLUn2n185 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn8nnBatchNorm2dn1n184, name = 'MobileNetn2nSequentialnmodelnnSequentialn8nnReLUn2n185')
    MobileNetn2nSequentialnmodelnnSequentialn8nnConv2dn3n186 = convolution(MobileNetn2nSequentialnmodelnnSequentialn8nnReLUn2n185, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn8nnConv2dn3n186')
    MobileNetn2nSequentialnmodelnnSequentialn8nnBatchNorm2dn4n187 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn8nnConv2dn3n186, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn8nnBatchNorm2dn4n187')
    MobileNetn2nSequentialnmodelnnSequentialn8nnReLUn5n188 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn8nnBatchNorm2dn4n187, name = 'MobileNetn2nSequentialnmodelnnSequentialn8nnReLUn5n188')
    MobileNetn2nSequentialnmodelnnSequentialn9nnConv2dn0n189_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn8nnReLUn5n188, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn9nnConv2dn0n189 = convolution(MobileNetn2nSequentialnmodelnnSequentialn9nnConv2dn0n189_pad, group=512, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn9nnConv2dn0n189')
    MobileNetn2nSequentialnmodelnnSequentialn9nnBatchNorm2dn1n190 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn9nnConv2dn0n189, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn9nnBatchNorm2dn1n190')
    MobileNetn2nSequentialnmodelnnSequentialn9nnReLUn2n191 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn9nnBatchNorm2dn1n190, name = 'MobileNetn2nSequentialnmodelnnSequentialn9nnReLUn2n191')
    MobileNetn2nSequentialnmodelnnSequentialn9nnConv2dn3n192 = convolution(MobileNetn2nSequentialnmodelnnSequentialn9nnReLUn2n191, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn9nnConv2dn3n192')
    MobileNetn2nSequentialnmodelnnSequentialn9nnBatchNorm2dn4n193 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn9nnConv2dn3n192, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn9nnBatchNorm2dn4n193')
    MobileNetn2nSequentialnmodelnnSequentialn9nnReLUn5n194 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn9nnBatchNorm2dn4n193, name = 'MobileNetn2nSequentialnmodelnnSequentialn9nnReLUn5n194')
    MobileNetn2nSequentialnmodelnnSequentialn10nnConv2dn0n195_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn9nnReLUn5n194, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn10nnConv2dn0n195 = convolution(MobileNetn2nSequentialnmodelnnSequentialn10nnConv2dn0n195_pad, group=512, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn10nnConv2dn0n195')
    MobileNetn2nSequentialnmodelnnSequentialn10nnBatchNorm2dn1n196 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn10nnConv2dn0n195, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn10nnBatchNorm2dn1n196')
    MobileNetn2nSequentialnmodelnnSequentialn10nnReLUn2n197 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn10nnBatchNorm2dn1n196, name = 'MobileNetn2nSequentialnmodelnnSequentialn10nnReLUn2n197')
    MobileNetn2nSequentialnmodelnnSequentialn10nnConv2dn3n198 = convolution(MobileNetn2nSequentialnmodelnnSequentialn10nnReLUn2n197, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn10nnConv2dn3n198')
    MobileNetn2nSequentialnmodelnnSequentialn10nnBatchNorm2dn4n199 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn10nnConv2dn3n198, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn10nnBatchNorm2dn4n199')
    MobileNetn2nSequentialnmodelnnSequentialn10nnReLUn5n200 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn10nnBatchNorm2dn4n199, name = 'MobileNetn2nSequentialnmodelnnSequentialn10nnReLUn5n200')
    MobileNetn2nSequentialnmodelnnSequentialn11nnConv2dn0n201_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn10nnReLUn5n200, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn11nnConv2dn0n201 = convolution(MobileNetn2nSequentialnmodelnnSequentialn11nnConv2dn0n201_pad, group=512, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn11nnConv2dn0n201')
    MobileNetn2nSequentialnmodelnnSequentialn11nnBatchNorm2dn1n202 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn11nnConv2dn0n201, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn11nnBatchNorm2dn1n202')
    MobileNetn2nSequentialnmodelnnSequentialn11nnReLUn2n203 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn11nnBatchNorm2dn1n202, name = 'MobileNetn2nSequentialnmodelnnSequentialn11nnReLUn2n203')
    MobileNetn2nSequentialnmodelnnSequentialn11nnConv2dn3n204 = convolution(MobileNetn2nSequentialnmodelnnSequentialn11nnReLUn2n203, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn11nnConv2dn3n204')
    MobileNetn2nSequentialnmodelnnSequentialn11nnBatchNorm2dn4n205 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn11nnConv2dn3n204, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn11nnBatchNorm2dn4n205')
    MobileNetn2nSequentialnmodelnnSequentialn11nnReLUn5n206 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn11nnBatchNorm2dn4n205, name = 'MobileNetn2nSequentialnmodelnnSequentialn11nnReLUn5n206')
    MobileNetn2nSequentialnmodelnnSequentialn12nnConv2dn0n207_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn11nnReLUn5n206, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn12nnConv2dn0n207 = convolution(MobileNetn2nSequentialnmodelnnSequentialn12nnConv2dn0n207_pad, group=512, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn12nnConv2dn0n207')
    MobileNetn2nSequentialnmodelnnSequentialn12nnBatchNorm2dn1n208 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn12nnConv2dn0n207, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn12nnBatchNorm2dn1n208')
    MobileNetn2nSequentialnmodelnnSequentialn12nnReLUn2n209 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn12nnBatchNorm2dn1n208, name = 'MobileNetn2nSequentialnmodelnnSequentialn12nnReLUn2n209')
    MobileNetn2nSequentialnmodelnnSequentialn12nnConv2dn3n210 = convolution(MobileNetn2nSequentialnmodelnnSequentialn12nnReLUn2n209, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn12nnConv2dn3n210')
    MobileNetn2nSequentialnmodelnnSequentialn12nnBatchNorm2dn4n211 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn12nnConv2dn3n210, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn12nnBatchNorm2dn4n211')
    MobileNetn2nSequentialnmodelnnSequentialn12nnReLUn5n212 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn12nnBatchNorm2dn4n211, name = 'MobileNetn2nSequentialnmodelnnSequentialn12nnReLUn5n212')
    MobileNetn2nSequentialnmodelnnSequentialn13nnConv2dn0n213_pad = tf.pad(MobileNetn2nSequentialnmodelnnSequentialn12nnReLUn5n212, paddings = [[0, 0], [1, 1], [1, 1], [0, 0]])
    MobileNetn2nSequentialnmodelnnSequentialn13nnConv2dn0n213 = convolution(MobileNetn2nSequentialnmodelnnSequentialn13nnConv2dn0n213_pad, group=1024, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn13nnConv2dn0n213')
    MobileNetn2nSequentialnmodelnnSequentialn13nnBatchNorm2dn1n214 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn13nnConv2dn0n213, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn13nnBatchNorm2dn1n214')
    MobileNetn2nSequentialnmodelnnSequentialn13nnReLUn2n215 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn13nnBatchNorm2dn1n214, name = 'MobileNetn2nSequentialnmodelnnSequentialn13nnReLUn2n215')
    MobileNetn2nSequentialnmodelnnSequentialn13nnConv2dn3n216 = convolution(MobileNetn2nSequentialnmodelnnSequentialn13nnReLUn2n215, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nSequentialnmodelnnSequentialn13nnConv2dn3n216')
    MobileNetn2nSequentialnmodelnnSequentialn13nnBatchNorm2dn4n217 = batch_normalization(MobileNetn2nSequentialnmodelnnSequentialn13nnConv2dn3n216, variance_epsilon=9.999999747378752e-06, name='MobileNetn2nSequentialnmodelnnSequentialn13nnBatchNorm2dn4n217')
    MobileNetn2nSequentialnmodelnnSequentialn13nnReLUn5n218 = tf.nn.relu(MobileNetn2nSequentialnmodelnnSequentialn13nnBatchNorm2dn4n217, name = 'MobileNetn2nSequentialnmodelnnSequentialn13nnReLUn5n218')
    MobileNetn2nConv2dnfc2n219 = convolution(MobileNetn2nSequentialnmodelnnSequentialn13nnReLUn5n218, group=1, strides=[1, 1], padding='VALID', name='MobileNetn2nConv2dnfc2n219')
    return input, MobileNetn2nConv2dnfc2n219


def batch_normalization(input, name, **kwargs):
    mean = tf.Variable(__weights_dict[name]['mean'], name = name + "_mean", trainable = is_train)
    variance = tf.Variable(__weights_dict[name]['var'], name = name + "_var", trainable = is_train)
    offset = tf.Variable(__weights_dict[name]['bias'], name = name + "_bias", trainable = is_train) if 'bias' in __weights_dict[name] else None
    scale = tf.Variable(__weights_dict[name]['scale'], name = name + "_scale", trainable = is_train) if 'scale' in __weights_dict[name] else None
    return tf.nn.batch_normalization(input, mean, variance, offset, scale, name = name, **kwargs)


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer
