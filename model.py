import tensorflow as tf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
def get_model(model):
    if model == 'SRCNN':
        return SRCNN()
    elif model == 'VDSR':
        return VDSR()
    elif model == 'VDSR-ref':
        return VDSR_ref()
    else:
        raise Exception('Unknown model %s' %model)

class BaseModel(object):
    def __init__(self):
        pass
    
    def inference(self):
        raise Exception('Unimplemented method')

    def loss(self, labels, outputs):
        with tf.name_scope('MSELoss'):
            return tf.losses.mean_squared_error(labels, outputs)
        
    def optimize(self, loss, global_step, initial_lr, decay_step, decay_rate):
        with tf.name_scope('Optimize'):
            lr = tf.train.exponential_decay(initial_lr,
                                            global_step,
                                            decay_step,
                                            decay_rate,
                                            staircase=True)
            tf.summary.scalar('learning_rate', lr)
            train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
            return train_step


class SRCNN(BaseModel):
    def inference(self, input_tensor):
        conv1 = tf.layers.conv2d(
                inputs=input_tensor,
                filters=64,
                kernel_size=[9, 9],
                padding='same',
                activation=tf.nn.relu,
                name='conv1')

        conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=32,
                kernel_size=[1, 1],
                padding='same',
                activation=tf.nn.relu,
                name='conv1')

        outputs = tf.layers.conv2d(
                inputs=conv2,
                filters=1,
                kernel_size=[5, 5],
                padding='same',
                name='conv3')
        return outputs

    
class VDSR1(BaseModel):
    def inference(self, input_tensor):
        conv_next = tf.layers.conv2d(inputs=input_tensor,
                                     filters=64,
                                     kernel_size=3,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     name='conv_first')
        for i in range(1,17):
            conv_next = tf.layers.conv2d(inputs=conv_next,
                                        filters=64,
                                        kernel_size=3,
                                        padding='same',
                                        activation=tf.nn.relu,
                                        name='conv_next_{}'.format(i))

        outputs = tf.layers.conv2d(inputs=conv_next,
                                   filters=1,
                                   kernel_size=3,
                                   padding='same',
                                   name='conv_last')
        return outputs


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.name = 'VRES'
        self.conv_first = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.conv_next = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv_last = nn.Conv2d(64, 1, 3, padding=1, bias=False)
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.relu = nn.ReLU(inplace=True)

        # xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        res = x
        out = self.relu(self.conv_first(x))
        out = self.residual_layer(out)
        out = self.conv_last(out)
        out = torch.add(out, res)
        return out


class VDSR_Naive(nn.Module):
    def __init__(self):
        super(VDSR_Naivem,self).__init__()
        self.conv_first = nn.Conv2d(1, 64, 3, padding=1, bias=False)

        self.relu = nn.RELU(inplace=True)
    def forward(self, x):
        res = x
        out = self.relu(self.conv_first(x))
        out = x 
        out = F.relu()
        pass

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class VDSR_ref(nn.Module):
    def __init__(self):
        super(VDSR_ref, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out


        






        
