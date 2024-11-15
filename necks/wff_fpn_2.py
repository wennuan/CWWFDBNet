# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
import math
import numpy as np
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

from ppocr.modeling.backbones.det_mobilenet_v3 import SEModule



class Conv_BN_ReLU(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias_attr=False)
        self.bn = nn.BatchNorm2D(out_planes)
        self.relu = nn.ReLU()



        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.weight.shape[0] * m.weight.shape[1] * m.weight.shape[2]
                v = np.random.normal(loc=0., scale=np.sqrt(2. / n), size=m.weight.shape).astype('float32')
                m.weight.set_value(v)
            elif isinstance(m, nn.BatchNorm):
                m.weight.set_value(np.ones(m.weight.shape).astype('float32'))
                m.bias.set_value(np.zeros(m.bias.shape).astype('float32'))

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SeparableConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2D(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias_attr=False)
        self.pointwise_conv = nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x



class WFFFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, use_asf=False,epsilon=1e-4, **kwargs):
        super(WFFFPN, self).__init__()
        self.out_channels = out_channels
        self.epsilon = epsilon
        self.use_asf = use_asf
        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

        planes = self.out_channels
        self.dwconv3_1 = nn.Conv2D(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   )
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv2_1 = nn.Conv2D(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   )
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)

        self.dwconv1_1 = nn.Conv2D(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=planes,
                                   )
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)
        self.dwconv2_2 = nn.Conv2D(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   )
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv3_2 = nn.Conv2D(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   )
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)

        self.dwconv4_2 = nn.Conv2D(planes,
                                   planes,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   groups=planes,
                                   )
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)

        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

        self.psa = PSAModule(self.out_channels, self.out_channels)

        self.gla_5=GLA(in_channels=in_channels[3], out_channels=self.out_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2D(2)
        self.w1 = self.create_weights(2)
        self.w2 = self.create_weights(2)
        self.w3 = self.create_weights(3)
        self.w4 = self.create_weights(3)
        self.relu = nn.ReLU()
        self.sep_conv = SeparableConv2d(planes, planes)
        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=[H, W], mode='bilinear') + y

    def create_weights(self, num_inputs):
        return paddle.create_parameter([num_inputs], dtype='float32',
                                       default_initializer=paddle.nn.initializer.Constant(1.0))
    def _weighted_sum(self, weights, *inputs):
        weights = self.relu(weights)  # Ensure weights are positive
        weight_sum = paddle.sum(weights) + self.epsilon  # Avoid division by zero
        weighted_inputs = [w * inp for w, inp in zip(weights, inputs)]
        return sum(weighted_inputs) / weight_sum

    def forward(self, x):
        c1, c2, c3, c4 = x

        f4 = self.gla_5(c4)
        f3 = self.in4_conv(c3)
        f2 = self.in3_conv(c2)
        f1 = self.in2_conv(c1)

        # f4_ = self.psa(f4)
        f4_ = f4
        f3_ = self._weighted_sum(self.w1, f3, self.upsample(f4_))
        f2_ = self._weighted_sum(self.w2, f2, self.upsample(f3_))
        f1_ = self._weighted_sum(self.w3, f1, self.upsample(f2_))


        f1_out = self.sep_conv(f1_)
        f2_out = self._weighted_sum(self.w4, f2_, self.downsample(f1_out))
        f2_out = self.sep_conv(f2_out)


        f3_out = self._weighted_sum(self.w4, f3_, self.downsample(f2_out))
        f3_out = self.sep_conv(f3_out)

        f4_out = self._weighted_sum(self.w4, f4_, self.downsample(f3_out))
        f4_out = self.sep_conv(f4_out)



        f1 = f1+f1_out
        f2 = self._weighted_sum(self.w4,f2_out,f2)
        f3 = self._weighted_sum(self.w4,f3_out,f3)
        f4 = self._weighted_sum(self.w4,f4_out,f4)

        p5 = self.p5_conv(f4)
        p4 = self.p4_conv(f3)
        p3 = self.p3_conv(f2)
        p2 = self.p2_conv(f1)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)



        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])

        return fuse







class ASFBlock(nn.Layer):
    """
    This code is refered from:
        https://github.com/MhLiao/DB/blob/master/decoders/feature_attention.py
    """

    def __init__(self, in_channels, inter_channels, out_features_num=4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super(ASFBlock, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2D(in_channels, inter_channels, 3, padding=1)

        self.spatial_scale = nn.Sequential(
            #Nx1xHxW
            nn.Conv2D(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                bias_attr=False,
                padding=1,
                weight_attr=ParamAttr(initializer=weight_attr)),
            # nn.BatchNorm2D(1),   #加
            nn.ReLU(),
            nn.Conv2D(
                in_channels=1,
                out_channels=1,
                kernel_size=1,
                bias_attr=False,
                weight_attr=ParamAttr(initializer=weight_attr)),
            # nn.BatchNorm2D(1),
            nn.Sigmoid())

        self.channel_scale = nn.Sequential(
            nn.Conv2D(
                in_channels=inter_channels,
                out_channels=out_features_num,
                kernel_size=1,
                bias_attr=False,
                # groups=inter_channels,         #分离卷积
                weight_attr=ParamAttr(initializer=weight_attr)),
            nn.Sigmoid())

    def forward(self, fuse_features, features_list):
        fuse_features = self.conv(fuse_features)
        spatial_x = paddle.mean(fuse_features, axis=1, keepdim=True)
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        attention_scores = self.channel_scale(attention_scores)
        assert len(features_list) == self.out_features_num

        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[:, i:i + 1] * features_list[i])
        return paddle.concat(out_list, axis=1)


# PSA模块
def conv111(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias_attr=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)


class SEWeightModule(nn.Layer):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


class PSAModule(nn.Layer):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv111(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv111(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv111(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv111(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # stage 1
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = paddle.concat((x1, x2, x3, x4), axis=1)
        feats = feats.reshape([batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3]])

        # stage 2
        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = paddle.concat((x1_se, x2_se, x3_se, x4_se), axis=1)
        attention_vectors = x_se.reshape([batch_size, 4, self.split_channel, 1, 1])
        attention_vectors = self.softmax(attention_vectors)  # stage 3

        # stage 4
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = paddle.concat((x_se_weight_fp, out), axis=1)

        return out


class CBAMModule(nn.Layer):
    def __init__(self, feature_channel, feature_height, feature_width):
        super(CBAMModule, self).__init__()
        self.c_maxpool = nn.MaxPool2D((feature_height, feature_width), 1)
        self.c_avgpool = nn.AvgPool2D((feature_height, feature_width), 1)
        self.s_maxpool = nn.MaxPool2D(1, 1)
        self.s_avgpool = nn.AvgPool2D(1, 1)
        self.s_conv = nn.Conv2D(int(feature_channel * 2), 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.shared_MLP = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(feature_channel, int(feature_channel / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_channel / 2), feature_channel),
            nn.ReLU()
        )

    def Channel_Attention(self, x):
        b, c, h, w = x.shape

        x_m = self.c_maxpool(x)
        x_a = self.c_avgpool(x)

        mlp_m = self.shared_MLP(x_m)
        mlp_a = self.shared_MLP(x_a)

        mlp_m = paddle.reshape(mlp_m, [b, c, 1, 1])
        mlp_a = paddle.reshape(mlp_a, [b, c, 1, 1])

        c_c = paddle.add(mlp_a, mlp_m)
        Mc = self.sigmoid(c_c)
        return Mc

    def Spatial_Attention(self, x):
        x_m = self.s_maxpool(x)
        x_a = self.s_avgpool(x)

        x_concat = paddle.concat([x_m, x_a], axis=1)
        x_x = self.s_conv(x_concat)
        Ms = self.sigmoid(x_x)

        return Ms

    def forward(self, x):
        Mc = self.Channel_Attention(x)
        F1 = Mc * x

        Ms = self.Spatial_Attention(F1)
        refined_feature = Ms * F1

        return refined_feature





class GLA(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(GLA, self).__init__()



        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias_attr=False),
            nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )


        self.local_branch = nn.Sequential(
            nn.Conv2D(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias_attr=False),
            nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )



    def forward(self, x):
        global_features = self.global_branch(x)


        local_features = self.local_branch(x)

        fused_features = global_features + local_features

        return fused_features