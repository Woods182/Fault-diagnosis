import torch
import numpy as np
from torch import nn
import config
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


class BaseModel(nn.Module):
    def forward(self, x):
        return self.main(x)

# 生成器:decoder
class GeneratorCNN(BaseModel):
    def __init__(self, input_num,n_classes, initial_conv_dim, output_num, repeat_num, hidden_num, num_gpu):
        '''
        :param input_num: 输入向量维度（h=z_num）
        :param initial_conv_dim: 初始卷积维度 [8,n]
        :param output_num: 输出向量 channel(1)
        :param repeat_num: CNN层数 (int(np.log2(data_length)) - 2)
        :param hidden_num: 隐含层个数 (hidden_num)
        '''
        super(GeneratorCNN, self).__init__()
        self.initial_conv_dim = initial_conv_dim
        self.preprocess = nn.Sequential(
            nn.Linear(input_num, np.prod(self.initial_conv_dim)),
            nn.BatchNorm1d(np.prod(self.initial_conv_dim)),
            nn.ReLU())

        self.labelembedding = nn.Embedding(n_classes , input_num)
        config.num_gpu = num_gpu
        layers = []
        for idx in range(repeat_num):
            layers.append(nn.Conv1d(hidden_num, hidden_num, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv1d(hidden_num, hidden_num, 3, 1, 1))
            layers.append(nn.ELU(True))
            #进行上采样
            if idx < repeat_num - 1:
                layers.append(nn.Upsample(scale_factor=2))

        layers.append(nn.Conv1d(hidden_num, output_num, 3, 1, 1))
        # layers.append(nn.Tanh())
        layers.append(nn.ELU(True))

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x, label):
        x = torch.mul(self.labelembedding(label), x)
        fc_out = self.preprocess(x).view([-1] + self.initial_conv_dim)
        return self.conv(fc_out)

# 判别器：encoder + decoder
class DiscriminatorCNN(BaseModel):
    def __init__(self, input_channel, n_classes, z_num, repeat_num, hidden_num, num_gpu):
        '''
        :param input_channel: 输入的通道数 (1)
        :param z_num: 编码器最后的输出维度 (h=z_num)
        :param repeat_num: CNN层数(int(np.log2(data_length)) - 2)
        :param hidden_num: 隐含层个数 (hidden_num)
        '''
        super(DiscriminatorCNN, self).__init__()
        self.num_gpu = num_gpu
        # Encoder
        layers = []
        layers.append(nn.Conv1d(input_channel, hidden_num, 3, 1, 1))
        layers.append(nn.ELU(True))

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            layers.append(nn.Conv1d(prev_channel_num, channel_num, 3, 1, 1))
            layers.append(nn.ELU(True))
            if idx < repeat_num - 1:
                layers.append(nn.Conv1d(channel_num, channel_num, 3, 2, 1))
                # layers.append(nn.MaxPool2d(2))
                # layers.append(nn.MaxPool2d(1, 2))
            else:
                layers.append(nn.Conv1d(channel_num, channel_num, 3, 1, 1))

            layers.append(nn.ELU(True))
            prev_channel_num = channel_num

        self.conv1_output_dim = [channel_num, 8]
        self.conv1 = torch.nn.Sequential(*layers)
        self.fc1 = nn.Linear(8*channel_num, z_num)


        # Decoder
        self.conv2_input_dim = [hidden_num, 8]
        self.fc2 = nn.Linear(z_num, np.prod(self.conv2_input_dim))

        layers = []
        for idx in range(repeat_num):
            layers.append(nn.Conv1d(hidden_num, hidden_num, 3, 1, 1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv1d(hidden_num, hidden_num, 3, 1, 1))
            layers.append(nn.ELU(True))

            if idx < repeat_num - 1:
                layers.append(nn.Upsample(scale_factor=2))

        layers.append(nn.Conv1d(hidden_num, input_channel, 3, 1, 1))
        # layers.append(nn.Tanh())
        layers.append(nn.ELU(True))

        self.conv2 = torch.nn.Sequential(*layers)
        self.label = nn.Linear(input_channel * 2048, n_classes + 1)

    def forward(self, x):
        conv1_out = self.conv1(x).view(-1, np.prod(self.conv1_output_dim))
        fc1_out = self.fc1(conv1_out)

        fc2_out = self.fc2(fc1_out).view([-1] + self.conv2_input_dim)
        conv2_out = self.conv2(fc2_out)
        emb = conv2_out.view(conv2_out.size(0), -1)
        label = self.label(emb)
        return conv2_out, label

class _Loss(nn.Module):

    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        # this won't still solve the problem
        # which means gradient will not flow through target
        # _assert_no_grad(target)
        backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(self.size_average)(input, target)

class L1Loss(_Loss):
    r"""Creates a criterion that measures the mean absolute value of the
    element-wise difference between input `x` and target `y`:

    :math:`{loss}(x, y)  = 1/n \sum |x_i - y_i|`

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the constructor argument `sizeAverage=False`
    """
    #pass
