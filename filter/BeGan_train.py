import time

import torch
import os
import numpy as np
import config
from glob import glob
from tqdm import trange
import torchsummary
from itertools import chain
from collections import deque
import torch
from torch import nn
import utils
import torch.nn.parallel
import torchvision.utils as vutils
from torch.autograd import Variable
from BeGan_model import *

import time
import logging


#初始化权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def next(loader):
    return loader.next()[0]

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

        self.n_classes = config.n_classes
        self.num_gpu = config.num_gpu
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.lambda_k = config.lambda_k
        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.data_length = config.data_length

        self.load_path = config.load_path
        self.log_path = config.log_path
        self.model_path = config.model_path
        self.tb_path = config.tb_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if  torch.cuda.is_available() else torch.LongTensor
        self.build_model()

        datefmt = '%Y-%m-%d  %H:%M:%S %a'
        t = time.strftime('%Y%m%d  %H:%M:%S', time.localtime(time.time()))
        myformatter = logging.Formatter(fmt="%(asctime)s - %(process)d -%(levelname)s - %(message)s", datefmt=datefmt)
        self.mylogger = logging.getLogger(__name__)
        for hdlr in self.mylogger.handlers:
            self.mylogger.removeHandler(hdlr)
        self.mylogger.setLevel(logging.INFO)
        log_path = self.log_path + t + '.txt'
        with open(log_path, 'w') as fp:
            file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(myformatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(myformatter)
        self.mylogger.addHandler(console_handler)
        self.mylogger.addHandler(file_handler)

        if self.num_gpu > 0:
            self.G.cuda()
            self.D.cuda()

        if self.load_path:
            self.load_model()
        # 记录日志
        self.use_tensorboard = config.use_tensorboard
    # 构建模型
    def build_model(self):
        channel = 1
        data_length = self.data_length
        repeat_num = int(np.log2(data_length)) - 2
        self.D = DiscriminatorCNN(
                channel, self.n_classes, self.z_num, repeat_num, self.conv_hidden_num, self.num_gpu)
        self.G = GeneratorCNN(
                self.z_num, self.n_classes, self.D.conv2_input_dim, channel, repeat_num, self.conv_hidden_num, self.num_gpu)

        self.G.apply(weights_init)
        self.D.apply(weights_init)
    # 模型训练
    def train(self):
        l1 = nn.L1Loss(size_average=True, reduce=True)
        criterion = nn.CrossEntropyLoss()
        z_D = Variable(self.FloatTensor(self.batch_size, self.z_num))
        z_G = Variable(self.FloatTensor(self.batch_size, self.z_num))
        with torch.no_grad():
            z_fixed = Variable(self.FloatTensor(self.batch_size, self.z_num).normal_(0, 1))

        if self.num_gpu > 0:
            l1.cuda()

            z_D = z_D.cuda()
            z_G = z_G.cuda()
            z_fixed = z_fixed.cuda()

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        #设置优化器
        def get_optimizer(lr):
            return optimizer(self.G.parameters(), lr=lr, betas=(self.beta1, self.beta2)), \
                   optimizer(self.D.parameters(), lr=lr, betas=(self.beta1, self.beta2))
        g_optim, d_optim = get_optimizer(self.lr)

        # 获取固定的真实数据
        data_loader = iter(self.data_loader)
        x_fixed = self._get_variable(next(data_loader).unsqueeze(1))
        k_t = 0
        prev_measure = 1
        measure_history = deque([0]* self.lr_update_step, self.lr_update_step)
        measure_list = []
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(self.tb_path)
        # 开始迭代
        for epoch in trange(self.start_step, self.max_step):
            for step, (x_real, label_real) in enumerate(self.data_loader):
                x_real = self._get_variable(self.FloatTensor(x_real)).unsqueeze(1)
                label_real = self._get_variable(self.LongTensor(label_real))
                label_fake = self._get_variable(self.LongTensor([10]).repeat(64,1).squeeze(1))
                # 梯度清零，计算损失函数
                self.D.zero_grad()
                self.G.zero_grad()
                z_D.data.normal_(0, 1)
                z_G.data.normal_(0, 1)

                #sample_z_D = self.G(z_D)
                sample_z_G = self.G(z_G, label_real)

                AE_x, AE_x_label = self.D(x_real)
                AE_G_d, AE_G_d_label= self.D(sample_z_G.detach())
                AE_G_g, AE_G_g_label= self.D(sample_z_G)

                d_loss_real = l1(AE_x, x_real)
                d_loss_fake = l1(AE_G_d, sample_z_G.detach())


                label_real_loss = criterion(AE_x_label, label_real)
                label_fake_loss = criterion(AE_G_g_label, label_real)
                label_real_or_false = criterion(AE_G_d_label, label_fake)

                d_loss = d_loss_real - k_t * d_loss_fake + 0.2 * label_real_loss + 0.2 * label_real_or_false
                g_loss = l1(sample_z_G, AE_G_g) + 0.2 * label_fake_loss # this won't still solve the problem

                d_loss.backward()
                g_loss.backward()
                # loss.backward()

                g_optim.step()
                d_optim.step()

                g_d_balance = (self.gamma * d_loss_real - d_loss_fake).data.item()

                # 保证 kt∈[0,1]
                k_t += self.lambda_k * g_d_balance
                k_t = max(min(1, k_t), 0)

                measure = d_loss_real.data.item() + abs(g_d_balance)
                measure_history.append(measure)

            #记录日志
            if epoch % self.log_step == 0:
                # torchsummary.summary(self.G.cuda(), (1, 128),())
                # torchsummary.summary(self.D.cuda(), (1, 2048),())
                # for name, parms in self.D.named_parameters():
                #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                #           ' -->grad_value:', parms.grad)
                # 记录参数
                if self.use_tensorboard:
                    self.mylogger.info("[{}/{}] Loss_D: {:.4f} L_x: {:.4f} Loss_G: {:.4f} "
                      "measure: {:.4f}, k_t: {:.4f}, generater_lr: {:.7f}, discriminater: {:.7f}". \
                      format(epoch, self.max_step, d_loss.data.item(), d_loss_real.data.item(),
                             g_loss.data.item(), measure, k_t, g_optim.state_dict()['param_groups'][0]['lr'], d_optim.state_dict()['param_groups'][0]['lr'] ))
                    writer.add_scalar('Train/measurement',measure, epoch)
                    writer.add_scalar('Train/g_lr',g_optim.state_dict()['param_groups'][0]['lr'], epoch)
                    writer.add_scalar('Train/d_lr',d_optim.state_dict()['param_groups'][0]['lr'], epoch)
                    writer.add_scalar('Train/Loss_D', d_loss.data.item(), epoch)
                    writer.add_scalar('Train/Loss_G', g_loss.data.item(), epoch)
                    measure_list.append(measure)

            # 存储模型
            if epoch % self.save_step == self.save_step - 1:
                self.save_model(epoch)

            # 更新学习率
            if epoch % self.lr_update_step == self.lr_update_step - 1:
                cur_measure = np.mean(measure_history)
                if cur_measure > prev_measure * 0.9999:
                    self.lr *= 0.5
                    g_optim, d_optim = get_optimizer(self.lr)
                prev_measure = cur_measure
    # 生成数据
    # def generate(self, inputs, path, idx=None):
    #     new_path = '{}/{}_G_0.txt'.format(path, idx)
    #     x = self.G(inputs)
    #     utils.save_data(x.data.cpu().numpy().tolist(), new_path)
    #     print(x.data.shape)
    #     self.mylogger.info('write data(num: %d, %d, %d) to file: %s'%(x.data.size(0), x.data.size(1), x.data.size(2), new_path))
    # #     print("[*] Samples saved: {}".format(new_path))
    #     return x

    # 自动编码器
    # def autoencode(self, inputs, path, idx=None, x_fake=None):
    #     x_path = '{}/{}_D_0.txt'.format(path, idx)
    #     x = self.D(inputs)
    #     torchsummary.summary(self.D.cuda(), (1,2048))
    #     #真实数据encoder后的数据
    #     utils.save_data(x.data.cpu().numpy().tolist(), x_path)
    #     print("[*] Samples saved: {}".format(x_path))
    #     #假数据encoder后的数据
    #     if x_fake is not None:
    #         x_fake_path = '{}/{}_D_fake.txt'.format(path, idx)
    #         x = self.D(x_fake)
    #         utils.save_data(x.data, x_fake_path)
    #         print("[*] Samples saved: {}".format(x_fake_path))

    def test(self):
        data_loader = iter(self.data_loader)
        x_fixed = self._get_variable(next(data_loader).unsqueeze(1))
        utils.save_data(x_fixed.data, '{}/x_fixed_test.txt'.format(self.model_dir))
        self.autoencode(x_fixed, self.model_dir, idx="test", x_fake=None)

    def save_model(self, step):
        print("[*] Save models to {}...".format(self.model_path))

        torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.model_path, step))
        torch.save(self.D.state_dict(), '{}/D_{}.pth'.format(self.model_path, step))

    def load_model(self):
        print("[*] Load models from {}...".format(self.load_path))

        paths = glob(os.path.join(self.load_path, 'G_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            return

        idxes = [int(os.path.basename(path.split('.')[1].split('_')[-1])) for path in paths]
        self.start_step = max(idxes)

        if self.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        G_filename = '{}/G_{}.pth'.format(self.load_path, self.start_step)
        self.G.load_state_dict(
            torch.load(G_filename, map_location=map_location))
        print("[*] G network loaded: {}".format(G_filename))

        D_filename = '{}/D_{}.pth'.format(self.load_path, self.start_step)
        self.D.load_state_dict(
            torch.load(D_filename, map_location=map_location))
        print("[*] D network loaded: {}".format(D_filename))

    def _get_variable(self, inputs):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
