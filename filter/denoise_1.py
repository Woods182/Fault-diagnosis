import re
import pywt
import os
from scipy.io import loadmat
import scipy.io as sio
import time

# 小波变换法
def wavelet_means(or_data):
    index = []
    data = []
    for i in range(len(or_data) - 1):
        X = float(i)
        Y = float(or_data[i])
        index.append(X)
        data.append(Y)

    # Create wavelet object and define parameters
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)  # Compute the maximum useful level of decomposition
    threshold = 0.04  # Threshold for filtering
    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)  # 将信号进行小波分解
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波
    datarec = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
    return datarec


def nonoise(data_path, save_path):
    start = time.time()
    file_1 = os.listdir(data_path)
    for k in file_1:
        d_path = os.path.join(data_path, k)
        filenames = os.listdir(d_path)
        for i in filenames:
            data = loadmat(os.path.join(d_path, i))
            names_number = re.findall(r"\d+", i)
            if int(names_number[0]) < 100:
                names_number = '0' + str(names_number[0])
            else:
                names_number = str(names_number[0])
                # 原始数据
            org_DE = data['X' + names_number + '_DE_time']
            org_FE = data['X' + names_number + '_FE_time']
            # org_BA = data['X' + names_number + '_BA_time']

            # 对数据降噪_小波变换法
            nonoise_DE = wavelet_means(org_DE)
            nonoise_FE = wavelet_means(org_FE)
            # nonoise_BA = wavelet_means(org_BA)
            # 降噪后的数据nonoise_DE、nonoise_FE、nonoise

            # 将数据存储为2维.mat数据
            sio.savemat(os.path.join(save_path, k, i),
                        {'X' + names_number + '_DE_time': nonoise_DE, 'X' + names_number + '_FE_time': nonoise_FE},
                        format='5',
                        long_field_names=False,
                        do_compression=False,
                        oned_as='column')
            print("De-noise is done", str(os.path.join(d_path, i)))
            end = time.time()
            print('Running time: %s Seconds' % (end - start))


path1 = r'data\12k Drive End Bearing Fault Data'  ##原始数据地址
path2 = r'nonoise_data'
nonoise(path1, path2)




import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import tensorflow as tf
