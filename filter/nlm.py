import re
import numpy as np
import  os
from scipy.io import loadmat
import scipy.io as sio
import time

def make_kernel(f):
    kernel = np.zeros((1, 2*f+1), np.float32)
    #print(kernel.shape)
    for d in range(1, f+1):
        for i in range(-d,d):
            kernel[0][f-i] += (1.0/((2*d+1)**2))
    return kernel/f

def nlm_1D_filter(src, f, t, h):
    m = len(src[0])
    #m = len(src) #测试用
    out = np.zeros((1, m), np.float32)
    # print(out)
    # memory for output

    # Replicate the boundaries of the input
    src_padding = np.pad(src[0], (0, f), mode='symmetric').astype(np.float32)
    #src_padding = np.pad(src, (0, f), mode='symmetric').astype(np.float32) #测试用
    # used kernel
    kernel = make_kernel(f)
    kernel = kernel / kernel.sum()

    for i in range(1, m):

        i1 = i + f

        if (i1 + f + 1) > src_padding.shape[0]:
            continue

        W1 = src_padding[i1 - f:i1 + f + 1]  # 领域窗口W1
        w_max = 0
        aver = 0
        weight_sum = 0
        rmin = max(i1 - t, f + 1);
        rmax = min(i1 + t, m + f);

        # 搜索窗口

        for r in range(rmin, rmax):
            if r == i1:
                continue
            elif (r + f + 1) > src_padding.shape[0]:
                continue
            else:
                W2 = src_padding[r - f:r + f + 1]  # 搜索区域内的相似窗口

                w0 = W2 - W1
                wc = np.dot(w0, w0)
                Dist2 = np.dot(kernel, wc).sum()
                w = np.exp(-Dist2 / h ** 2)
                if w > w_max:
                    w_max = w
                weight_sum = weight_sum + w
                aver = aver + w * src_padding[r]

        aver = aver + w_max * src_padding[i1]  # 自身领域取最大的权重
        weight_sum = weight_sum + w_max
        out[0, i] = aver / weight_sum

    return out


# 批量进行NML
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

            # 对数据降噪_非局部均值降噪
            nonoise_DE = nlm_1D_filter(org_DE.T, 10, 2, 5)
            nonoise_FE = nlm_1D_filter(org_DE.T, 10, 2, 5)
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
    print('DONE THE PROGRAM')
    end = time.time()
    print('Running time: %s Seconds' % (end - start))


path1 = r'data\12k Drive End Bearing Fault Data'  ##原始数据地址
path2 = r'nml_denoise_data'
nonoise(path1, path2)

x=show_plot()