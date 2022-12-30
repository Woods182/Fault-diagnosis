import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
# 查看GPU是否可用
print(tf.config.list_physical_devices('GPU'))

import n_model as md
# 模型参数
model_param = {
    "a_shape": 1000,
    "b_shape": 2,
    "label_count": 4,
    "num_b":5
}

data_shape=(model_param['a_shape'],model_param['b_shape'])
# 模型实例化
model = md.CNN_ResNet_model(model_param['label_count'] , model_param['num_b'] , data_shape=data_shape)
# 使用学习率进行训练
res_model = model.model_create(learning_rate = 1e-4)
# 模型网络结构
print("实例化模型成功，网络结构如下：")
print(res_model.summary())
# 设置模型log输出地址
log_dir = os.path.join("logs/ResNet")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)