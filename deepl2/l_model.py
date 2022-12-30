from keras.layers import  LSTM
from tensorflow import keras
from tensorflow.keras import layers

def lstm_create():
    loss = 'sparse_categorical_crossentropy'
    metrics = ['acc']
    inputs = keras.Input(shape=(1000, 2))
    
    x=LSTM(64)
    #x = layers.Conv1D(64, 3, activation='relu')(inputs)
    #x = layers.MaxPooling1D(16)(x)
    # 全局平均池化GAP层
    #x = layers.GlobalAveragePooling1D()(x)
    # 几个密集分类层
    x = layers.Dense(32, activation='relu')(x)
    # 退出层
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop',
                  loss=loss,
                  metrics=metrics)
    print("实例化模型成功，参数如下：")
    print(model.summary())
    return model