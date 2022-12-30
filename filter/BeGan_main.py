import torch
import os
import utils
import numpy as np
from BeGan_train import Trainer
from config import get_config
import torch.utils.data as Data
from utils import prepare_dirs_and_logger, save_config

def main(config):
    prepare_dirs_and_logger(config)
    torch.manual_seed(config.random_seed)
    #导入全部数据集, data_loader生成数据与数据标签迭代器
    x_train, y_train = utils.loadDataSet(config.dataset_path, 0)
    tensor_train_x = torch.as_tensor(torch.from_numpy(x_train), dtype=torch.float32).to('cuda')
    tensor_train_y = torch.as_tensor(torch.from_numpy(y_train), dtype=torch.long).to('cuda')
    train_dataset = Data.TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
                                        drop_last=True)

    # #导入部分数据集，data_loader
    # file_name = data_path
    # x_train= utils.loadDataSet_C(file_name, ind=0)
    # tensor_train = torch.as_tensor(torch.from_numpy(x_train), dtype=torch.float32)
    # data_dataset = Data.TensorDataset(tensor_train)
    # data_loader = Data.DataLoader(dataset=data_dataset, batch_size=batch_size, shuffle=do_shuffle, num_workers=0, drop_last=True)
    # data_loader.shape = [1, 2048]
    # trainer = Trainer(config, data_loader)

    if config.is_train:
        save_config(config)
        trainer = Trainer(config, train_loader)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer = Trainer(config, train_loader)
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)