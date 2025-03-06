from importlib import import_module  #特殊的导入类的方法
from torch.utils.data import DataLoader


class Data:
    def __init__(self, args):
        self.args = args
        
        

        # load training dataset
        if not self.args.test_only:
            self.data_train = args.data_train #如果是训练，那么就只加载训练数据集
            m_train = import_module('data.' + self.data_train.lower())
            trainset = getattr(m_train, self.data_train.upper())(self.args, name=self.data_train, train=True)
            self.loader_train = DataLoader(
                trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                pin_memory=not self.args.cpu,
                num_workers=0
            )
        else:
            self.data_test = args.data_test#如果是测试，那么就只加载测试数据集
            self.loader_train = None

        # load testing dataset
        m_test = import_module('data.' + self.data_test.lower())
        testset = getattr(m_test, self.data_test.upper())(self.args, name=self.data_test, train=False)
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not self.args.cpu,
            num_workers=0
        )
