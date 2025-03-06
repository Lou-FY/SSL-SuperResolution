import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model.kernel import KernelNet

class Trainer:
    def __init__(self, args, loader, my_loss, ckp):
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        # self.device = torch.device('cpu')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.kernel_net=KernelNet(ksize=13).to(self.device)
        self.loss = my_loss
        self.ckp = ckp

        # if args.load != '.':
        #     self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
        #     for _ in range(len(ckp.psnr_log)):
        #         self.scheduler.step()

    # def make_optimizer(self):
    #     kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
    #     return optim.Adam(self.model.parameters(), **kwargs)

    def train(self):
        pass

    def test(self):
        pass

    def terminate(self):
        if self.args.test_only:
            self.test()     # 如果是测试模式，直接运行 test() 并终止
            return True
        else:
            epoch = self.scheduler_model.last_epoch + 1# 获取当前 epoch 轮数
            return epoch >= self.args.epochs# 如果达到最大训练轮数，则终止
