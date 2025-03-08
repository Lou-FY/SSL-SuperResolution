import torch
import imageio
import numpy as np
import os
import datetime
import pickle
import skimage.color as sc

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Logger:
    def __init__(self, args):
        self.args = args
        self.psnr_log = torch.Tensor()
        self.loss_log = torch.Tensor()
        #如果load是空，说明是新实验，如果save保存路径是空就用时间戳作为save保存路径，实验目录Logger.dir就是experiment和save保存路径的组合。如果新建的这个save存在了，就加个“archive”
        #如果load不是空，实验目录Logger.dir就是experiment和load加载路径的组合，这个路径不存在，把load置为“.”，重新新建一个实验目录。正常的话，把loss和psnr的log加载进来，打印继续训练
        if args.load == '.':#如果`args.load` 为 `.`，说明是新实验，则新建生成个实验目录，有save变量用save，没有就用时间戳，重复了就加上时间戳备份
            if args.save == '.':# 如果 `args.save` 也为 `.`，则自动生成一个时间戳
                args.save = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
            self.dir = args.experiment_dir + args.save# 实验数据存放目录
            if os.path.exists(self.dir) and not args.test_only:
                new_dir = self.dir + '_archived_' + datetime.datetime.now().strftime('%Y%m%d_%H:%M')
                # os.rename(self.dir, new_dir)
        else:#有arg.load参数，说明是继续试验，所以加载之前的实验目录
            self.dir = args.experiment_dir + args.load
            if not os.path.exists(self.dir):
                args.load = '.'#给的目录不存在，就新建一个实验目录
            else:
                self.loss_log = torch.load(self.dir + '/loss_log.pt')[:, -1]
                self.psnr_log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.psnr_log)))

        if not os.path.exists(self.dir):#load不为空但是目录不存在的情况
            os.makedirs(self.dir)

        if not os.path.exists(self.dir + '/model'):#保证dir下的model目录存在
            os.makedirs(self.dir + '/model')

        if not os.path.exists(self.dir + '/result/' + self.args.data_test):#保证dir下的result目录存在
            print("Creating dir for saving images...", self.dir + '/result/' + self.args.data_test)
            os.makedirs(self.dir + '/result/' + self.args.data_test)

        print('Save Path : {}'.format(self.dir))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type) #存入log文件
        with open(self.dir + '/config.txt', open_type) as f: #存入config文件
            f.write('From epoch {}...'.format(len(self.psnr_log)) + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def write_log(self, log):#写日志信息
        print(log)
        self.log_file.write(log + '\n')

    def save(self, trainer, epoch, is_best):#没有用到
        trainer.model.save(self.dir, epoch, is_best)
        torch.save(self.psnr_log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))#保留优化器Adam的参数
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)
        self.plot_psnr_log(epoch)

    def save_images(self, filename, save_list, epoch):
        f = filename.split('.')
        dirname = '{}/result/{}/{}'.format(self.dir, self.args.data_test, f[0])
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        filename = '{}/{}'.format(dirname, f[1])
        if self.args.task == 'PretrainKernel':
            postfix = ['est_kernel']
        elif self.args.task == 'FlowVideoSR':
            # postfix = ['gt', 'lr', 'sr', 'lr_cycle', 'est_kernel']
            postfix = ['gt', 'lr', 'sr']
        else:
            raise NotImplementedError('Task [{:s}] is not found'.format(self.args.task))
        for img, post in zip(save_list, postfix):
            img = img[0].data
            img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            if img.shape[2] == 1:
                img = img.squeeze(axis=2)
            elif img.shape[2] == 3 and self.args.n_colors == 1:
                img = sc.ycbcr2rgb(img.astype('float')).clip(0, 1)
                img = (255 * img).round().astype('uint8')
            imageio.imwrite('{}_{}.png'.format(filename, post), img)

    def start_log(self, train=True):
        if train:
            self.loss_log = torch.cat((self.loss_log, torch.zeros(1)))
        else:
            self.psnr_log = torch.cat((self.psnr_log, torch.zeros(1)))

    def report_log(self, item, train=True):
        if train:
            self.loss_log[-1] += item
        else:
            self.psnr_log[-1] += item

    def end_log(self, n_div, train=True):
        if train:
            self.loss_log[-1].div_(n_div)
        else:
            self.psnr_log[-1].div_(n_div)

    def plot_loss_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('Loss Graph')
        plt.plot(axis, self.loss_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'loss.pdf'))
        plt.close(fig)

    def plot_psnr_log(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('PSNR Graph')
        plt.plot(axis, self.psnr_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'psnr.pdf'))
        plt.close(fig)

    def plot_log(self, data_list, filename, title):
        epoch = len(data_list)
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('{} Graph'.format(title))
        plt.plot(axis, np.array(data_list))
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, filename))
        plt.close(fig)

    def done(self):
        self.log_file.close()
