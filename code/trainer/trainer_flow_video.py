import decimal
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm
from astropy.io import fits
import scipy.ndimage
from scipy.ndimage import zoom
import numpy as np
from utils import data_utils
from PIL import Image
import cv2
from trainer.trainer import Trainer
from model.kernel import KernelNet
import torchvision.utils as utils
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from math import log10
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
import model.srgan
import torchvision.transforms as transforms

input_transform = transforms.Compose([
   transforms.Grayscale(1), #这一句就是转为单通道灰度图像
   transforms.ToTensor(),
])
class Trainer_Flow_Video(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_Flow_Video, self).__init__(args, loader, my_loss, ckp)
        print("Using Trainer_Flow_Video")
        self.model = my_model
        self.l1_loss = torch.nn.L1Loss().to(self.device)
        self.cycle_psnr_log = []
        self.mid_loss_log = []
        self.cycle_loss_log = []
        self.kernel_net=KernelNet(ksize=13).to(self.device)
        self.optimizer_model = self.make_optimizer_model()  # 两个
        self.scheduler_model = self.make_scheduler_model()
        self.optimizer_kernel = self.make_optimizer_kernel()
        self.scheduler_kernel = self.make_scheduler_kernel()

        if args.load != '.':
            mid_logs = torch.load(os.path.join(ckp.dir, 'mid_logs.pt'))
            self.cycle_psnr_log = mid_logs['cycle_psnr_log']
            self.mid_loss_log = mid_logs['mid_loss_log']
            self.cycle_loss_log = mid_logs['cycle_loss_log']

    def make_optimizer_model(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        # optimizer_model = optim.Adam([{"params": self.model.get_model().fusion_conv.parameters()},
        #                         {"params": self.model.get_model().out_conv.parameters()},
        #                         {"params": self.model.get_model().upsample_layers.parameters()},#5.00e-5
        #                         {"params": self.model.get_model().pcd_align.parameters(),"lr": 1e-4},
        #                         {"params": self.model.get_model().conv_first.parameters()},
        #                         {"params": self.model.get_model().feature_extraction.parameters()},
        #                         {"params": self.model.get_model().lrelu.parameters()},
        #                         {"params": self.model.get_model().conv_l2_1.parameters()},
        #                         {"params": self.model.get_model().conv_l2_2.parameters()},
        #                         {"params": self.model.get_model().conv_l3_1.parameters()},
        #                         {"params": self.model.get_model().conv_l3_2.parameters()},
        #                         {"params": self.model.get_model().resblocks.parameters()}],**kwargs)
        optimizer_model = optim.Adam(self.model.parameters(),**kwargs)
        return optimizer_model
    def make_optimizer_kernel(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer_kernel=optim.Adam(self.kernel_net.parameters(),lr=1e-5,eps=1e-08, weight_decay=0)
        # optimizer_kernel = optim.Adam(self.kernel_net.parameters(),**kwargs)
        return optimizer_kernel

    def make_scheduler_model(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lr_scheduler.StepLR(self.optimizer_model, **kwargs)

    def make_scheduler_kernel(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lr_scheduler.StepLR(self.optimizer_kernel, **kwargs)
    def train(self):
        print("Now training")
        self.scheduler_model.step() #看一下是否需要改这玩意儿
        self.scheduler_kernel.step()
        self.loss.step()
        epoch = self.scheduler_kernel.last_epoch + 1
        lr = self.scheduler_model.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        self.kernel_net.train()
        self.ckp.start_log()
        mid_loss_sum = 0.
        l_self_sum=0.
        cycle_loss_sum = 0.
        #

        #     #备用分支
        # for batch, (input, _) in enumerate(self.loader_train):
        #     input = input.to(self.device)
        #     b,t,c,h,w=input.size()
        #     input_list = [input[:, i, :, :, :] for i in range(self.args.n_sequence)]
        #     input_mid = input_list[self.args.n_sequence // 2]
        #     kernel_list = [self.kernel_net(f) for f in input_list]
        #     est_kernel= kernel_list[self.args.n_sequence // 2]
        #     # est_kernel=self.kernel_net(input)
        #     # aux_lr_seq=torch.stack([self.blur_down(g,est_kernel, 4) for g in input_list],dim=1)
        #     aux_lr_seq = torch.stack([self.blur_down(g, k, self.args.scale) for g, k in zip(input_list, kernel_list)],dim=1)
        #     # output_dict, mid_loss = self.model({'x': aux_lr_seq})
        #     output_dict, mid_loss = self.model(aux_lr_seq)
        #     # output_dict, mid_loss = self.model({'x':  input})
        #     aux_sr = output_dict['recons']
        #     #主分支
        #     latent_hr, mid_loss = self.model(input)
        #     latent_hr=latent_hr['recons']
        #     degraded_lr=torch.cat([self.blur_down(latent_hr,est_kernel, 4)],dim=0)
        #
        #     self.optimizer_model.zero_grad()
        #     self.optimizer_kernel.zero_grad()
        #     #
        #     # 损失
        #     # kernel_Loss=((torch.norm(est_kernel.reshape(13,13)))**0.5).to(self.device)
        #     # kernel_Loss = (torch.sum(torch.sqrt(est_kernel))).to(self.device)
        #     # kernel_Loss = torch.max(est_kernel).to(self.device)
        #     # kernel_Loss = est_kernel[0, 0]% 2
        #     kernel_Loss= torch.sum(est_kernel ** 0.5).to(self.device)
        #     # kernel_Loss = torch.sum(torch.sqrt(est_kernel)).to(self.device)
        #     # kernel_Loss = torch.mean(est_kernel).to(self.device)
        #
        #     loss= self.loss(aux_sr, input_mid)
        #
        #     l_self= 0.
        #     l_self = l_self + self.l1_loss(degraded_lr,input_mid)
        #     l_self_sum = l_self_sum+ l_self.item()
        #     # l_self=self.l1_loss(degraded_lr,input_mid)
        #     loss=loss+l_self
        #     if mid_loss:  # mid loss is the loss during the model
        #         loss = loss + self.args.mid_loss_weight * mid_loss
        #         mid_loss_sum= mid_loss_sum + mid_loss.item()
        #     # for name, param in self.model.named_parameters():  # 打印想要的模型的各个卷积层的名字和参数
        #     #     print(name)
        #     #     print(param)
        #     loss.backward()
        #     self.optimizer_model.step()
        #     self.optimizer_kernel.step()
        #     # for name, param in self.model.named_parameters():  # 打印想要的模型的各个卷积层的名字和参数
        #     #     print(name)
        #     #     print(param)
        #     # if (epoch) % 10 == 0:
        #     torch.save(self.model.state_dict(), 'D:\BaiduNetdiskDownload\dantu\epoch_hongwai/model_epoch_%d_%d.pth' % (4, epoch))
        #     torch.save(self.kernel_net.state_dict(), 'D:\BaiduNetdiskDownload\dantu\epoch_hongwai/kernel_epoch_%d_%d.pth' % (4, epoch))
        #     # for name, param in self.model.named_parameters():  # 打印想要的模型的各个卷积层的名字和参数
        #     #     print(name)
        #     #     print(param)
        #     # print(list(self.model.parameters()))
        #     self.ckp.report_log(loss.item())
        #     # if (batch + 1) % self.args.print_every == 0:
        #     if (batch + 1) % 100== 0:
        #         self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[cycle: {:.4f}][mid: {:.4f}]'.format(
        #                 (batch + 1) * self.args.batch_size,
        #                 len(self.loader_train.dataset),
        #                 self.ckp.loss_log[-1] / (batch + 1),#平均下来每一个batch的总的损失
        #                 self.loss.display_loss(batch),#p平均下来的每一个batch的l_I
        #                 l_self_sum / (batch + 1),
        #                 kernel_Loss / (batch + 1)
        #             ))

        #验证rgb图像
        # self.model.eval()
        # self.kernel_net.eval()
        # netG= model.srgan.Generator(4).to(self.device)
        # # out_path = 'valid_result/SRF_' + str(4) + '/'
        # out_path = 'E:\\BaiduNetdiskDownload\\self_supervision_code\\training_results/SRF_' + str(4) + '/'
        # if not os.path.exists(out_path):
        #     os.makedirs(out_path)
        # with torch.no_grad():
        #     valing_results = {'psnr': 0, 'batch_sizes': 0, 'mse': 0}
        #     val_images = []
        #     tqdm_test = tqdm(self.loader_test, ncols=80)
        #     for idx_img, (input, gt, filename) in enumerate(tqdm_test):
        #         batch_size = input.size(0)
        #         valing_results['batch_sizes'] += batch_size
        #         input_list = [input[:, i, :, :, :].to(self.device) for i in range(self.args.n_sequence)]
        #         input_mid = input_list[self.args.n_sequence // 2]
        #         gt_list = [gt[:, i, :, :, :].to(self.device) for i in range(self.args.n_sequence)]
        #         gt_mid = gt_list[self.args.n_sequence // 2]
        #         if torch.cuda.is_available():
        #             input=input.cuda()
        #             gt=gt.cuda()
        #         # for name, param in self.model.named_parameters():  # 打印想要的模型的各个卷积层的名字和参数
        #         #     print(name)
        #         #     print(param)
        #         output_dict, mid_loss =self.model({'x': input})
        #         sr_self= output_dict['recons']
        #         sr_srgan =netG(input_mid)
        #         sr_interpolate= F.interpolate(input_mid, scale_factor=4, mode='bilinear', align_corners=False)
        #         val_images.extend(
        #             [(sr_srgan.cpu().squeeze(0)), (sr_self.cpu().squeeze(0)),
        #              (sr_interpolate.cpu().squeeze(0))])
        #         batch_mse = ((sr_srgan - gt_mid) ** 2).data.mean()
        #         valing_results['mse'] += batch_mse * batch_size
        #         valing_results['psnr'] = 10 * log10((gt_mid.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
        #
        #         print(valing_results['psnr'])
        #     val_images = torch.stack(val_images)
        #     val_images = torch.chunk(val_images, val_images.size(0))
        #     val_save_bar = tqdm(val_images, desc='[saving training results]')
        #     # if (epoch) % 10 == 0:
        #     index = 1
        #     for image in val_save_bar:
        #         image = utils.make_grid(image, nrow=3, padding=5)
        #         utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
        #         index += 1
        # # torch.save(self.model.state_dict(), 'E:\\BaiduNetdiskDownload\\self_supervision_code\\epoch/model_epoch_%d_%d.pth' % (4, epoch))
        # # torch.save(self.kernel_net.state_dict(), 'E:\\BaiduNetdiskDownload\\self_supervision_code\\epoch/kernel_epoch_%d_%d.pth' % (4, epoch))
        # out_path = 'E:\\BaiduNetdiskDownload\\self_supervision_code\\statistics/'
        # data_frame = pd.DataFrame(
        #     data={'PSNR': valing_results['psnr'], 'MSE': valing_results['mse']},
        #     index=range(1, epoch + 1))
        # data_frame.to_csv(out_path + 'srf_' + str(4) + '_train_results.csv', index_label='Epoch')

        # self.loss.end_log(len(self.loader_train))
        # self.mid_loss_log.append(mid_loss_sum / len(self.loader_train))
        # self.l_self_log.append(l_self_sum / len(self.loader_train))

        #验证红外图像
        self.model.eval()
        self.kernel_net.eval()
        # netG= model.srgan.Generator(4).to(self.device)
        out_path = 'D:\BaiduNetdiskDownload\dantu\\training_results/SRF_' + str(4) + '/'

        # out_path = '/tmp/pycharm_project_528/training_results/SRF_' + str(4) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with torch.no_grad():
            valing_results = {'psnr': 0, 'batch_sizes': 0, 'mse': 0}
            val_images = []
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for batch, (input, gt, filename) in enumerate(self.loader_test):
                # tqdm_test = tqdm(self.loader_test, ncols=80)
                batch_size = input.size(0)
                valing_results['batch_sizes'] += batch_size
                input_list = [input[:, i, :, :, :].to(self.device) for i in range(self.args.n_sequence)]
                input_mid = input_list[self.args.n_sequence // 2]
                gt_list = [gt[:, i, :, :, :].to(self.device) for i in range(self.args.n_sequence)]
                gt_mid = gt_list[self.args.n_sequence // 2]
                # PSNR1 = data_utils.calc_psnr(gt_mid, input_mid, rgb_range=self.args.rgb_range, is_rgb=True)
                # s=gt_mid-input_mid
                # utils.save_image(s, out_path + 'epoch_%d_index_%d.png' % (epoch, 4), padding=5)
                if torch.cuda.is_available():
                    input = input.cuda()
                # for name, param in self.model.named_parameters():  # 打印想要的模型的各个卷积层的名字和参数
                #     print(name)
                #     print(param)
                sr_interpolate = F.interpolate(input_mid, scale_factor=4, mode='bilinear', align_corners=False)
                utils.save_image(sr_interpolate, out_path + 'epoch_%d_index_%d.png' % (epoch, 3), padding=5)
                output_dict, mid_loss = self.model(input)
                sr_self = output_dict['recons']

                PSNR1 = data_utils.calc_psnr(gt_mid, sr_interpolate, rgb_range=self.args.rgb_range, is_rgb=True)
                PSNR2 = data_utils.calc_psnr(gt_mid, sr_self, rgb_range=self.args.rgb_range, is_rgb=True)
                a = (PSNR2 / PSNR1) - 1
                print(a)


                # sr_self1 = sr_self[:,0:1,:,:]
                # sr_self2 = sr_self[:, 1:2, :, :]
                # sr_self3 = sr_self[:, 2:3, :, :]
                # sr_self=(sr_self1+sr_self2+sr_self3)/3
                # utils.save_image(sr_self, out_path + 'liuxing_%d_index_%d.png' % (batch,4), padding=5)
                # # sr_srgan =netG(input_mid)
                # # utils.save_image(sr_srgan, out_path + 'epoch_%d_index_%d.png' % (epoch, 5), padding=5)
                # sr_interpolate = F.interpolate(input_mid, scale_factor=4, mode='bicubic', align_corners=False)
                # utils.save_image(sr_interpolate, out_path + 'epoch_%d_index_%d.png' % (epoch, 5), padding=5)
                # PSNR1 = data_utils.calc_psnr(gt_mid, sr_interpolate, rgb_range=self.args.rgb_range, is_rgb=True)
                # PSNR2 = data_utils.calc_psnr(gt_mid, sr_self, rgb_range=self.args.rgb_range, is_rgb=True)
                # # PSNR3=data_utils.calc_psnr(gt_mid, sr_srgan,rgb_range=self.args.rgb_range, is_rgb=True)
                # print(PSNR1)
                # print(PSNR2)
                # val_images.extend(
                #     [(sr_self.cpu().squeeze(0)),(sr_srgan.cpu().squeeze(0)),(sr_interpolate.cpu().squeeze(0))])
                val_images.extend(
                    [(sr_self.cpu().squeeze(0))])
                # batch_mse = ((sr_srgan - gt_mid) ** 2).data.mean()
                # valing_results['mse'] += batch_mse * batch_size
                # valing_results['psnr'] = 10 * log10((gt_mid.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                # print(valing_results['psnr'])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0))
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            # torch.save(self.model.state_dict(),
            #            'D:/BaiduNetdiskDownload/yuancheng/epoch_hongwai/model_epoch_%d_%d.pth' % (4, epoch))
            # torch.save(self.kernel_net.state_dict(),
            #            'D:/BaiduNetdiskDownload/yuancheng/epoch_hongwai/kernel_epoch_%d_%d.pth' % (4, epoch))

            # if (epoch) % 10 == 0:
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
                # val_save_bar.set_description(desc='[saving training results]')
            # for batch, (input, gt, filename) in enumerate(self.loader_test):
            #
            #     prefix = filename[0][0].split('.')[0]
            #     output_filename = filename[2][0].split('.', 1)[1] + '_4.fits'
            #     fits_position = 'D:\\BaiduNetdiskDownload\\01113\\test\\HR\\' + prefix + '\\' + \
            #                     filename[2][0].split('.', 1)[1] + '.fits'
            #     if torch.cuda.is_available():
            #         input = input.cuda()
            #     output_tensor = (torch.zeros((1, 3,1600, 1600), dtype=torch.float32))
            #     if torch.cuda.is_available():
            #         output_tensor = output_tensor.cuda()
            #     # Iterate over each region in the input images
            #     for i in range(4):
            #         for j in range(4):
            #             # Extract the region from each input image
            #             region_input = input[:, :, :, i * 100:(i + 1) * 100, j * 100:(j + 1) * 100]
            #             if torch.cuda.is_available():
            #                 region_input = region_input.cuda()
            #             output_dict, mid_loss = self.model(region_input)
            #             sr_self = output_dict['recons']
            #             output_tensor[:, :, i * 400:(i + 1) * 400, j * 400:(j + 1) * 400] += sr_self
            #
            #     output_tensor = output_tensor.squeeze(0)
            #     utils.save_image(output_tensor, 'A.png')
            #     # min_value = torch.min(output_tensor)
            #     # max_value = torch.max(output_tensor)
            #     # image_array= (output_tensor * (3950 - 375) + 375).cpu().numpy()
            #     # image_array = output_tensor.cpu().numpy() *65535
            #     image_array = output_tensor.cpu().numpy()
            #     # 转换为 16 位无符号整数
            #     image_array = np.uint16(image_array)
            #     # 选择适当的索引，例如 [0, 0, :, :]
            #     image_array = image_array[0, 0, :, :]
            #     header = fits.getheader(fits_position)  # 使用已有的头文件
            #     # original_objectxy = header['OBJECTXY']
            #     # new_objectxy = ' '.join(f'{float(coord) * 4:.2f}' for coord in original_objectxy.split())
            #     # header['OBJECTXY'] = new_objectxy
            #     hdu = fits.PrimaryHDU(image_array, header=header)
            #     hdul = fits.HDUList([hdu])
            #     hdul.writeto(output_filename, overwrite=True)

        # torch.save(self.model.state_dict(), 'E:\\BaiduNetdiskDownload\\self_supervision_code\\epoch/model_epoch_%d_%d.pth' % (4, epoch))
        # torch.save(self.kernel_net.state_dict(), 'E:\\BaiduNetdiskDownload\\self_supervision_code\\epoch/kernel_epoch_%d_%d.pth' % (4, epoch))
        # out_path = 'E:\\BaiduNetdiskDownload\\self_supervision_code\\statistics/'
        # data_frame = pd.DataFrame(
        #     data={'PSNR': valing_results['psnr'], 'MSE': valing_results['mse']},
        #     index=range(1, epoch + 1))
        # data_frame.to_csv(out_path + 'srf_' + str(4) + '_train_results.csv', index_label='Epoch')
        #
        # self.loss.end_log(len(self.loader_train))
        # self.mid_loss_log.append(mid_loss_sum / len(self.loader_train))
        # self.l_self_log.append(l_self_sum / len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')      #取出当前 epoch，并记录日志
        self.model.eval()               #关闭 dropout、batch normalization，使模型进入测试模式
        self.kernel_net.eval()           #关闭 dropout、batch normalization，使模型进入测试模式
        self.ckp.start_log(train=False)  #开始记录测试日志
        cycle_psnr_list = []

        with torch.no_grad():  # 意味着模型的梯度都不会更新了，循环内的每一个张量的required_grada都设置为false
            tqdm_test = tqdm(self.loader_test, ncols=80)  # 显示进度条
            for idx_img, (input, gt, filename) in enumerate(tqdm_test): # idx_img是索引，tqdm_test解包出（input是输入，gt是目标，filename是文件名）

                filename = filename[self.args.n_sequence // 2][0]

                input = input.to(self.device)
                input_center = input[:, self.args.n_sequence // 2, :, :, :]
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)

                output_dict, _ = self.model({'x': input})
                output = output_dict['recons']
                # kernel_list = output_dict['kernel_list']
                # est_kernel = kernel_list[self.args.n_sequence // 2]

                # lr_cycle_center = self.blur_down(gt, est_kernel, self.args.scale)

                # cycle_PSNR = data_utils.calc_psnr(input_center, lr_cycle_center, rgb_range=self.args.rgb_range,
                #                                   is_rgb=True)
                PSNR = data_utils.calc_psnr(gt, output, rgb_range=self.args.rgb_range, is_rgb=True)
                self.ckp.report_log(PSNR, train=False)
                # cycle_psnr_list.append(cycle_PSNR)

                # if self.args.save_images:
                #     gt, input_center, output, lr_cycle_center = data_utils.postprocess(
                #         gt, input_center, output, lr_cycle_center,
                #         rgb_range=self.args.rgb_range,
                #         ycbcr_flag=False,
                #         device=self.device)
                if self.args.save_images:
                    gt, input_center, output= data_utils.postprocess(
                        gt, input_center, output,
                        rgb_range=self.args.rgb_range,
                        ycbcr_flag=False,
                        device=self.device)

                    # est_kernel = self.process_kernel(est_kernel)
                    # save_list = [gt, input_center, output, lr_cycle_center, est_kernel]
                    save_list = [gt, input_center, output]
                    self.ckp.save_images(filename, save_list, epoch)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                self.ckp.psnr_log[-1],  # 整个测试集的平均psnr
                best[0], best[1] + 1))
            # self.cycle_psnr_log.append(sum(cycle_psnr_list) / len(cycle_psnr_list))

            # if not self.args.test_only:
            #     # self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
            #     self.ckp.plot_log(self.cycle_psnr_log, filename='cycle_psnr.pdf', title='Cycle PSNR')
            #     self.ckp.plot_log(self.mid_loss_log, filename='mid_loss.pdf', title='Mid Loss')
            #     self.ckp.plot_log(self.cycle_loss_log, filename='cycle_loss.pdf', title='Cycle Loss')
            #     torch.save({
            #         'cycle_psnr_log': self.cycle_psnr_log,
            #         'mid_loss_log': self.mid_loss_log,
            #         'cycle_loss_log': self.cycle_loss_log,
            #     }, os.path.join(self.ckp.dir, 'mid_logs.pt'))

    def conv_func(self, input, kernel, padding='same'):
        b, c, h, w = input.size()
        assert b == 1, "only support b=1!"
        _, _, ksize, ksize = kernel.size()
        if padding == 'same':
            pad = ksize // 2
        elif padding == 'valid':
            pad = 0
        else:
            raise Exception("not support padding flag!")

        conv_result_list = []
        for i in range(c):
            conv_result_list.append(F.conv2d(input[:, i:i + 1, :, :], kernel, bias=None, stride=1, padding=pad))
        conv_result = torch.cat(conv_result_list, dim=1)
        return conv_result

    def blur_down(self, x, kernel, scale):
        b,c, h, w = x.size()
        _,kc, ksize, _ = kernel.size()
        psize = ksize // 2
        assert kc == 1, "only support kc=1!"

        # blur
        x = F.pad(x, (psize, psize, psize, psize), mode='replicate')
        blur_list = []
        for i in range(b):
            blur_list.append(self.conv_func(x[i:i + 1, :, :, :], kernel[i:i + 1, :, :, :]))
        blur = torch.cat(blur_list, dim=0)
        blur = blur[:, :, psize:-psize, psize:-psize]

        # down
        blurdown = blur[:, :, ::scale, ::scale]

        return blurdown

    def process_kernel(self, kernel):
        mi = torch.min(kernel)
        ma = torch.max(kernel)
        kernel = (kernel - mi) / (ma - mi)
        kernel = torch.cat([kernel, kernel, kernel], dim=1)
        kernel = kernel.mul(255.).clamp(0, 255).round()
        return kernel
