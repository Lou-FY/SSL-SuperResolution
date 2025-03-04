import torch
import torch.nn as nn
from model import flow_pwc
import model.blocks as blocks
from model.kernel import KernelNet
import torch.nn.functional as F
import math
import model.deconv_fft as deconv_fft
from model.arch_util import ResidualBlockNoBN, make_layer, DCNv2Pack
from model.pwc_recons import  PWC_Recons
class Main_batch(nn.Module):
    def __init__(self, n_colors=3, n_sequence=5, n_feat=32,
                 scale=4, flow_pretrain_fn='.'):
        super(Main_batch, self).__init__()
        self.kernel_net=KernelNet(ksize=13).to(self.device)
        self.net= PWC_Recons(n_colors=3, n_sequence=5, n_feat=32,
                 scale=4, flow_pretrain_fn='.')


    def forward(self,  input):
        input_list = [input[:, i, :, :, :] for i in range(self.args.n_sequence)]
        est_kernel=self.Kernel_net(input)
        aux_lr_seq = torch.stack([self.blur_down(g, est_kernel, 4) for g in input_list], dim=1)




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

