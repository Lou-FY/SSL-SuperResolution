import os
import glob
import utils.data_utils as utils
import numpy as np
import imageio
import torch
import cv2
import random
import math
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F




class VALID(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.n_seq = args.n_sequence
        self.n_frames_per_video = args.n_frames_per_video
        print("n_seq:", args.n_sequence)
        print("n_frames_per_video:", args.n_frames_per_video)

        self.n_frames_video = []

        if train:
            self._set_filesystem(args.dir_data)
        else:
            self._set_filesystem(args.dir_data_test)

        self.images_gt, self.images_input = self._scan()

        self.num_video = len(self.images_gt)
        self.num_frame = sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

        if train:
            self.repeat = max(args.test_every // max((self.num_frame // self.args.batch_size), 1), 1)
            self.repeat = 1
            print("Dataset repeat:", self.repeat)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data
        self.dir_gt = os.path.join(self.apath, 'HR')
        self.dir_input = os.path.join(self.apath, 'LR')
        print("DataSet GT path:", self.dir_gt)

    def _scan(self):
        vid_gt_names = sorted(glob.glob(os.path.join(self.dir_gt, '*')))
        vid_input_names = sorted(glob.glob(os.path.join(self.dir_input, '*')))
        assert len(vid_gt_names) == len(vid_input_names), "len(vid_gt_names) must equal len(vid_input_names)"

        images_gt = []
        images_input = []
        for vid_gt_name, vid_input_name in zip(vid_gt_names, vid_input_names):
            # for vid_input_name in vid_input_names:
            if self.train:
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))[:self.args.n_frames_per_video]
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))[:self.args.n_frames_per_video]
            else:
                gt_dir_names = sorted(glob.glob(os.path.join(vid_gt_name, '*')))
                input_dir_names = sorted(glob.glob(os.path.join(vid_input_name, '*')))
            images_gt.append(gt_dir_names)
            images_input.append(input_dir_names)
            self.n_frames_video.append(len(gt_dir_names))
            # self.n_frames_video.append(len(input_dir_names))

        return images_gt, images_input

        # def __getitem__(self, idx):
        #     lqs, filenames = self._load_file(idx)
        #
        #     lqs_list = [lqs[i, :, :, :] for i in range(self.n_seq)]
        #     lqs_concat = np.concatenate(lqs_list, axis=2)
        #     # lqs_concat = self.get_patch(lqs_concat, self.args.size_must_mode, self.args.scale)  #为什么会报错
        #     lqs_concat = self.get_patch(lqs_concat, self.args.size_must_mode, scale=1)
        #     lqs_list = [lqs_concat[:, :, i * self.args.n_colors:(i + 1) * self.args.n_colors] for i in range(self.n_seq)]
        #     #
        #     # kernel = self.get_blur_kernel(trian=self.train)
        #     # inputs_list = [self.get_lr_blur_down(g, kernel, self.args.scale) for g in gts_list]  # blur + downsample
        #
        #     inputs = np.array(lqs_list)
        #     # gts = np.array(gts_list)
        #
        #     input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        #     # gt_tensors = utils.np2Tensor(*gts, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        #
        #     # return torch.stack(input_tensors), torch.stack(gt_tensors), filenames
        #
        #     return torch.stack(input_tensors),  filenames

    def __getitem__(self, idx):
        # inputs, gts, filenames = self._load_file(idx)
        #
        # inputs_list = [inputs[i,  :, :] for i in range(self.n_seq)]
        # inputs_concat = np.stack(inputs_list, axis=2)
        # gts_list = [gts[i, :, :] for i in range(self.n_seq)]
        # gts_concat = np.stack(gts_list, axis=2)
        # inputs_concat, gts_concat = self.get_patch(inputs_concat, gts_concat,
        #                                            self.args.size_must_mode, scale=self.args.scale)
        # inputs_list = [inputs_concat[:, :, i * 1:(i + 1) *1] for i in
        #                range(self.n_seq)]
        # gts_list = [gts_concat[:, :, i * 1:(i + 1) * 1] for i in
        #             range(self.n_seq)]
        #
        # inputs = np.array(inputs_list)
        # gts = np.array(gts_list)
        #
        # input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        # gt_tensors = utils.np2Tensor(*gts, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        inputs, gts, filenames = self._load_file(idx)

        inputs_list = [inputs[i, :, :, :] for i in range(self.n_seq)]
        inputs_concat = np.concatenate(inputs_list, axis=2)
        gts_list = [gts[i, :, :, :] for i in range(self.n_seq)]
        gts_concat = np.concatenate(gts_list, axis=2)
        inputs_concat, gts_concat = self.get_patch(inputs_concat, gts_concat,
                                                   self.args.size_must_mode, scale=self.args.scale)
        inputs_list = [inputs_concat[:, :, i * self.args.n_colors:(i + 1) * self.args.n_colors] for i in
                       range(self.n_seq)]
        gts_list = [gts_concat[:, :, i * self.args.n_colors:(i + 1) * self.args.n_colors] for i in
                    range(self.n_seq)]

        inputs = np.array(inputs_list)
        gts = np.array(gts_list)

        input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        gt_tensors = utils.np2Tensor(*gts, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)

        return torch.stack(input_tensors), torch.stack(gt_tensors), filenames

        return torch.stack(input_tensors), torch.stack(gt_tensors), filenames

    def __len__(self):
        if self.train:
            return self.num_frame * self.repeat
        else:
            return self.num_frame

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_frame
        else:
            return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        idx = self._get_index(idx)
        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)  # test时，根据idx获取对应的视频id和帧id
        f_gts = self.images_gt[video_idx][frame_idx:frame_idx + self.n_seq]
        f_inputs = self.images_input[video_idx][frame_idx:frame_idx + self.n_seq]
        # gts = np.array([imageio.imread(hr_name) for hr_name in f_gts], dtype=np.float64)
        # inputs = np.array([imageio.imread(lr_name) for lr_name in f_inputs], dtype=np.float64)
        gts = np.array([imageio.imread(hr_name) for hr_name in f_gts], dtype=np.float32)# 2 180 320 3
        # inputs = np.expand_dims(np.array([imageio.imread(lr_name) for lr_name in f_inputs], dtype=np.float32), axis=-1)
        inputs = np.array([imageio.imread(lr_name) for lr_name in f_inputs], dtype=np.float32)
        filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_gts]

        return inputs, gts, filenames

    def get_patch(self, input, gt, size_must_mode=1, scale=1):
        if self.train:
            input_patch, gt_patch = utils.get_patch(input, gt, patch_size=self.args.patch_size, scale=scale)
            mid_b, mid_e = (self.n_seq // 2) * self.args.n_colors, (self.n_seq // 2 + 1) * self.args.n_colors
            mean_edge = self.cal_smooth(gt_patch[:, :, mid_b:mid_e])
            n_loop = 1
            while mean_edge < 7 and n_loop < 5:  # drop smooth patch
                input_patch, gt_patch = utils.get_patch(input, gt, patch_size=self.args.patch_size, scale=scale)
                mean_edge = self.cal_smooth(gt_patch[:, :, mid_b:mid_e])
                n_loop += 1

            h, w, c = input_patch.shape
            size_must_mode = size_must_mode
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input_patch, gt_patch = input_patch[:new_h, :new_w, :], gt_patch[:new_h * scale, :new_w * scale, :]
            if not self.args.no_augment:
                input_patch, gt_patch = utils.data_augment(input_patch, gt_patch)
        else:
            input_patch, gt_patch = input, gt
            h, w, c = input_patch.shape
            size_must_mode = size_must_mode
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input_patch, gt_patch = input_patch[:new_h, :new_w, :], gt_patch[: new_h*4, :new_w*4, :]

            # input_patch, gt_patch = input_patch[:512, :512, :], gt_patch[: 400 * 4, :400 * 4, :] #左上
            # input_patch, gt_patch = input_patch[:512, 512:1024, :], gt_patch[: 400 * 4, :400 * 4, :] #右上
            # input_patch, gt_patch = input_patch[512:1024, :512, :], gt_patch[: 400 * 4, :400 * 4, :]#左下
            # input_patch, gt_patch = input_patch[512:1025, 512:1024, :], gt_patch[: 400 * 4, :400 * 4, :]#右下
        return input_patch, gt_patch

    def cal_smooth(self, img):
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        smooth = np.mean(dst)
        return smooth

    def matlab_style_gauss2D(self, shape=(5, 5), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_blur_kernel(self, trian=True):
        if trian:
            gaussian_sigma = random.choice(
                [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        else:
            gaussian_sigma = 2.0
        gaussian_blur_kernel_size = int(math.ceil(gaussian_sigma * 3) * 2 + 1)
        kernel = self.matlab_style_gauss2D((gaussian_blur_kernel_size, gaussian_blur_kernel_size), gaussian_sigma)
        return kernel

    def get_lr_blur_down(self, img_gt, kernel, scale):
        img_gt = np.array(img_gt).astype('float32')
        gt_tensor = torch.from_numpy(img_gt.transpose(2, 0, 1)).unsqueeze(0).float()

        kernel_size = kernel.shape[0]
        psize = kernel_size // 2
        gt_tensor = F.pad(gt_tensor, (psize, psize, psize, psize), mode='replicate')

        gaussian_blur = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size, stride=1,
                                  padding=int((kernel_size - 1) // 2), bias=False)
        nn.init.constant_(gaussian_blur.weight.data, 0.0)
        gaussian_blur.weight.data[0, 0, :, :] = torch.FloatTensor(kernel)
        gaussian_blur.weight.data[1, 1, :, :] = torch.FloatTensor(kernel)
        gaussian_blur.weight.data[2, 2, :, :] = torch.FloatTensor(kernel)

        blur_tensor = gaussian_blur(gt_tensor)
        blur_tensor = blur_tensor[:, :, psize:-psize, psize:-psize]

        blur_tensor = blur_tensor.clamp(0, 255).round()
        blur = blur_tensor[0].detach().numpy().transpose(1, 2, 0).astype('uint8').astype('float32')

        blurdown = blur[::scale, ::scale, :]

        return blurdown
