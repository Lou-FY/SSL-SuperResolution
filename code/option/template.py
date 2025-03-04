def set_template(args):
    if args.template == 'KernelPredict':
        args.task = "PretrainKernel"
        args.model = "Kernel"
        args.save = "Kernel_Pretrain"
        args.data_train = 'REDS_ONLINE'
        # args.data_train = 'REDS_HRLR'
        args.dir_data = 'E:\\BaiduNetdiskDownload\\Deep-Blind-VSR-main\\dataset'
        args.data_test = 'REDS_HRLR'
        args.dir_data_test = 'E:\\BaiduNetdiskDownload\\Deep-Blind-VSR-main\\dataset'
        args.scale = 4
        args.patch_size = 64
        args.n_sequence = 5
        args.n_frames_per_video = 50
        args.est_ksize = 13
        args.loss = '1*L1'
        args.lr = 5e-4
        args.lr_decay = 20
        args.save_middle_models = True
        args.save_images = True
        args.epochs = 100
        args.batch_size = 1
        # args.resume = True
        # args.load = args.save
    elif args.template == 'VideoSR':
        args.task = "FlowVideoSR"
        args.model = "PWC_Recons"
        args.save = "Deep_Blind_VSR"
        args.data_train = 'REDS_ONLINE'
        # args.dir_data = '/tmp/pycharm_project_528/train-liuxing'
        args.dir_data = 'D:\BaiduNetdiskDownload\dantu\\train'
        args.data_test = 'VALID'
        # args.dir_data_test = '/tmp/pycharm_project_528/test'
        args.dir_data_test = 'D:\BaiduNetdiskDownload\dantu\\test3'
        args.scale = 4
        args.patch_size = 64
        args.n_sequence = 5
        args.n_frames_per_video = 50
        args.n_feat = 256
        args.n_cond = 64
        args.est_ksize = 13
        args.extra_RBS = 1
        args.recons_RBS = 3
        args.loss = '1*L1'
        args.lr = 1e-5
        args.lr_decay = 100
        args.save_middle_models = True
        args.save_images = True
        args.epochs = 500
        args.batch_size = 1
        # args.resume = True
        # args.load = args.save
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
