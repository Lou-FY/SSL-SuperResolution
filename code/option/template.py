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
        # args.resume = True #true的话，从最近保存中拉取
        # args.load = args.save    #如果args.load为'.'「default」,则说明是新实验
    elif args.template == 'VideoSR':
        args.task = "FlowVideoSR"
        args.model = "PWC_Recons"
        args.save = "Deep_Blind_VSR"#用于logger中生成实验目录

        
        ##args.dir_data = 'D:\BaiduNetdiskDownload\dantu\\train'     #训练模式下，data.test文件中会调用，来构建train数据集低分辨率图像路径
        # args.dir_data = '/tmp/pycharm_project_528/train-liuxing'
        ##args.dir_data_test = 'D:\BaiduNetdiskDownload\dantu\\test3'#测试模式下用在了data.test、data.valid文件中组成test数据集LR路径
        # args.dir_data_test = '/tmp/pycharm_project_528/test'
        
        ##args.data_train = 'REDS_ONLINE' #data.init文件train数据导入中，用来动态导入对应名称的train数据集模块；
        args.data_test = 'VALID'          #logger、data.init文件中，用在了组成test生成图像HR结果保存路径中
        #在 __init__.py 中，会通过 args.data_test 动态导入对应名称（大写形式，如 VALID）的测试数据集模块；
        # args.data_test = 'REDS_HRLR'
        
        
    
        
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
