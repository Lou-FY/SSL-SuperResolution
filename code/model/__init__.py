import os
from importlib import import_module
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.args = args
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_middle_models = args.save_middle_models#中间模型是否保存
        module = import_module('model.' + args.model.lower())#动态导入template文件中规定的模型
        self.model = module.make_model(args).to(self.device)#调用make_model函数，实例化生成模型
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))
        self.load(
            ckp.dir,#ckp是个logger实例，dir是实验目录
            pre_train=args.pre_train,
            resume=args.resume,#true的话，从最近的保存中拉取
            cpu=args.cpu
        )
        print(self.get_model(), file=ckp.log_file)

    def forward(self, *args):
        return self.model(*args)

    def get_model(self):
        if not self.cpu and self.n_GPUs > 1:
            return self.model.module
        else:
            return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )
        if self.save_middle_models:
            if epoch % 1 == 0:
                torch.save(
                    target.state_dict(),
                    os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
                )

    def load(self, apath, pre_train='.', resume=False, cpu=False):  #
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs), strict=False
            )
        elif resume:
            print('Loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(os.path.join(apath, 'model', 'model_latest.pt'), **kwargs),
                strict=False
            )
        elif self.args.test_only:
            self.get_model().load_state_dict(
                torch.load(os.path.join(apath, 'model', 'model_best.pt'), **kwargs),
                strict=False
            )
        else:
            pass
