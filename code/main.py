import torch
print(torch.__version__)  # PyTorch 版本
print(torch.version.cuda)  # PyTorch 使用的 CUDA 版本
print(torch.cuda.is_available())  # 是否检测到 GPU
print(torch.cuda.get_device_name(0))  # 显示 GPU 型号
import data
import model
import loss
import option
from trainer.trainer_kernel import Trainer_Kernel
from trainer.trainer_flow_video import Trainer_Flow_Video
from logger import logger
##--template KernelPredict
#--template VideoSR
args = option.args
torch.manual_seed(args.seed)
chkp = logger.Logger(args)

print("Selected task: {}".format(args.task))
model = model.Model(args, chkp)
loss = loss.Loss(args, chkp) if not args.test_only else None
loader = data.Data(args)

if args.task == 'PretrainKernel':
    t = Trainer_Kernel(args, loader, model, loss, chkp)
elif args.task == 'FlowVideoSR':
    t = Trainer_Flow_Video(args, loader, model, loss, chkp)
else:
    raise NotImplementedError('Task [{:s}] is not found'.format(args.task))


while not t.terminate(): 
    #t.train()
    t.test()
    # torch.save(model.state_dict(), '/tmp/pycharm_project_431/epoch100.pth')

chkp.done()
