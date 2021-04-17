import torch
import numpy as np
from thop import profile
from thop import clever_format
import os


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:adam
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    #optimizer：adam
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay   #learning rate will change to: lrarning_rate * decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def draw_loss(loss_list,title):
    import matplotlib.pyplot as plt
    save_path="./loss/"
    os.makedirs(save_path, exist_ok=True)
    # 设置图表标题，并给坐标轴加上标签
    plt.plot(loss_list, linewidth=2)  # 参数linewidth决定了plot()绘制的线条的粗细。
    plt.title(title, fontsize=24)  # 函数title()给图表指定标题
    plt.xlabel('Epoch', fontsize=14)  # 函数xlabel()和ylabel()让你能够为每条轴设置标题
    plt.ylabel('Loss', fontsize=14)
    plt.tick_params(axis='both',
                    labelsize=12)  # 而函数tick_params()设置刻度的样式，其中指定的实参将影响x轴和y轴上的刻度（axes='both'），并将刻度标记的字号设置为14（labelsize=14）。
    plt.savefig(save_path+title+".png")
    plt.show()




def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))