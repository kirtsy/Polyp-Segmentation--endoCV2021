import torch
import numpy as np
import argparse
from scipy import misc
from test_related_file.test_related_class import *
import os
import cv2
from datetime import datetime
import imageio


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size') #384
parser.add_argument('--pth_path', type=str, default='./ckpts/model-50.pth')#ckpt_YCH_THU_002.pth


testset= "./TestDataset/"  #"../endocv2021-test-noCopyAllowed-v1/"
testset_folder=os.listdir(testset)

def post_process(img):
    ret, th = cv2.threshold(img, 254, 256, cv2.THRESH_BINARY)
    if (th[np.nonzero(th)]).shape[0] > 100:
        ret1, th1 = cv2.threshold(img, 180, 256, cv2.THRESH_BINARY)
        return th1
    else:
        ret1, th1 = cv2.threshold(img, 255, 256, cv2.THRESH_BINARY)
        return th1

count=0

ablation='no_ppd'

model = None
model = PraNet().cuda()


for _data_name in testset_folder: # input the different test dataset folder name
    time = []
    time_without=[]
    data_path ='./TestDataset/{}/'.format(_data_name)
    save_path = '/EndoCV2021/segmentation/{}/'.format(_data_name+"_pred")
    opt = parser.parse_args()
    model = PraNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    test_loader = test_dataset(data_path+_data_name+"/",opt.testsize)

    for i in range(test_loader.size):


        image, size, name = test_loader.load_data()
        result_size = np.asarray(size, np.float32)
        result_size /= (result_size.max() + 1e-8)

        image = image.cuda()
        start_time = datetime.now()

        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.interpolate(res, size=result_size.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        data_file=save_path+name.split('.')[0]+ '_mask.jpg'

        misc.imsave(data_file, res)

        # -------------------- post-processing ---------------------------
        img = cv2.imread(data_file)
        res_post=post_process(img)
        cv2.imwrite(data_file, res_post)



print("------------------------------Segmentation Finish------------------------------")