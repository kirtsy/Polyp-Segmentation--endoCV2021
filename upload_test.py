import numpy as np
import argparse
from scipy import misc
from test_related_file.test_related_class import *


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/self-train/PraNet-20.pth')#./snapshots/PraNet_Res2Net/PraNet-19.pth

for _data_name in ['CVC-300']:#, 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB'
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/PraNet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = PraNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        # image, gt, name = test_loader.load_data()
        # gt = np.asarray(gt, np.float32)
        # gt /= (gt.max() + 1e-8)
        image, size, name = test_loader.load_data()
        result_size = np.asarray(size, np.float32)
        result_size /= (result_size.max() + 1e-8)

        # img_size = np.asarray(image.size, np.float32)
        # img_size /= (img_size.size.max() + 1e-8)

        image = image.cuda()


        print("======================= gt size:",result_size.shape,"=======================\n")

        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.interpolate(res, size=result_size.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)

print("finish")