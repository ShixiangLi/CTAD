import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class BaseDataset(Dataset):
    def __init__(self, args, is_train=True, class_list=[]):
        self.dataset_path = args.data_path + args.dataset
        self.cropsize = args.crop_size
        self.is_train = is_train
        self.class_list = class_list

        self.x, self.y, self.mask, self.anomalies = self.load_dataset()

        self.transform_x = T.Compose([
            T.Resize(args.img_size, Image.LANCZOS),
            T.CenterCrop(args.crop_size),
            T.ToTensor()
        ])
        self.transform_mask = T.Compose([
            T.Resize(args.img_size, Image.NEAREST),
            T.CenterCrop(args.crop_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(args.norm_mean, args.norm_std)])

    def __getitem__(self, idx):
        img_path, y, mask, anomaly = self.x[idx], self.y[idx], self.mask[idx], self.anomalies[idx]
        
        x = Image.open(img_path).convert('RGB')
        x = self.normalize(self.transform_x(x))
        
        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
            if mask.shape[0] > 1:
                mask = mask[0].reshape(1, mask.shape[1], mask.shape[2])
        
        return x, y, mask, os.path.basename(img_path[:-4]), anomaly

    def __len__(self):
        return len(self.x)

    def load_dataset(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask, anomalies = [], [], [], []

        for class_name in self.class_list:
            img_dir = os.path.join(self.dataset_path, class_name, phase)
            gt_dir = os.path.join(self.dataset_path, class_name, 'ground_truth')

            anomaly_list = sorted(os.listdir(img_dir))
            for anomaly in anomaly_list:
                anomaly_img_dir = os.path.join(img_dir, anomaly)
                if not os.path.isdir(anomaly_img_dir):
                    continue
                img_list = sorted([os.path.join(anomaly_img_dir, f) for f in os.listdir(anomaly_img_dir)])
                x.extend(img_list)

                # load gt labels
                if anomaly == 'good':
                    y.extend([0] * len(img_list))
                    mask.extend([None] * len(img_list))
                    anomalies.extend(['good'] * len(img_list))
                else:
                    y.extend([1] * len(img_list))
                    anomaly_gt_dir = os.path.join(gt_dir, anomaly)
                    img_names = [os.path.splitext(os.path.basename(f))[0] for f in img_list]
                    gt_list = [os.path.join(anomaly_gt_dir, f + '_mask.png') for f in img_names]
                    mask.extend(gt_list)
                    anomalies.extend([anomaly] * len(img_list))

        return list(x), list(y), list(mask), list(anomalies)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='')
    # basic config
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--with_decoder', action='store_true', default=False)
    # dataset config
    parser.add_argument('--dataset', default='smd', type=str, choices=['mvtec', 'visa', 'smd'])
    parser.add_argument('--data_path', default='datasets/smd/', type=str)
    parser.add_argument('--inp_size', default=256, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    # model config
    parser.add_argument('--backbone_arch', default='tf_efficientnet_b6', type=str)
    parser.add_argument('--feature_levels', default=2, type=int)
    parser.add_argument('--out_indices', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--block_mode', type=str, default='parallel', choices=['parallel', 'serial'])
    parser.add_argument('--blocks', nargs='+', type=str, default=['mca', 'nsa'])
    parser.add_argument('--blocks_gate', type=str, default='none', choices=['none', 'gate', 'net'])
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--ref_len', type=int, default=1)
    parser.add_argument('--ref_loss', action='store_true', default=False)
    # trainer config
    parser.add_argument('--feature_jitter', type=float, default=0.0)
    parser.add_argument('--noise_prob', type=float, default=1.0)
    parser.add_argument('--no_avg', action='store_true', default=False)
    parser.add_argument('--with_mask', action='store_true', default=False)
    # misc
    parser.add_argument('--save_path', type=str, default='workspace/temp/py/llm/llms-for-ad/save')
    parser.add_argument('--save_prefix', type=str, default='')
    parser.add_argument('--vis', action='store_true', default=False)
    args = parser.parse_args()
    args.device = torch.device("cuda")
    args.img_size = (args.inp_size, args.inp_size)
    args.crop_size = (args.inp_size, args.inp_size)
    args.norm_mean, args.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    args.save_path = f'{args.root_path}/{args.save_path}'
    # os.makedirs(args.save_path, exist_ok=True)

    dataset = BaseDataset(args, is_train=False, class_list=['time_series'])
    print(len(dataset))
