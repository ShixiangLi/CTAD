from .base import BaseDataset
import torch


PSM_CLASS_NAMES = [
    'time_series'
]

class PSMDataset(BaseDataset):
    def __init__(self, args, is_train=True, class_name=None):
        if class_name is not None:
            class_list = [class_name]
        else:
            class_list = PSM_CLASS_NAMES
        super().__init__(args, is_train, class_list)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='D:/workspace/temp/py/llm/rlr/datasets/psm/', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--norm_mean', default=[0.485, 0.456, 0.406], type=list)
    parser.add_argument('--norm_std', default=[0.229, 0.224, 0.225], type=list)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    args = parser.parse_args()
    dataset = PSMDataset(args, is_train=False, class_name='time_series')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    for i, (image, label, mask, file_name, img_anomaly) in enumerate(dataloader):
        print(image.shape, label, mask, file_name, img_anomaly)
        if i == 0:
            break