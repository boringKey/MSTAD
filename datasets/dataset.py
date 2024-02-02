import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

# 定义数据集路径和转换操作
data_dir = r'F:\Data\MvTec'
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),  # 将所有图片都调整为 256x256 大小
    transforms.ToTensor()  # 将图片转换为张量
])

# 创建一个字典，将每个数据集的名称和类别数映射起来
dataset_info = {
    'bottle': 1,
    'cable': 2,
    'capsule': 3,
    'carpet': 4,
    'grid': 5,
    'hazelnut': 6,
    'leather': 7,
    'metal_nut': 8,
    'pill': 9,
    'screw': 10,
    'tile': 11,
    'toothbrush': 12,
    'transistor': 13,
    'wood': 14,
    'zipper': 15
}

# 定义一个自定义 dataset 类，实现 __len__ 和 __getitem__ 方法
class MvTecDataset(torch.utils.data.Dataset):
    def __init__(self,  root_dir, class_name, size, mode="Train"):
        self.root_dir = root_dir
        self.class_name = class_name
        self.mode = mode
        self.size = size

        self.transform_gt = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor()])
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor()])
        x, y, gt, class_name_list = [], [], [], []
        self.image_names, self.y, self.gt, self.class_name_list = self.load_dataset(x, y, gt, class_name_list)

    def __len__(self):
        return len(self.image_names)

    def load_dataset(self, x, y, gt, class_name_list):
        for clsname in self.class_name:
            image_path = os.path.join(self.root_dir, clsname, self.mode)
            gt_path = os.path.join(self.root_dir, clsname, 'ground_truth')

            image_types = sorted(os.listdir(image_path))

            for type in image_types:
                image_type_path = os.path.join(image_path, type)
                if not os.path.isdir(image_type_path):
                    continue
                image_list = sorted([os.path.join(image_type_path, f)
                                     for f in os.listdir(image_type_path) if f.endswith('.png')])
                x.extend(image_list)
                for name in range(len(x)):
                    class_name_list.append(clsname)

                if type == 'good':
                    y.extend([0] * len(image_list))
                    gt.extend([None] * len(image_list))

                else:
                    y.extend([1] * len(image_list))
                    image_name_list = [os.path.splitext(os.path.basename(f))[0] for f in image_list]
                    gt_list = [os.path.join(gt_path, type, image_name + '_mask.png')
                               for image_name in image_name_list]
                    gt.extend(gt_list)
                assert len(x) == len(y) and len(x) == len(gt)
        return list(x), list(y), list(gt), list(class_name_list)


    def __getitem__(self, idx):
        image_names, y, gt, class_name_list = self.image_names[idx], self.y[idx], self.gt[idx], self.class_name_list[idx]
        x = Image.open(image_names).convert('RGB')
        x = self.transform(x)
        if y == 0:
            mask = torch.zeros([1, self.size, self.size])
        else:
            mask = Image.open(gt).convert('1')
            mask = self.transform_gt(mask)
        # mask.to(torch.bool)
        return x, y, mask, class_name_list

class BTADDataset(torch.utils.data.Dataset):
    def __init__(self,  root_dir, class_name, size, mode="Train"):
        self.root_dir = root_dir
        self.class_name = class_name
        self.mode = mode
        self.size = size

        self.transform_gt = transforms.Compose([
            transforms.Resize([self.size, self.size]),
            transforms.CenterCrop(self.size),
            transforms.ToTensor()])
        self.transform = transforms.Compose([
            # transforms.Resize(self.size),
            transforms.Resize([self.size, self.size]),
            transforms.CenterCrop(self.size),
            transforms.ToTensor()])
        x, y, gt, class_name_list = [], [], [], []
        self.image_names, self.y, self.gt, self.class_name_list = self.load_dataset(x, y, gt, class_name, class_name_list)

    def __len__(self):
        return len(self.image_names)

    def load_dataset(self, x, y, gt, clsname, class_name_list):
        image_path = os.path.join(self.root_dir, clsname, self.mode)
        gt_path = os.path.join(self.root_dir, clsname, 'ground_truth')

        image_types = sorted(os.listdir(image_path))

        for type in image_types:
            image_type_path = os.path.join(image_path, type)
            if not os.path.isdir(image_type_path):
                continue
            image_list = sorted([os.path.join(image_type_path, f)
                                 for f in os.listdir(image_type_path) if f.endswith('.bmp') or f.endswith('.png')])
            x.extend(image_list)
            for name in range(len(x)):
                class_name_list.append(clsname)
            if type == 'ok':
                y.extend([0] * len(image_list))
                gt.extend([None] * len(image_list))

            else:
                y.extend([1] * len(image_list))
                image_name_list = [os.path.splitext(os.path.basename(f))[0] for f in image_list]
                if clsname == '03':
                    gt_list = [os.path.join(gt_path, type, image_name + '.bmp')
                               for image_name in image_name_list]
                else:
                    gt_list = [os.path.join(gt_path, type, image_name + '.png')
                               for image_name in image_name_list]
                gt.extend(gt_list)
            assert len(x) == len(y) and len(x) == len(gt)
        return list(x), list(y), list(gt), class_name_list


    def __getitem__(self, idx):
        image_names, y, gt, class_name_list = self.image_names[idx], self.y[idx], self.gt[idx], self.class_name_list[idx]
        input = {}
        input.update({
            "filename": image_names,
            "label": y,
            "clsname": class_name_list,
        })
        x = Image.open(image_names).convert('RGB')
        input.update({
            'width': x.size[0],
            'height': x.size[1]
        })
        x = self.transform(x)

        if y == 0:
            mask = torch.zeros([1, self.size, self.size])
        else:
            mask = Image.open(gt).convert('1')
            mask = self.transform_gt(mask)
        # mask.to(torch.bool)
        input.update({
            "mask": mask,
            "image": x
        })
        return input

class VisaDataset(torch.utils.data.Dataset):
    def __init__(self,  root_dir, class_name, size, mode="Normal"):
        self.root_dir = root_dir
        self.class_name = class_name
        self.mode = mode
        self.size = size

        self.transform_gt = transforms.Compose([
            transforms.Resize([self.size, self.size]),
            transforms.CenterCrop(self.size),
            transforms.ToTensor()])
        self.transform = transforms.Compose([
            transforms.Resize([self.size, self.size]),
            transforms.CenterCrop(self.size),
            transforms.ToTensor()])
        x, y, gt, class_name_list = [], [], [], []
        self.image_names, self.y, self.gt, self.class_name_list = self.load_dataset(x, y, gt, class_name, class_name_list)

    def __len__(self):
        return len(self.image_names)

    def load_dataset(self, x, y, gt, clsname, class_name_list):
        image_path = os.path.join(self.root_dir, clsname,'Data', 'Images')
        gt_path = os.path.join(self.root_dir, clsname,'Data', 'Masks')

        image_types = sorted(os.listdir(image_path))
        type = self.mode
        # for type in image_types:
        image_type_path = os.path.join(image_path, type)
        # if not os.path.isdir(image_type_path):
        #     continue
        image_list = sorted([os.path.join(image_type_path, f)
                             for f in os.listdir(image_type_path) if f.endswith('.bmp') or f.endswith('.JPG') or f.endswith('.png')])
        x.extend(image_list)
        for name in range(len(x)):
            class_name_list.append(clsname)
        if type == 'Normal':
            y.extend([0] * len(image_list))
            gt.extend([None] * len(image_list))
        else:
            y.extend([1] * len(image_list))

            image_name_list = [os.path.splitext(os.path.basename(f))[0] for f in image_list]
            gt_list = [os.path.join(gt_path, type, image_name + '.png')
                       for image_name in image_name_list]
            gt.extend(gt_list)
        assert len(x) == len(y) and len(x) == len(gt)
        return list(x), list(y), list(gt), class_name_list


    def __getitem__(self, idx):
        image_names, y, gt, class_name_list = self.image_names[idx], self.y[idx], self.gt[idx], self.class_name_list[idx]
        input = {}
        input.update({
            "filename": image_names,
            "label": y,
            "clsname": class_name_list,
        })
        x = Image.open(image_names).convert('RGB')
        input.update({
            'width': x.size[0],
            'height': x.size[1]
        })
        x = self.transform(x)
        if y == 0:
            mask = torch.zeros([1, self.size, self.size])
        else:
            mask = Image.open(gt).convert('1')
            mask_array = np.array(mask)
            mask_array[mask_array != 0] = 255
            mask = Image.fromarray(mask_array)
            # mask = Image.open(mask)
            mask = self.transform_gt(mask)
        # mask.to(torch.bool)
        input.update({
            "mask": mask,
            "image": x
        })
        return input



if __name__ == '__main__':

    # 创建一个列表，存储所有的 dataset
    datasets_list = []
    dataloader_list = []
    # 遍历 dataset_info 中的每个数据集
    # for dataset_name, num_classes in dataset_info.items():
    #     # 构建数据集对象
    #     Train_dataset = MvTecDataset(r'F:\Data\MvTec', dataset_name, 32, 'Train')
    #     Test_dataset = MvTecDataset(r'F:\Data\MvTec', dataset_name, 32, 'Test')
    #     Train_loader = DataLoader(Train_dataset, batch_size=32, shuffle=True, num_workers=1)
    #     Test_loader = DataLoader(Test_dataset, batch_size=32, shuffle=True, num_workers=1)
    #     datasets_list.append(Test_dataset)
    #     dataloader_list.append(Test_loader)
    # train01 = BTADDataset('/home/scu-its-gpu-001/PycharmProjects/BTech_Dataset_transformed', '01', 32, 'train')
    # train02 = BTADDataset('/home/scu-its-gpu-001/PycharmProjects/BTech_Dataset_transformed', '02', 32, 'train')
    # train03 = BTADDataset('/home/scu-its-gpu-001/PycharmProjects/BTech_Dataset_transformed', '03', 32, 'train')
    # allDataset = ConcatDataset([train01, train02, train03])
    # dataloader = DataLoader(allDataset, batch_size=64, shuffle=True)
    # test = BTADDataset('/home/scu-its-gpu-001/PycharmProjects/BTech_Dataset_transformed', '01', 32, 'test')
    train = VisaDataset('/home/scu-its-gpu-001/PycharmProjects/VisaData', 'candle', 224, 'Normal')
    test = VisaDataset('/home/scu-its-gpu-001/PycharmProjects/VisaData', 'candle', 224, 'Anomaly')
    # print(train)


