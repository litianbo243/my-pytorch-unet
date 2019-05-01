import os
from PIL import Image
from torch.utils import data
import random
from torchvision import transforms as T

# import sys
# sys.path.append("..")


# 数据集
# 分好训练，验证，测试集
# 做数据扩增
# 对数据进行0中心化

class DataSet(data.Dataset):

    def __init__(self, root, train=True, val=False, test=False, is_transforms=True, ):

        self.root = root
        self.train = train
        self.val = val
        self.test = test
        self.is_transforms = is_transforms

        self.train_data_root = os.path.join(self.root, "/train/image")
        self.train_label_root = os.path.join(self.root, "/train/label")
        self.val_data_root = os.path.join(self.root, "/train/image")
        self.val_label_root = os.path.join(self.root, "/train/label")
        self.test_data_root = os.path.join(self.root, "/test")
        self.test_label_root = os.path.join(self.root, "/test")

        self.normalize = T.Normalize(mean=0.5, std=0.5)

        if self.train or self.val:
            data_files = os.listdir(self.train_data_root)

            random.seed(0)
            random.shuffle(data_files)
            data_files_name = [data_file.split(".")[0] for data_file in data_files]
            label_files = [data_files_name + "_predict.png" for data_files_name in data_files_name]
            files_num = len(data_files)

            if self.train:
                self.data_full_root = [os.path.join(self.train_data_root, data_file) for data_file in data_files][:int(0.7*files_num)]
                self.label_full_root = [os.path.join(self.train_label_root, label_file) for label_file in label_files][:int(0.7*files_num)]

            if self.val:
                data_files = os.listdir(self.test_data_root)
                data_files_name = [data_file.split(".")[0] for data_file in data_files]
                label_files = [data_files_name + "_predict.png" for data_files_name in data_files_name]
                files_num = len(data_files)
                self.data_full_root = [os.path.join(self.train_data_root, data_file) for data_file in data_files][int(0.7*files_num):]
                self.label_full_root = [os.path.join(self.train_label_root, label_file) for label_file in label_files][int(0.7*files_num):]

        if self.test:

            data_files = os.listdir(self.test_data_root)
            test_data_files = []
            test_label_files = []
            for file in data_files:
                if "predict" not in file:
                    test_data_files.append(file)
                if "predict" in file:
                    test_label_files.append(file)

            self.data_full_root = [os.path.join(self.test_data_root, file) for file in test_data_files]
            self.label_full_root = [os.path.join(self.test_label_root, file) for file in test_label_files]

        if self.is_transforms:
            if self.val or self.test:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    self.normalize
                ])
            if self.train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    self.normalize
                ])
        else:
            self.transforms = T.Compose([T.ToTensor(), self.normalize])


    def __getitem__(self, item):
        data_full_root = self.data_full_root[item]
        label_full_root = self.label_full_root[item]

        data = Image.open(data_full_root)
        data = self.transforms(data)
        label = Image.open(label_full_root)
        label = self.transforms(label)
        return data, label

    def __len__(self):
        return len(self.data_full_root)