import torch
from scipy.io import loadmat,savemat
import numpy as np
from torchvision import transforms, datasets
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
import os
import random

def y2label(x):
    # print(x)
    y = []
    for i in range(len(x)):
        # print(x[i])
        x[i] = x[i].replace(' ', '')
        if(x[i] == 'Control'):
            y.append(0)
        # elif(x[i] == 'Schizophrenia'):
        elif (x[i] in  ['Autism', 'Schizophrenia', 'ADHD', 'Bipolar']):
            # print(x[i])
            y.append(1)
        # else: print(x[i])
    # print(y)
    return y

def y2onehot(x):
    y = []
    for i in range(len(x)):
        if(x[i] == 0):
            y.append([1,0])
        elif(x[i] == 1):
            y.append([0,1])
    return y

# 单个模块的数据划分
class MyDataset(Dataset):

    def __init__(self, datapath, type):
        matdata = loadmat(datapath)
        # self.label = np.expand_dims(lable2onehot(matdata['label']), axis=1)
        self.label = y2label(matdata['label'])
        if(type == 'fnc'):
            self.data = matdata['fnc']
        if(type == 'tc'):
            self.data = matdata['tc']

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        return self.label[index], self.data[index]

    def __len__(self):
        # 返回数据的总数量
        return len(self.label)

# 数据集划分
def training_data(path, type, batchsize):
    mydata = MyDataset(path, type)
    train_db, val_db = torch.utils.data.random_split(mydata, [90, 10])
    nw = min([os.cpu_count(), batchsize if batchsize > 1 else 0, 8])  # number of workers
    train_loader = DataLoader(dataset=train_db, batch_size=batchsize, shuffle=True, num_workers=nw)
    val_loader = DataLoader(dataset=val_db, batch_size=batchsize, shuffle=True, num_workers=nw)
    return train_loader, val_loader

# 对两个model的数据一起划分
# K折交叉验证
class Dataset2BothModel(Dataset):

    def __init__(self, sfc_path, tc_path, ki=0, K=10, typ='train', transform=None, rand=False):
        '''
        sfc_path、tc_path: 数据的路径
        ki：当前是第几折,从0开始，范围为[0, K)
        K：总的折数
        typ：用于区分训练集与验证集
        transform：对图片的数据增强
        rand：是否随机
        '''

        seed = 50

        matdata_sfc = loadmat(sfc_path)
        # matdata_tc = loadmat(tc_path)
        self.labelTemp = y2label(matdata_sfc['labels'])
        self.data_sfcTemp = matdata_sfc['fnc']
        self.data_tcTemp = matdata_sfc['tcs']

        # print(self.labelTemp.shape)
        # print(self.data_sfcTemp.shape)
        # print(self.data_tcTemp.shape)

        self.label0 = []
        self.label1 = []
        self.data_sfcTemp0 = []
        self.data_sfcTemp1 = []
        self.data_tcTemp0 = []
        self.data_tcTemp1 = []

        setShuffleTimes = 0
        if setShuffleTimes != 0:
            for i in range(setShuffleTimes):
                random.seed(seed)
                random.shuffle(self.labelTemp)
                random.seed(seed)
                random.shuffle(self.data_sfcTemp)
                random.seed(seed)
                random.shuffle(self.data_tcTemp)

        for i in range(len(self.labelTemp)):
            if self.labelTemp[i] == 0:
                self.label0.append(self.labelTemp[i])
                self.data_sfcTemp0.append(self.data_sfcTemp[i])
                self.data_tcTemp0.append(self.data_tcTemp[i])
            else:
                self.label1.append(self.labelTemp[i])
                self.data_sfcTemp1.append(self.data_sfcTemp[i])
                self.data_tcTemp1.append(self.data_tcTemp[i])


        self.label0 = np.array(self.label0)
        self.label1 = np.array(self.label1)
        self.data_sfcTemp0 = np.array(self.data_sfcTemp0)
        self.data_sfcTemp1 = np.array(self.data_sfcTemp1)
        self.data_tcTemp0 = np.array(self.data_tcTemp0)
        self.data_tcTemp1 = np.array(self.data_tcTemp1)

        leng0 = len(self.label0)
        leng1 = len(self.label1)
        every_z_len0 = leng0 // K
        every_z_len1 = leng1 // K
        # self.data_sfc = np.array([[]])
        # print(self.data_sfc.shape)
        if typ == 'val':
            # self.data_info = self.all_data_info[every_z_len * ki: every_z_len * (ki + 1)]
            self.label = self.label0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.label = np.append(self.label, self.label1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)
            self.data_sfc = self.data_sfcTemp0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)
            self.data_tc = self.data_tcTemp0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)

        elif typ == 'train':
            # self.data_info = self.all_data_info[: every_z_len * ki] + self.all_data_info[every_z_len * (ki + 1):]
            self.label = self.label0[: every_z_len0 * ki]
            self.label = np.append(self.label, self.label1[: every_z_len1 * ki], axis=0)
            # self.label = self.labelTemp[every_z_len * (ki + 1):]
            self.label = np.append(self.label, self.label0[every_z_len0 * (ki + 1):], axis=0)
            self.label = np.append(self.label, self.label1[every_z_len1 * (ki + 1):], axis=0)
            # print(self.label.shape)
            self.data_sfc = self.data_sfcTemp0[: every_z_len0 * ki]
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[: every_z_len1 * ki], axis=0)
            # self.data_sfc = self.data_sfcTemp[every_z_len * (ki + 1):]
            # print(self.data_sfc.shape)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp0[every_z_len0 * (ki + 1):], axis=0)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[every_z_len1 * (ki + 1):], axis=0)
            # print(self.data_sfc.shape)

            self.data_tc = self.data_tcTemp0[: every_z_len0 * ki]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[: every_z_len1 * ki], axis=0)
            # self.data_tc = self.data_tcTemp[every_z_len * (ki + 1):]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp0[every_z_len0 * (ki + 1):], axis=0)
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[every_z_len1 * (ki + 1):], axis=0)

        # print(self.label.shape, self.data_sfc.shape, self.data_tc.shape)

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        return self.label[index], self.normalize_data(self.data_sfc[index]), self.normalize_data(self.data_tc[index])

    def __len__(self):
        # 返回数据的总数量
        return len(self.label)

    def normalize_data(self, data):
        # return data
        max_val = np.max(np.abs(data))
        normalized_data = data / max_val
        return normalized_data

class Dataset2BothModel_ml(Dataset):

    def __init__(self, sfc_path, tc_path, ki=0, K=10, typ='train', transform=None, rand=False):
        '''
        sfc_path、tc_path: 数据的路径
        ki：当前是第几折,从0开始，范围为[0, K)
        K：总的折数
        typ：用于区分训练集与验证集
        transform：对图片的数据增强
        rand：是否随机
        '''

        seed = 50

        matdata_sfc = loadmat(sfc_path)
        # matdata_tc = loadmat(tc_path)
        self.labelTemp = y2label(matdata_sfc['labels'])
        self.data_sfcTemp = matdata_sfc['fnc_flatten']
        self.data_tcTemp = matdata_sfc['tcs']

        # print(self.labelTemp)
        self.label0 = []
        self.label1 = []
        self.data_sfcTemp0 = []
        self.data_sfcTemp1 = []
        self.data_tcTemp0 = []
        self.data_tcTemp1 = []

        setShuffleTimes = 0
        if setShuffleTimes != 0:
            for i in range(setShuffleTimes):
                random.seed(seed)
                random.shuffle(self.labelTemp)
                random.seed(seed)
                random.shuffle(self.data_sfcTemp)
                random.seed(seed)
                random.shuffle(self.data_tcTemp)

        for i in range(len(self.labelTemp)):
            if self.labelTemp[i] == 0:
                self.label0.append(self.labelTemp[i])
                self.data_sfcTemp0.append(self.data_sfcTemp[i])
                self.data_tcTemp0.append(self.data_tcTemp[i])
            else:
                self.label1.append(self.labelTemp[i])
                self.data_sfcTemp1.append(self.data_sfcTemp[i])
                self.data_tcTemp1.append(self.data_tcTemp[i])


        self.label0 = np.array(self.label0)
        self.label1 = np.array(self.label1)
        self.data_sfcTemp0 = np.array(self.data_sfcTemp0)
        self.data_sfcTemp1 = np.array(self.data_sfcTemp1)
        self.data_tcTemp0 = np.array(self.data_tcTemp0)
        self.data_tcTemp1 = np.array(self.data_tcTemp1)

        leng0 = len(self.label0)
        leng1 = len(self.label1)
        every_z_len0 = leng0 // K
        every_z_len1 = leng1 // K
        # self.data_sfc = np.array([[]])
        # print(self.data_sfc.shape)
        if typ == 'val':
            # self.data_info = self.all_data_info[every_z_len * ki: every_z_len * (ki + 1)]
            self.label = self.label0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.label = np.append(self.label, self.label1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)
            self.data_sfc = self.data_sfcTemp0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)
            self.data_tc = self.data_tcTemp0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)

        elif typ == 'train':
            # self.data_info = self.all_data_info[: every_z_len * ki] + self.all_data_info[every_z_len * (ki + 1):]
            self.label = self.label0[: every_z_len0 * ki]
            self.label = np.append(self.label, self.label1[: every_z_len1 * ki], axis=0)
            # self.label = self.labelTemp[every_z_len * (ki + 1):]
            self.label = np.append(self.label, self.label0[every_z_len0 * (ki + 1):], axis=0)
            self.label = np.append(self.label, self.label1[every_z_len1 * (ki + 1):], axis=0)
            # print(self.label.shape)
            self.data_sfc = self.data_sfcTemp0[: every_z_len0 * ki]
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[: every_z_len1 * ki], axis=0)
            # self.data_sfc = self.data_sfcTemp[every_z_len * (ki + 1):]
            # print(self.data_sfc.shape)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp0[every_z_len0 * (ki + 1):], axis=0)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[every_z_len1 * (ki + 1):], axis=0)
            # print(self.data_sfc.shape)

            self.data_tc = self.data_tcTemp0[: every_z_len0 * ki]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[: every_z_len1 * ki], axis=0)
            # self.data_tc = self.data_tcTemp[every_z_len * (ki + 1):]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp0[every_z_len0 * (ki + 1):], axis=0)
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[every_z_len1 * (ki + 1):], axis=0)

        # print(self.label.shape, self.data_sfc.shape, self.data_tc.shape)

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        return self.label[index], self.normalize_data(self.data_sfc[index]), self.normalize_data(self.data_tc[index])

    def __len__(self):
        # 返回数据的总数量
        return len(self.label)

    def normalize_data(self, data):
        max_val = np.max(np.abs(data))
        normalized_data = data / max_val
        return normalized_data


def y2label_sub_cls(x):
    # print(x)
    y = []
    category = {'Autism': 0, 'Aspergers': 0, 'PDD-NOS': 0, 'Control': 0, 'other': 0}
    for i in range(len(x)):
        # print(x[i])
        x[i] = x[i].replace(' ', '')
        # print(x[i])
        if(x[i] == 'Autism'):
            y.append(0)
            category['Autism'] += 1
        elif (x[i] == 'Aspergers'):
            y.append(1)
            category['Aspergers'] += 1
        elif (x[i] == 'PDD-NOS'):
            y.append(2)
            category['PDD-NOS'] += 1
        elif (x[i] == 'Control'):
            y.append(3)
            category['Control'] += 1
        else:
            # print(x[i])
            category['other'] += 1
        # else: print(x[i])
    # print(y)
    print("Total: ", category)
    return y
class Dataset2BothModel_sub_cls(Dataset):

    def __init__(self, sfc_path, tc_path, ki=0, K=10, typ='train', transform=None, rand=False):
        print('----------------------{}----------------------'.format(typ))
        '''
        sfc_path、tc_path: 数据的路径
        ki：当前是第几折,从0开始，范围为[0, K)
        K：总的折数
        typ：用于区分训练集与验证集
        transform：对图片的数据增强
        rand：是否随机
        '''

        seed = 50

        matdata_sfc = loadmat(sfc_path)
        # matdata_tc = loadmat(tc_path)
        self.labelTemp = y2label_sub_cls(matdata_sfc['subtype'])
        # print(self.labelTemp)
        self.data_sfcTemp = matdata_sfc['fnc']
        self.data_tcTemp = matdata_sfc['tcs']

        # print(self.labelTemp)
        self.label0 = []
        self.label1 = []
        self.label2 = []
        self.label3 = []
        self.data_sfcTemp0 = []
        self.data_sfcTemp1 = []
        self.data_sfcTemp2 = []
        self.data_sfcTemp3 = []
        self.data_tcTemp0 = []
        self.data_tcTemp1 = []
        self.data_tcTemp2 = []
        self.data_tcTemp3 = []


        setShuffleTimes = 0
        if setShuffleTimes != 0:
            for i in range(setShuffleTimes):
                random.seed(seed)
                random.shuffle(self.labelTemp)
                random.seed(seed)
                random.shuffle(self.data_sfcTemp)
                random.seed(seed)
                random.shuffle(self.data_tcTemp)

        for i in range(len(self.labelTemp)):
            if self.labelTemp[i] == 0:
                self.label0.append(self.labelTemp[i])
                self.data_sfcTemp0.append(self.data_sfcTemp[i])
                self.data_tcTemp0.append(self.data_tcTemp[i])
            elif self.labelTemp[i] == 1:
                self.label1.append(self.labelTemp[i])
                self.data_sfcTemp1.append(self.data_sfcTemp[i])
                self.data_tcTemp1.append(self.data_tcTemp[i])
            elif self.labelTemp[i] == 2:
                self.label2.append(self.labelTemp[i])
                self.data_sfcTemp2.append(self.data_sfcTemp[i])
                self.data_tcTemp2.append(self.data_tcTemp[i])
            elif self.labelTemp[i] == 3:
                self.label3.append(self.labelTemp[i])
                self.data_sfcTemp3.append(self.data_sfcTemp[i])
                self.data_tcTemp3.append(self.data_tcTemp[i])

        self.label0 = np.array(self.label0)
        self.label1 = np.array(self.label1)
        self.label2 = np.array(self.label2)
        self.label3 = np.array(self.label3)
        self.data_sfcTemp0 = np.array(self.data_sfcTemp0)
        self.data_sfcTemp1 = np.array(self.data_sfcTemp1)
        self.data_sfcTemp2 = np.array(self.data_sfcTemp2)
        self.data_sfcTemp3 = np.array(self.data_sfcTemp3)
        self.data_tcTemp0 = np.array(self.data_tcTemp0)
        self.data_tcTemp1 = np.array(self.data_tcTemp1)
        self.data_tcTemp2 = np.array(self.data_tcTemp2)
        self.data_tcTemp3 = np.array(self.data_tcTemp3)

        leng0 = len(self.label0)
        leng1 = len(self.label1)
        leng2 = len(self.label2)
        leng3 = len(self.label3)
        every_z_len0 = leng0 // K
        every_z_len1 = leng1 // K
        every_z_len2 = leng2 // K
        every_z_len3 = leng3 // K
        # self.data_sfc = np.array([[]])
        # print(self.data_sfc.shape)
        if typ == 'val':
            # self.data_info = self.all_data_info[every_z_len * ki: every_z_len * (ki + 1)]
            self.label = self.label0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.label = np.append(self.label, self.label1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)
            self.label = np.append(self.label, self.label2[every_z_len2 * ki: every_z_len2 * (ki + 1)], axis=0)
            # self.label = np.append(self.label, self.label3[every_z_len3 * ki: every_z_len3 * (ki + 1)], axis=0)
            self.data_sfc = self.data_sfcTemp0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp2[every_z_len2 * ki: every_z_len2 * (ki + 1)], axis=0)
            # self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp3[every_z_len3 * ki: every_z_len3 * (ki + 1)], axis=0)
            self.data_tc = self.data_tcTemp0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)
            self.data_tc = np.append(self.data_tc, self.data_tcTemp2[every_z_len2 * ki: every_z_len2 * (ki + 1)], axis=0)
            # self.data_tc = np.append(self.data_tc, self.data_tcTemp3[every_z_len3 * ki: every_z_len3 * (ki + 1)], axis=0)

        elif typ == 'train':
            # self.data_info = self.all_data_info[: every_z_len * ki] + self.all_data_info[every_z_len * (ki + 1):]
            self.label = self.label0[: every_z_len0 * ki]
            self.label = np.append(self.label, self.label1[: every_z_len1 * ki], axis=0)
            self.label = np.append(self.label, self.label2[: every_z_len2 * ki], axis=0)
            # self.label = np.append(self.label, self.label3[: every_z_len3 * ki], axis=0)
            # self.label = self.labelTemp[every_z_len * (ki + 1):]
            self.label = np.append(self.label, self.label0[every_z_len0 * (ki + 1):], axis=0)
            self.label = np.append(self.label, self.label1[every_z_len1 * (ki + 1):], axis=0)
            self.label = np.append(self.label, self.label2[every_z_len2 * (ki + 1):], axis=0)
            # self.label = np.append(self.label, self.label3[every_z_len3 * (ki + 1):], axis=0)
            # print(self.label.shape)
            self.data_sfc = self.data_sfcTemp0[: every_z_len0 * ki]
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[: every_z_len1 * ki], axis=0)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp2[: every_z_len2 * ki], axis=0)
            # self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp3[: every_z_len3 * ki], axis=0)
            # self.data_sfc = self.data_sfcTemp[every_z_len * (ki + 1):]
            # print(self.data_sfc.shape)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp0[every_z_len0 * (ki + 1):], axis=0)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[every_z_len1 * (ki + 1):], axis=0)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp2[every_z_len2 * (ki + 1):], axis=0)
            # self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp3[every_z_len3 * (ki + 1):], axis=0)
            # print(self.data_sfc.shape)

            self.data_tc = self.data_tcTemp0[: every_z_len0 * ki]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[: every_z_len1 * ki], axis=0)
            self.data_tc = np.append(self.data_tc, self.data_tcTemp2[: every_z_len2 * ki], axis=0)
            # self.data_tc = np.append(self.data_tc, self.data_tcTemp3[: every_z_len3 * ki], axis=0)
            # self.data_tc = self.data_tcTemp[every_z_len * (ki + 1):]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp0[every_z_len0 * (ki + 1):], axis=0)
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[every_z_len1 * (ki + 1):], axis=0)
            self.data_tc = np.append(self.data_tc, self.data_tcTemp2[every_z_len2 * (ki + 1):], axis=0)
            # self.data_tc = np.append(self.data_tc, self.data_tcTemp3[every_z_len3 * (ki + 1):], axis=0)


        print(self.label.shape, self.data_sfc.shape, self.data_tc.shape)

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        return self.label[index], self.normalize_data(self.data_sfc[index]), self.normalize_data(self.data_tc[index])

    def __len__(self):
        # 返回数据的总数量
        return len(self.label)

    def normalize_data(self, data):
        # return data
        max_val = np.max(np.abs(data))
        normalized_data = data / max_val
        return normalized_data

def y2label_comb_cls(x):
    # print(x)
    y = []
    category = {'Schizophrenia': 0, 'Autism': 0, 'ADHD': 0}
    for i in range(len(x)):
        # print(x[i])
        x[i] = x[i].replace(' ', '')
        # print(x[i])
        if(x[i] == 'Schizophrenia'):
            y.append(0)
            category['Schizophrenia'] += 1
        elif (x[i] == 'Autism'):
            y.append(1)
            category['Autism'] += 1
        elif (x[i] == 'ADHD'):
            y.append(2)
            category['ADHD'] += 1
        # else: print(x[i])
    # print(y)
    print("Total: ", category)
    return y
class Dataset2BothModel_combine_cls(Dataset):

    def __init__(self, sfc_path, tc_path, ki=0, K=10, typ='train', transform=None, rand=False):
        print('----------------------{}----------------------'.format(typ))
        '''
        sfc_path、tc_path: 数据的路径
        ki：当前是第几折,从0开始，范围为[0, K)
        K：总的折数
        typ：用于区分训练集与验证集
        transform：对图片的数据增强
        rand：是否随机
        '''

        seed = 50

        matdata_sfc = loadmat(sfc_path)
        # matdata_tc = loadmat(tc_path)
        self.labelTemp = y2label_comb_cls(matdata_sfc['labels'])
        # print(self.labelTemp)
        self.data_sfcTemp = matdata_sfc['fnc']
        self.data_tcTemp = matdata_sfc['tcs']

        # print(self.labelTemp)
        self.label0 = []
        self.label1 = []
        self.label2 = []
        self.label3 = []
        self.data_sfcTemp0 = []
        self.data_sfcTemp1 = []
        self.data_sfcTemp2 = []
        self.data_sfcTemp3 = []
        self.data_tcTemp0 = []
        self.data_tcTemp1 = []
        self.data_tcTemp2 = []
        self.data_tcTemp3 = []


        setShuffleTimes = 0
        if setShuffleTimes != 0:
            for i in range(setShuffleTimes):
                random.seed(seed)
                random.shuffle(self.labelTemp)
                random.seed(seed)
                random.shuffle(self.data_sfcTemp)
                random.seed(seed)
                random.shuffle(self.data_tcTemp)

        for i in range(len(self.labelTemp)):
            if self.labelTemp[i] == 0:
                self.label0.append(self.labelTemp[i])
                self.data_sfcTemp0.append(self.data_sfcTemp[i])
                self.data_tcTemp0.append(self.data_tcTemp[i])
            elif self.labelTemp[i] == 1:
                self.label1.append(self.labelTemp[i])
                self.data_sfcTemp1.append(self.data_sfcTemp[i])
                self.data_tcTemp1.append(self.data_tcTemp[i])
            elif self.labelTemp[i] == 2:
                self.label2.append(self.labelTemp[i])
                self.data_sfcTemp2.append(self.data_sfcTemp[i])
                self.data_tcTemp2.append(self.data_tcTemp[i])
            # elif self.labelTemp[i] == 3:
            #     self.label3.append(self.labelTemp[i])
            #     self.data_sfcTemp3.append(self.data_sfcTemp[i])
            #     self.data_tcTemp3.append(self.data_tcTemp[i])

        self.label0 = np.array(self.label0)
        self.label1 = np.array(self.label1)
        self.label2 = np.array(self.label2)
        # self.label3 = np.array(self.label3)
        self.data_sfcTemp0 = np.array(self.data_sfcTemp0)
        self.data_sfcTemp1 = np.array(self.data_sfcTemp1)
        self.data_sfcTemp2 = np.array(self.data_sfcTemp2)
        # self.data_sfcTemp3 = np.array(self.data_sfcTemp3)
        self.data_tcTemp0 = np.array(self.data_tcTemp0)
        self.data_tcTemp1 = np.array(self.data_tcTemp1)
        self.data_tcTemp2 = np.array(self.data_tcTemp2)
        # self.data_tcTemp3 = np.array(self.data_tcTemp3)

        leng0 = len(self.label0)
        leng1 = len(self.label1)
        leng2 = len(self.label2)
        # leng3 = len(self.label3)
        every_z_len0 = leng0 // K
        every_z_len1 = leng1 // K
        every_z_len2 = leng2 // K
        # every_z_len3 = leng3 // K
        # self.data_sfc = np.array([[]])
        # print(self.data_sfc.shape)
        if typ == 'val':
            # self.data_info = self.all_data_info[every_z_len * ki: every_z_len * (ki + 1)]
            self.label = self.label0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.label = np.append(self.label, self.label1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)
            self.label = np.append(self.label, self.label2[every_z_len2 * ki: every_z_len2 * (ki + 1)], axis=0)
            # self.label = np.append(self.label, self.label3[every_z_len3 * ki: every_z_len3 * (ki + 1)], axis=0)
            self.data_sfc = self.data_sfcTemp0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp2[every_z_len2 * ki: every_z_len2 * (ki + 1)], axis=0)
            # self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp3[every_z_len3 * ki: every_z_len3 * (ki + 1)], axis=0)
            self.data_tc = self.data_tcTemp0[every_z_len0 * ki: every_z_len0 * (ki + 1)]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[every_z_len1 * ki: every_z_len1 * (ki + 1)], axis=0)
            self.data_tc = np.append(self.data_tc, self.data_tcTemp2[every_z_len2 * ki: every_z_len2 * (ki + 1)], axis=0)
            # self.data_tc = np.append(self.data_tc, self.data_tcTemp3[every_z_len3 * ki: every_z_len3 * (ki + 1)], axis=0)

        elif typ == 'train':
            # self.data_info = self.all_data_info[: every_z_len * ki] + self.all_data_info[every_z_len * (ki + 1):]
            self.label = self.label0[: every_z_len0 * ki]
            self.label = np.append(self.label, self.label1[: every_z_len1 * ki], axis=0)
            self.label = np.append(self.label, self.label2[: every_z_len2 * ki], axis=0)
            # self.label = np.append(self.label, self.label3[: every_z_len3 * ki], axis=0)
            # self.label = self.labelTemp[every_z_len * (ki + 1):]
            self.label = np.append(self.label, self.label0[every_z_len0 * (ki + 1):], axis=0)
            self.label = np.append(self.label, self.label1[every_z_len1 * (ki + 1):], axis=0)
            self.label = np.append(self.label, self.label2[every_z_len2 * (ki + 1):], axis=0)
            # self.label = np.append(self.label, self.label3[every_z_len3 * (ki + 1):], axis=0)
            # print(self.label.shape)
            self.data_sfc = self.data_sfcTemp0[: every_z_len0 * ki]
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[: every_z_len1 * ki], axis=0)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp2[: every_z_len2 * ki], axis=0)
            # self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp3[: every_z_len3 * ki], axis=0)
            # self.data_sfc = self.data_sfcTemp[every_z_len * (ki + 1):]
            # print(self.data_sfc.shape)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp0[every_z_len0 * (ki + 1):], axis=0)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp1[every_z_len1 * (ki + 1):], axis=0)
            self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp2[every_z_len2 * (ki + 1):], axis=0)
            # self.data_sfc = np.append(self.data_sfc, self.data_sfcTemp3[every_z_len3 * (ki + 1):], axis=0)
            # print(self.data_sfc.shape)

            self.data_tc = self.data_tcTemp0[: every_z_len0 * ki]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[: every_z_len1 * ki], axis=0)
            self.data_tc = np.append(self.data_tc, self.data_tcTemp2[: every_z_len2 * ki], axis=0)
            # self.data_tc = np.append(self.data_tc, self.data_tcTemp3[: every_z_len3 * ki], axis=0)
            # self.data_tc = self.data_tcTemp[every_z_len * (ki + 1):]
            self.data_tc = np.append(self.data_tc, self.data_tcTemp0[every_z_len0 * (ki + 1):], axis=0)
            self.data_tc = np.append(self.data_tc, self.data_tcTemp1[every_z_len1 * (ki + 1):], axis=0)
            self.data_tc = np.append(self.data_tc, self.data_tcTemp2[every_z_len2 * (ki + 1):], axis=0)
            # self.data_tc = np.append(self.data_tc, self.data_tcTemp3[every_z_len3 * (ki + 1):], axis=0)


        print(self.label.shape, self.data_sfc.shape, self.data_tc.shape)

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        return self.label[index], self.normalize_data(self.data_sfc[index]), self.normalize_data(self.data_tc[index])

    def __len__(self):
        # 返回数据的总数量
        return len(self.label)

    def normalize_data(self, data):
        # return data
        max_val = np.max(np.abs(data))
        normalized_data = data / max_val
        return normalized_data

if __name__ == '__main__':

    # sfc_path = 'data/tz_sfc.mat'  # (100,1225)
    # tc_path = 'data/tz_tc.mat'  # (100,100,50)
    # train_loader, val_loader = training_data(sfc_path, 'fnc', 10)
    # print("train:", len(train_loader))
    # print("val:", len(val_loader))
    # for i in train_loader:
    #     print(i[0].shape, i[1].shape)
    #     # print(i[0])
    #     # print(i[1])
    #     break
    K = 10
    for ki in range(K):
        trainset = Dataset2BothModel(sfc_path = 'data/tz_sfc.mat', tc_path = 'data/tz_tc.mat', ki=ki, K=K, typ='train')
        valset = Dataset2BothModel(sfc_path = 'data/tz_sfc.mat', tc_path = 'data/tz_tc.mat', ki=ki, K=K, typ='val')
        # print(ki)

        train_loader = DataLoader(
            dataset=trainset,
            batch_size=10,
            shuffle=True)
        val_loader = DataLoader(
            dataset=valset,
            batch_size=2,
        )
        print(len(train_loader))
        for i in train_loader:
            print("train")
            print(i[0].shape, i[1].shape, i[2].shape)

            break
        # print(len(val_loader))
        # for i in val_loader:
        #     print("val")
        #     print(i[0].shape, i[1].shape, i[2].shape)
        #     # print(i[0])
        #     # print(i[1])
        #     break
        break
