import numpy as np
import torch
from scipy.io import loadmat
import nibabel as nib
from nilearn import plotting
import loadData
import model as md
import argparse
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

def configs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, default='COBRE', help='dataset name')
    parser.add_argument('--model', type=str, default='DP-FP', help='model name')
    parser.add_argument('--weight', type=str, default='./weights/best_network_IC100_s50.pth', help='trained model weight')
    parser.add_argument('--data', type=str, default='./datas/COBRE_IC100_s50.mat', help='dataset path')
    parser.add_argument('--structure', type=bool, default=False, help='print model structure')
    parser.add_argument('--target_label', type=str, default="SZ")
    return parser

def get_data(args):

    matdata_sfc = loadmat(args.data)
    label = np.array(loadData.y2label(matdata_sfc['labels']))
    fc = np.array(matdata_sfc['fnc'])
    tc = np.array(matdata_sfc['tcs'])
    # print('fc shape:', fc.shape, 'tc shape:', tc.shape, 'label shape:', label.shape)

    input_FC = []
    input_TC = []
    input_Label = []

    len = label.shape[0]

    # 只保留SZ
    for i in range(len):
        if args.target_label == "SZ":
            if label[i] == 0: continue
        elif args.target_label == "HC":
            if label[i] == 1: continue

        input_FC.append(fc[i])
        input_TC.append(tc[i])
        input_Label.append(label[i])

    return input_FC, input_TC, input_Label

def compute_feature_contributions(model, input_FC, input_TC, input_Label):

    # 将模型设置为评估模式
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    # 获取最后一层的权重
    last_layer_weights = model.fc.weight
    LR_weights = model.LR.weight
    # LR_weights 与last_layer_weights点乘
    # total_weight = torch.matmul(LR_weights, last_layer_weights)
    # print(total_weight.shape, last_layer_weights.shape, LR_weights.shape)
    # 创建一个空的特征贡献度列表
    feature_contributions = []

    # 遍历输入数据
    for i, data in enumerate(zip(input_FC, input_TC, input_Label)):

        # 获取数据
        fc, tc, l = data
        # 将数据转换为张量
        fc_tensor = torch.Tensor(fc)
        tc_tensor = torch.Tensor(tc)
        # 增加一个维度
        fc_tensor = fc_tensor.unsqueeze(0)
        tc_tensor = tc_tensor.unsqueeze(0)

        # bm = model(tc_tensor, fc_tensor,eva = True)

        # print(bm.shape)


        # 前向传播计算输出
        output, _ = model(tc_tensor, fc_tensor)
        loss = loss_function(output, torch.Tensor([l]).long())
        # 反向传播计算梯度
        loss.backward()
        # 获取fc的输入梯度
        input_gradients = model.fc.weight.grad

        # print(input_gradients.shape, last_layer_weights.shape)
        # 计算特征贡献度
        # feature_contribution = torch.abs(torch.matmul(last_layer_weights.T, input_gradients)).mean(dim=0)
        # 将梯度作为特征贡献度
        # feature_contribution = torch.abs(input_gradients.T).mean(dim=0)
        feature_contribution = input_gradients.T.mean(dim=0)
        # 归一化
        # feature_contribution = feature_contribution / feature_contribution.max()
        # print(feature_contribution.shape)
        # feature_contribution的shape为torch.Size([250])，我需要把他拆分成五组，0~49，50~99，100~149，150~199，200~249
        feas_arr = []
        for j in range(5):
            feas_arr.append(feature_contribution[50 * j:50 * (j + 1)].tolist())
        # print(feas_arr)
        # feas_arr按ic把五个值相加
        feas_arr_ = np.array(feas_arr)
        feas_arr_ = feas_arr_.sum(axis=0)
        # 找到feas_arr中，绝对值平均数
        abs_mean = np.abs(feas_arr_).max()
        # print(feas_arr.shape)
        if abs_mean != 0:
            feas_arr_ = feas_arr_ / abs_mean
        # feas_arr = feas_arr.tolist()
        # print(feas_arr.shape)

        # 找到前三个最大的贡献度对于的成分
        # print(type(feas_arr))
        if type(feas_arr_) == list:
            feature_list = feas_arr_
        else:
            feature_list = feas_arr_.tolist()
        tmp = feature_list.copy()
        # 找到最大的三个值对应的索引
        index1 = tmp.index(max(tmp))
        tmp[index1] = -1
        index2 = tmp.index(max(tmp))
        tmp[index2] = -1
        index3 = tmp.index(max(tmp))
        # print(index1, index2, index3)
        print(f"Processing sample {i + 1}/{len(input_FC)}---Contribution1 {index1 + 1}: {feature_list[index1]}, Contribution2 {index2 + 1}: {feature_list[index2]}, Contribution3 {index3 + 1}: {feature_list[index3]}")


        # 添加到特征贡献度列表中
        feature_contributions.append(feature_list)
        # 清除梯度
        model.zero_grad()

    return feature_contributions


def eval(args, input_FC, input_TC, input_Label):

    print("Start evaluating...")

    model = md.my_model(ic=50,tp=140)
    state_dict =torch.load(args.weight, map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(state_dict.keys())
    model.load_state_dict(state_dict)
    # 输出模型结构， 用summary方法
    if args.structure:
        summary(model, input_size=[(50, 140), (50, 50)], batch_size=1, device="cpu")

    feature_contributions = compute_feature_contributions(model, input_FC, input_TC, input_Label)
    '''
    # 打印特征贡献度
    for i, fc in enumerate(feature_contributions):
        print(f"Feature contributions for sample {i + 1}:")
        for j, contribution in enumerate(fc):
            print(f"Feature {j + 1}: {contribution}")
    '''
    feature_mean = np.mean(feature_contributions, axis=0)
    # 归一化
    feature_mean = feature_mean / feature_mean.sum()
    # 按每一个成分取均值
    feature_mean = np.mean(feature_contributions, axis=0)
    # feature_mean = feature_mean / feature_mean.max()
    return feature_mean

# 用折线图展示feature_contributions
def show_feature_contributions(feature_mean):

    # #按每一个成分取均值
    # feature_mean = np.mean(feature_contributions, axis=0)
    # 归一化

    # print(feature_mean.shape)
    # 用折线图展示feature_contributions
    # 长度为50
    plt.figure()
    # 字体放大
    plt.rcParams['font.size'] = 11
    # x轴：Contributions, y轴：Cmoponents
    # 不显示图例
    plt.ylabel('Contribution scores')
    plt.xlabel('Components')

    # plt.title('Feature Contributions')
    # for i in range(len(feature_contributions)):
    #     plt.plot(feature_contributions[i], label=i)
    plt.plot(feature_mean)
    # plt.legend()
    plt.show()

    # [18,21,25,26,27,28,29,30,31,33,34,38,40,41,42,43,44,46,47,48,49,50]
def gene_weighted_sum_map(feature_mean=None):
    # COBRE
    mri_img = nib.load('./datas/_mean_component_ica_s_all_.nii')
    remove = [1,2,3,4,6,8,10,11,12,15,
              16,17,19,23,24,25,26,28,29,31,33,
              35,36,39,40,41,42,47,48,53,57,
              59,61,62,63,71,75,76,77,78,79,
              80,83,89,92,93,94,95,97,98]

    # ABIDE
    # mri_img = nib.load(r'/data/xuruipeng/ABIDE1_Group_ICA/results_TR2_IC100/_mean_component_ica_s_all_.nii')
    # remove = [1,2,3,4,5,6,9,11,12,13,15,
    #           17,21,22,24,26,27,28,29,30,33,34,36,39,40,41,43,46,55,58,61,
    #           64,66,69,70,73,75,76,80,83,84,
    #           86,87,90,91,92,93,98,99,100,
    #           ]

    # for each sub 1
    remove = [i-1 for i in remove]
    ic_data = mri_img.get_fdata()
    ic_data = np.delete(ic_data, remove, axis=3)
    # print(ic_data.shape) # (61, 73, 61, 50)
    # 将feature_mean作为权重，计算加权和
    if feature_mean is None:
        feature_mean = np.ones(50)
    # print(feature_mean.shape)

    new_map = np.zeros((61, 73, 61))
    for i in range(50):
        new_map += ic_data[:, :, :, i] * feature_mean[i]
    # print(new_map.shape)

    # 保存为nii文件
    new_img = nib.Nifti1Image(new_map, mri_img.affine, mri_img.header)
    nib.save(new_img, r'./results/'+args.dataset+'/weighted_sum_map_IC100_s50_'+args.target_label+'.nii')

    plotting.plot_stat_map(new_img, draw_cross=True, colorbar=True, alpha=0.9)
    plotting.show()

if __name__ == '__main__':

    args = configs().parse_args()

    input_FC, input_TC, input_Label = get_data(args)
    feature_contributions = eval(args, input_FC, input_TC, input_Label)
    # print(feature_contributions)
    # feature_contributions *= 1e-1
    show_feature_contributions(feature_contributions)
    gene_weighted_sum_map(feature_contributions)