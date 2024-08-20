import csv
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import loadData
import model as md
import time
from tqdm import tqdm
import torch.optim as optim
from torchsummary import summary
from fuctionUtility import acc_pre_recall_f, EarlyStopping
import torch.nn.functional as F
from prettytable import PrettyTable
import argparse

def parse_args():
    """ Parsing and configuration """
    parser = argparse.ArgumentParser(description="DP-FP")
    parser.add_argument('--dataset', type=str, default="COBRE", help='COBRE, ABIDE1, so on')
    parser.add_argument('--dataset_path', type=str, default="./datas/COBRE.mat", help='in ./datas')
    parser.add_argument('--epochs', type=int, default=1000, help='The number of epochs to run')
    parser.add_argument('--early_stopping', type=int, default=40, help='The number of early stopping')
    parser.add_argument('--val_step', type=int, default=1, help='The number of val step')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    return parser.parse_args()

args = parse_args()

D = args.dataset
if D == "COBRE":
    dim = 50
    tp = 140
elif D == "ABIDE1":
    dim = 50
    tp = 100

ttb = PrettyTable(['Fold', 'ACC', 'SPE', 'SEN', 'F1', 'AUC'])
acctmplist = []
spetmplist = []
sentmplist = []
f1tmplist = []
roctmplist = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mytime = (time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
# torch.cuda.set_device(0)

def l1_regularization(model, l1_alpha):
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            module.weight.grad.data.add_(l1_alpha * torch.sign(module.weight.data))

def l2_regularization(model, l2_alpha):
    for module in model.modules():
        if type(module) is nn.Conv2d:
            module.weight.grad.data.add_(l2_alpha * module.weight.data)

def save_logit_logs(label, logits, flag):

    save_path = "./results/test_logits.csv"
    if flag:
        save_path = "./results/test_logits_last_epoch.csv"

    with open(save_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow([label, logits[0], logits[1]])

def train(train_loader, val_loader, ki=None ,pretrain=None):

    net = md.my_model(ic=dim, tp=tp)
    net.to(device)

    early_stopping = EarlyStopping("./weights", args.early_stopping, True)

    epochs = args.epochs

    train_steps = len(train_loader)
    val_steps = 1

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    loss_function = nn.CrossEntropyLoss()

    # three different learning rate scheduler
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0, last_epoch=-1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    if pretrain is not None:
        print(torch.cuda.current_device())
        pretrained_dict = torch.load(pretrain)['state_dict']

        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # update model_dict
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    val_acc = 0.0
    val_loss = 9e+9
    # train fnc_path
    for epoch in range(epochs):

        features = np.zeros((1, dim))
        label = []
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        scheduler.step()
        optimizer.zero_grad()
        for step, data in enumerate(train_bar):
            labels, fnc, tcs = data

            labels = labels.long().to(device)
            tcs = tcs.float().to(device)
            fnc = fnc.float().to(device)
            # print(tcs.shape, fnc.shape)
            logitsfnc,_ = net(tcs, fnc)
            lossfnc = loss_function(logitsfnc, labels)
            running_loss += lossfnc.item()

            features = np.append(features, _.cpu().detach().numpy(), axis=0)
            labels = labels.tolist()
            label.extend(labels)
            lossfnc.backward()

            l1_regularization(net, l1_alpha=0.05)
            l2_regularization(net, l2_alpha=0.1)

            optimizer.step()

            train_bar.desc = "Fold[{}] Train epoch[{}/{}]".format(ki+1, epoch + 1, epochs)
            train_bar.set_postfix(lr=scheduler.get_lr()[0], loss=running_loss / train_steps)
        features = np.delete(features, 0, 0)
        # validate
        net.eval()
        acc = 0.0
        runValloss = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, colour='green')

            for val_data in val_bar:
                val_labels, val_fnc, val_tcs = val_data

                val_labels = val_labels.long().to(device)
                val_tcs = val_tcs.float().to(device)
                val_fnc = val_fnc.float().to(device)

                outputs,_ = net(val_tcs, val_fnc)
                outputs = F.softmax(outputs, dim=1)

                vallosstc = loss_function(outputs, val_labels)

                runValloss += vallosstc.item()

                tmp = torch.max(outputs, dim=1)
                predict_y = tmp[1]

                acc += torch.eq(predict_y, val_labels).sum().item()

                val_bar.desc = "Fold[{}] Valid True[{}] ".format(ki+1, acc)
                val_bar.set_postfix(val_loss=runValloss / val_steps, val_acc=acc / val_steps)

            early_stopping(runValloss / val_steps - acc / val_steps, net)

            if early_stopping.early_stop or epoch+1 == epochs:
                features_train = features
                print("Early stopping")
                break

    torch.save(net.state_dict(), "weights/model_last.pth")
    print("Save model_last.pth")
    return features_train, np.array(label)

def LR(data, label, data_train, label_train):

    epochs = 1000

    labels = torch.Tensor(label_train)
    datas = torch.Tensor(data_train)
    labels_test = torch.Tensor(label)
    datas_test = torch.Tensor(data)

    net = md.LogisticRegression(dim)
    net.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1/epochs)

    # try:
    #     os.path.exists(model_weight_path)
    #     missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
    # except OSError as reason:
    #     print("file {} does not exist.".format(model_weight_path))

    train_bar = tqdm(range(epochs))
    i = 0
    train_bar.desc = "LR train"

    best_acc = 0.0
    save_path = './results/lr_best.pth'
    Minloss = 9e+9
    for _ in train_bar:
        net.train()
        out = net(datas.to(device))
        loss = criterion(out.squeeze(1), labels.to(device))
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        net.eval()
        out = net(datas_test.to(device))
        submittion = out.cpu().ge(0.5).float().squeeze(1)

        predict = submittion == labels_test
        acc = predict.sum().float() / datas_test.size()[0]

        train_bar.desc = "acc: {}".format(acc)
        if loss.item() < Minloss:
            torch.save(net.state_dict(), save_path)
            Minloss = loss.item()

    net.eval()
    labels = torch.Tensor(label)
    datas = torch.Tensor(data)
    net.load_state_dict(torch.load(save_path), strict=False)

    with torch.no_grad():
        out = net(datas.to(device))
    submittion = out.cpu().ge(0.5).float().squeeze(1)
    predict = submittion == labels
    acc = predict.sum().float() / datas.size()[0]
    train_bar.set_postfix(acc=acc.item())
    prob = torch.max(out)

    return submittion.numpy(), out.cpu().float().squeeze(1).detach().numpy()

def save_feature(feature, label):
    features = []
    for i in range(len(feature)):
        cat = {"features": feature[i], "label": label[i]}
        features.append(cat)
    json_file = "./feature_cls.json"
    json_fp = open(json_file, 'w')
    json_str = json.dumps(feature, indent=2)
    json_fp.write(json_str)
    json_fp.close()


def test(test_loader, flag=0, ki=None):

    predicets = []
    scores = []
    label = []

    features = np.zeros((1,dim))

    net = md.my_model(ic=dim, tp=tp)
    net.eval()
    sm = nn.Softmax(1)
    net.to(device)
    if flag:
        net.load_state_dict(torch.load('weights/model_last.pth'), strict=False)
    else:
        net.load_state_dict(torch.load('weights/best_network.pth'), strict=False)

    test_bar = tqdm(test_loader, colour='yellow')

    for step, data in enumerate(test_bar):

        labels, fnc, tcs = data

        tcs = tcs.float().to(device)
        fnc = fnc.float().to(device)

        with torch.no_grad():

            logits, feature = net(tcs, fnc)
            logits = sm(logits)

        features = np.append(features, feature.cpu().numpy(), axis=0)
        test_bar.desc = "Test"

        tmp = torch.max(logits, dim=1)

        score = logits[:,1]
        predict_y = tmp[1]
        predict_y = predict_y.cpu().tolist()
        score = score.cpu().tolist()
        labels = labels.tolist()

        # save_logit_logs(labels[0], logits.cpu().tolist()[0], flag)

        predicets.append(predict_y)
        scores.append(score)
        label.append(labels)

    # print(predicets,scores,label)
    features = np.delete(features,0,0)
    acc, spe, sen, f1, roc_auc = acc_pre_recall_f(np.array(label), np.array(predicets), np.array(scores))


    print("________________________________________________________________")
    print("acc:{} spe:{} sen:{} f1:{} roc_auc:{}".format(acc, spe, sen, f1, roc_auc))
    print("________________________________________________________________")
    ttb.add_row([ki, acc, spe, sen, f1, roc_auc])
    acctmplist.append(acc)
    spetmplist.append(spe)
    sentmplist.append(sen)
    f1tmplist.append(f1)
    roctmplist.append(roc_auc)
    return np.array(label), np.array(predicets), np.array(scores), features

def run():

    print("Using {} device.".format(device))

    datapath = args.dataset_path

    pretrained = None

    y = np.array([[0]])
    pre = np.array([[0]])
    y_score = np.array([[0]])

    featurejson = []
    K = 10  # 10折交叉验证

    for ki in range(K):
        print('Fold', ki + 1)

        trainset = loadData.Dataset2BothModel(sfc_path=datapath, tc_path=datapath, ki=ki, K=K, typ='train')
        valset = loadData.Dataset2BothModel(sfc_path=datapath, tc_path=datapath, ki=ki, K=K, typ='val')
        # print(len(trainset), len(valset))
        train_loader = DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True)

        val_loader = DataLoader(
            dataset=valset,
            batch_size=32,
        )

        test_loader = DataLoader(
            dataset=valset,
            batch_size=1,
        )

        model = md.my_model(ic=dim, tp=tp)
        if ki == 0:
            summary(model, input_size=[(dim, tp),(dim, dim)], batch_size=16, device="cpu")


        features_train, label_train = train(train_loader, val_loader, ki, pretrained)
        label, predicets, scores, features = test(test_loader,ki=ki)

        y = np.append(y, label, axis=0)
        pre = np.append(pre, predicets, axis=0)
        y_score = np.append(y_score, scores, axis=0)

        for i in range(len(features)):
            cat = {"features": [features[i].tolist()], "label": label[i].tolist()[0]}
            featurejson.append(cat)

    # json_file = "./feature_cls_600.json"
    # json_fp = open(json_file, 'w')
    # json_str = json.dumps(featurejson, indent=2)
    # json_fp.write(json_str)
    # json_fp.close()

    y = np.delete(y, 0, 0)
    pre = np.delete(pre, 0, 0)
    y_score = np.delete(y_score, 0, 0)

    acc, spe, sen, f1, roc_auc = acc_pre_recall_f(y, pre, y_score)
    print("\n\n\n")

    print("*****************************************************************")
    print("Finished!")
    print("Performance of DP-FP.")
    print("acc:{} spe:{} sen:{} f1:{} roc_auc:{}".format(acc, spe, sen, f1, roc_auc))
    print("*****************************************************************")


    # ['Fold', 'ACC', 'SPE', 'SEN', 'F1', 'AUC']
    # compute the mean and std of the ttb

    acc_mean = np.mean(acctmplist)
    acc_std = np.std(acctmplist)
    spe_mean = np.mean(spetmplist)
    spe_std = np.std(spetmplist)
    sen_mean = np.mean(sentmplist)
    sen_std = np.std(sentmplist)
    f1_mean = np.mean(f1tmplist)
    f1_std = np.std(f1tmplist)
    roc_auc_mean = np.mean(roctmplist)
    roc_auc_std = np.std(roctmplist)

    ttb.add_row(['mean', acc_mean, spe_mean, sen_mean, f1_mean, roc_auc_mean])
    ttb.add_row(['std', acc_std, spe_std, sen_std, f1_std, roc_auc_std])

    print(ttb)

    # save the ttb
    # with open('results/ABIDE1_TR2_IC100_s50_ours.txt', 'w') as f:
    #     f.write(str(ttb))

if __name__ == '__main__':

    run()