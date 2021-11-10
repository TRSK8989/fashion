import torch
from torch import nn
from torchsummary import summary
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
#from torch.utils.data.dataset import Subset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import cv2
import csv
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm

from models.sence_mobilenet_v2 import SenceMobilenetV2
from datasets.wear import WearDataset




def main():

    #GPUの認識
    device = torch.device("cuda:0")
    #入力画像のチャネル数
    n_channels = 3
    #分類クラスのリスト
    classes = ["formal", "casual"]
    #UNetのフィルタ数の基本単位
    filter_num = 16
    #オプティマイザの学習率
    lr = 5e-4
    #学習時のバッチサイズ
    batch_size = 16
    #入力画像の前処理
    image_preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #データセットが存在するディレクトリのパス指定
    data_source_dir = ""
    #学習エポック数
    num_epochs = 50
    #最良ロスの定義(初期値はinf)
    best_loss = float("inf")
    #実験結果のディレクトリのパス指定
    result_dir = ""
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)



    #UNetの定義
    model = SenceMobilenetV2(n_classes=len(classes), pretrained=True)
    #UNetのパラメータをGPUへ
    model = model.to(device)
    #損失関数の定義
    criterion = nn.CrossEntropyLoss().cuda()
    #オプティマイザの定義
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #スケジューラの定義
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)
    #ネットワーク構成の確認
    #summary(model, (3,240,320))

    #学習用画像と検証用画像の枚数
    n_train_images = 1200
    n_validation_images = 100
    #n_test_images = 100

    #サブセット作成のためのインデックス
    train_subset_indices = list(range(1, n_train_images + 1)) # = [0, 1, 2, ..., n_train_images - 1]
    validation_subset_indices = list(range(n_train_images + 1, n_train_images + n_validation_images + 1)) # = [n_train_images, ..., n_validation_images - 1]
    #test_subset_indices = list(range(n_validation_images + 1, n_validation_images + n_test_images + 1))

    train_dataset = WearDataset(train_subset_indices, classes, data_source_dir=data_source_dir, image_preprocessing=image_preprocessing)
    validation_dataset = WearDataset(validation_subset_indices, classes, data_source_dir=data_source_dir, image_preprocessing=image_preprocessing)



    #データローダーの定義
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    print(len(train_dataloader.dataset))



    for epoch in range(num_epochs):

        print(f"Starting epoch: {epoch}")

        accuracy = 0
        losses = 0.0
        n_total_images = len(train_dataloader.dataset)
        #UNet(特にbatch_norm)をtrainモードへ
        model.train()
        #プログレスバー表示準備
        tbar = tqdm(train_dataloader)
        #オプティマイザの勾配情報を初期化
        optimizer.zero_grad()

        for itr, batch in enumerate(tbar):

            #データローダーの出力タプルを入力画像とラベルに分割
            img, label, name = batch
            print(name)
            #入力画像をGPUへ
            img = img.to(device)
            #ラベルをGPUへ
            label = label.to(device)
            #入力画像をUNetに流して出力を得る(順伝搬)
            output = model(img)
            #出力とラベルから損失を計算
            loss = criterion(output, label)
            #損失から勾配情報を算出(逆伝搬)
            loss.backward()
            #勾配情報からUNetの各層の重みとバイアスを更新
            optimizer.step()
            #オプティマイザの勾配情報を初期化
            optimizer.zero_grad()
            #処理済みのバッチの累計損失を計算
            losses += loss.item()
            #バッチ処理中の平均損失を表示
            tbar.set_description('loss: %.7f' % (losses / (itr + 1)))
            #accuracyを計算
            accuracy += (output.max(1)[1] == label).sum().item()

        #エポック終了時の平均損失の算出
        train_epoch_loss = losses / n_total_images
        train_epoch_acc = accuracy / n_total_images
        print("train_loss :", train_epoch_loss)
        print("train_acc :", train_epoch_acc)


        #勾配情報の算出を省略
        with torch.no_grad():
            accuracy = 0
            losses = 0.0
            n_total_images = len(validation_dataloader.dataset)
            #UNet(特にbatch_norm)をevalモードへ
            model.eval()
            tbar = tqdm(validation_dataloader)

            for itr, batch in enumerate(tbar):

                img, label, name = batch
                print(name)
                img = img.to(device)
                label = label.to(device)
                output = model(img)
                loss = criterion(output, label)
                losses += loss.item()
                tbar.set_description('loss: %.7f' % (losses / (itr + 1)))
                accuracy += (output.max(1)[1] == label).sum().item()

            validation_epoch_loss = losses / n_total_images
            validation_epoch_acc = accuracy / n_total_images
            print("validation_loss :", validation_epoch_loss)
            print("validation_acc :", validation_epoch_acc)
            #スケジューラの更新
            scheduler.step(validation_epoch_loss)



        #CSVファイルへログの書き出し
        with open((os.path.join(result_dir, 'training_log.csv')), 'a') as f:

            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(["epoch", "train_loss", "train_acc", "validation_loss", "validation_acc"])
            writer.writerow([epoch+1, train_epoch_loss, train_epoch_acc, validation_epoch_loss, validation_epoch_acc])



        #ロスが最小のときモデルのパラメータを保存する
        if validation_epoch_loss < best_loss:

            best_loss = validation_epoch_loss
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
            }
            filename = os.path.join(result_dir, "checkpoint.pth.tar")
            torch.save(state, filename)
            print("----------new optimal found! saving state---------- ")







if __name__ == '__main__':
    main()