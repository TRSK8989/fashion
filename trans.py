import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import glob

import torch
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from loss.myloss import MyLoss



from models.sence_mobilenet_v2 import SenceMobilenetV2
from datasets.mydataset import MyDatasetPerImage

seed=1024
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True



def train_visualize(output_dir, output, input_img, epoch, ori_resize, lower_tensor, inner_tensor, outer_tensor):
    hardtanh = nn.Hardtanh(-1.0, 1.0)
    hardtanh_hue = nn.Hardtanh(0.0, 0.705)
    hardtanh_sat = nn.Hardtanh(0.0, 1.0)
    hardtanh_val = nn.Hardtanh(0.0, 1.0)
    input_clone = input_img.clone()
    add_img_hue = input_clone[0] + (lower_tensor * hardtanh(output[0] / 10)) + (inner_tensor * hardtanh(output[1] / 10)) + (outer_tensor * hardtanh(output[2] / 10))
    input_clone[0] = hardtanh_hue(add_img_hue)
    add_img_sat = input_clone[1] + (lower_tensor * hardtanh(output[3] / 10)) + (inner_tensor * hardtanh(output[4] / 10)) + (outer_tensor * hardtanh(output[5] / 10))
    input_clone[1] = hardtanh_sat(add_img_sat)
    add_img_val = input_clone[2] + (lower_tensor * hardtanh(output[6] / 10)) + (inner_tensor * hardtanh(output[7] / 10)) + (outer_tensor * hardtanh(output[8] / 10))
    input_clone[2] = hardtanh_val(add_img_val)
    out_np = input_clone.cpu().data.numpy()
    out_img = out_np * 255
    out_img = out_img.transpose(2, 1, 0)
    out_img = out_img.transpose(1, 0, 2)
    out_img = out_img.astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2BGR)

    con_img = cv2.hconcat([ori_resize, out_img])

    #if(epoch == 2):
    cv2.imwrite(output_dir + "/epoch{}.jpeg".format(epoch+1), con_img)




def main():


    #各種パラメータ設定
    source_dir = "/home/es1video10/datasets/fashion/muraoka_made/"
    check_path = "/home/es1video10/pytorch_envs/ad/recon/TRSK8989/fashion/result/checkpoint.pth.tar"
    output_dir = "/home/es1video10/pytorch_envs/ad/recon/TRSK8989/fashion/result/output"
    img_name = "5_796.jpeg"
    width = 240
    height = 320
    classes = ["formal", "casual"]
    num_epochs = 50
    lr = 5e-4
    device = torch.device("cuda:0")



    #変換する画像の設定（RGB・マスク画像）
    ori = cv2.imread(source_dir + "original/" + img_name)
    back = cv2.imread(source_dir + "parts/background/" + img_name)# // 255
    inner = cv2.imread(source_dir + "parts/inner/" + img_name)# // 255
    lower = cv2.imread(source_dir + "parts/lower/" + img_name)# // 255
    outer = cv2.imread(source_dir + "parts/outer/" + img_name)# // 255

    ori_resize = cv2.resize(ori, (width, height))
    back_resize = cv2.resize(back, (width, height))
    inner_resize = cv2.resize(inner, (width, height))
    lower_resize = cv2.resize(lower, (width, height))
    outer_resize = cv2.resize(outer, (width, height))

    back_mask = back_resize#[:, :, 0]
    lower_mask = lower_resize[:, :, 0]
    lower_tensor = torch.from_numpy(lower_mask).cuda().to(torch.float) / 255
    inner_mask = inner_resize[:, :, 0]
    inner_tensor = torch.from_numpy(inner_mask).cuda().to(torch.float) / 255
    outer_mask = outer_resize[:, :, 0]
    outer_tensor = torch.from_numpy(outer_mask).cuda().to(torch.float) / 255


    #モデルの設定
    net_t = SenceMobilenetV2(n_classes=9, pretrained=True)
    net_c = SenceMobilenetV2(n_classes=len(classes), pretrained=False)
    net_t = net_t.to(device)
    net_c = net_c.to(device).eval()

    checkpoint = torch.load(check_path)
    net_c.load_state_dict(checkpoint["state_dict"])


    #学習データに関する設定（HSV画像）
    transform = transforms.Compose(
            [transforms.ToTensor()]
        )
    #train_imgs = glob.glob("/home/es1video10/datasets/fashion/muraoka_made/original/*")
    #dataset = MyDataset(train_imgs, transform, classes, "casual")
    train_imgs = ["/home/es1video10/datasets/fashion/muraoka_made/original/" + img_name]
    dataset = MyDatasetPerImage(train_imgs, transform, classes, "casual")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)


    #損失・オプティマイザ・スケジューラ
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer_t = optim.Adam(net_t.parameters(), lr=lr)
    exp_lr_scheduler = StepLR(optimizer_t, step_size=10, gamma=0.1)


    #ここから色変換開始
    since = time.time()
    best_model_wts = copy.deepcopy(net_t.state_dict())
    best_loss = 100000.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                exp_lr_scheduler.step()
                net_t.train(True)

                running_loss = 0.0

                for i, (data, label) in enumerate(dataloader):
                    inputs = data
                    label = label.to(device)

                    inputs = Variable(inputs.cuda())

                    optimizer_t.zero_grad()

                    outputs = net_t(inputs)

                    loss = MyLoss(outputs, inputs, label, net_c, criterion, lower_tensor, inner_tensor, outer_tensor)

                    loss.backward()
                    optimizer_t.step()
                    running_loss += loss.data

                epoch_loss = running_loss / len(train_imgs)

                print('{} Loss: {} '.format(phase, epoch_loss))
                print(outputs[0])
                train_visualize(output_dir, outputs[0], inputs[0], epoch, ori_resize, lower_tensor, inner_tensor, outer_tensor)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(net_t.state_dict())
                    #torch.save(model.state_dict(), 'check_' + str(best_loss) + '.model')

            else:
                print("test is null")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:.4f}'.format(best_loss))

    #load best model weights
    net_t.load_state_dict(best_model_wts)


if __name__ == '__main__':
    main()