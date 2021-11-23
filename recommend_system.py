#完成版 

import copy
import cv2
import glob
import numpy as np
import os
import random
import sys
import time
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets
from natsort import natsorted
from PIL import Image

sys.path.append("/home/es1video3_2/デスクトップ/pytorch/py/fashion/recommend/function")

from style_gen import *
from okikae import *
from contrast import *

sys.path.append("/home/es1video3_2/デスクトップ/pytorch/py/fashion/recommend/color_change")

from multi_change import *

sys.path.append("./function")

from DetectColor import *

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def img_to_tensorGPU(img):
    img = cv2pil(img)
    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    img_tensorGPU = img_tensor.to(device)
    return img_tensorGPU

preprocess = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device('cuda')

def texture_convert(file_name):
    t1 = time.time()
    image_name = file_name
    processed_map = contrast_emp(image_name)
    width = 240
    height = 320
    #inner_area = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/inner/" + image_name)
    #inner_area = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/outer/" + image_name)
    inner_area = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/lower/" + image_name)
    inner_area = cv2.resize(inner_area, (width, height))

    right = 0
    left = width
    top = height
    bottom = 0
    for i in range(height):
        for j in range(width):
            if(inner_area[i][j][0] == 255):
                if(right < j):
                    right = j
                elif(left > j):
                    left = j
                elif(top > i):
                    top = i
                elif(bottom < i):
                    bottom = i

    model_cas = torchvision.models.mobilenet_v2(pretrained=True)
    for param in model_cas.parameters(): param.requires_grad = False
    num_features = model_cas.classifier[1].in_features
    model_cas.classifier[1] = nn.Linear(num_features, 2)
    model_cas = model_cas.to(device)
    #model_cas.load_state_dict(torch.load('/home/es1video3_2/デスクトップ/pytorch/py/fashion/check/2020_1105/cas_gray_0.9050.model'))
    model_cas.load_state_dict(torch.load('/home/es1video3_2/デスクトップ/pytorch/py/fashion/check/2020_1124/gara_0.7150.model'))
    model_cas.eval()

    
    #stripe
    
    input_gen = stripe(1, 0, 255)
    input_img = fusion_part(input_gen, left, right, top, bottom, image_name, processed_map)
    input_tensor = img_to_tensorGPU(input_img)

    for i in range(2, 15):
        for j in range(1, 16):
            gen = stripe(i, j * 16, 0)
            gen_img = fusion_part(gen, left, right, top, bottom, image_name, processed_map)
            t2 = time.time()
            gen_img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
            gen_img_gray = cv2.cvtColor(gen_img_gray, cv2.COLOR_GRAY2BGR)
            gen_tensor = img_to_tensorGPU(gen_img_gray)
            input_tensor = torch.cat([input_tensor, gen_tensor], axis = 0)

    input_tensor = input_tensor[1:]
    output1 = model_cas(input_tensor)

    torch.cuda.empty_cache()

    input_gen = stripe(1, 0, 0)
    input_img = fusion_part(input_gen, left, right, top, bottom, image_name, processed_map)
    input_tensor = img_to_tensorGPU(input_img)

    for i in range(2, 15):
        for j in range(1, 16):
            gen = stripe(i, j * 16, 255)
            gen_img = fusion_part(gen, left, right, top, bottom, image_name, processed_map)
            t2 = time.time()
            gen_img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
            gen_img_gray = cv2.cvtColor(gen_img_gray, cv2.COLOR_GRAY2BGR)
            gen_tensor = img_to_tensorGPU(gen_img_gray)
            input_tensor = torch.cat([input_tensor, gen_tensor], axis = 0)

    input_tensor = input_tensor[1:]
    output2 = model_cas(input_tensor)

    t2 = time.time()
    print(t2 - t1)
    output = torch.cat([output1, output2])
    max_index = torch.argmax(output, dim = 0)
    print(output.shape, max_index)

    max_value = max_index.cpu().data.numpy()[0]
    k = int(max_value / 195)
    i = int((max_value - (k * 195)) / 15)
    j = max_value - (k * 195) - (i * 15)
    print(i, j, k)

    result_color = stripe_color(i + 2, (j + 1) * 16, k * 255)
    '''
    #border
    input_gen = border(1, 0, 255)
    input_img = fusion_part(input_gen, left, right, top, bottom, image_name, processed_map)
    input_tensor = img_to_tensorGPU(input_img)

    for i in range(2, 10):
        for j in range(1, 16):
            gen = border(i, j * 16, 0)
            gen_img = fusion_part(gen, left, right, top, bottom, image_name, processed_map)
            t2 = time.time()
            gen_img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
            gen_img_gray = cv2.cvtColor(gen_img_gray, cv2.COLOR_GRAY2BGR)
            gen_tensor = img_to_tensorGPU(gen_img_gray)
            input_tensor = torch.cat([input_tensor, gen_tensor], axis = 0)

    input_tensor = input_tensor[1:]
    output1 = model_cas(input_tensor)

    torch.cuda.empty_cache()

    input_gen = border(1, 0, 0)
    input_img = fusion_part(input_gen, left, right, top, bottom, image_name, processed_map)
    input_tensor = img_to_tensorGPU(input_img)

    for i in range(2, 10):
        for j in range(1, 16):
            gen = border(i, j * 16, 255)
            gen_img = fusion_part(gen, left, right, top, bottom, image_name, processed_map)
            t2 = time.time()
            gen_img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
            gen_img_gray = cv2.cvtColor(gen_img_gray, cv2.COLOR_GRAY2BGR)
            gen_tensor = img_to_tensorGPU(gen_img_gray)
            input_tensor = torch.cat([input_tensor, gen_tensor], axis = 0)

    input_tensor = input_tensor[1:]
    output2 = model_cas(input_tensor)

    t2 = time.time()
    print(t2 - t1)
    output = torch.cat([output1, output2])
    max_index = torch.argmax(output, dim = 0)
    print(output.shape, max_index)

    max_value = max_index.cpu().data.numpy()[0]
    k = int(max_value / 120)
    i = int((max_value - (k * 120)) / 15)
    j = max_value - (k * 120) - (i * 15)
    print(i, j, k)

    result_color = border_color(i + 2, (j + 1) * 16, k * 255)
    
    
    #check
    input_gen = check(1, 0, 255)
    input_img = fusion_part(input_gen, left, right, top, bottom, image_name, processed_map)
    input_tensor = img_to_tensorGPU(input_img)

    for i in range(2, 10):
        for j in range(1, 16):
            gen = check(i, j * 16, 0)
            gen_img = fusion_part(gen, left, right, top, bottom, image_name, processed_map)
            t2 = time.time()
            gen_img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
            gen_img_gray = cv2.cvtColor(gen_img_gray, cv2.COLOR_GRAY2BGR)
            gen_tensor = img_to_tensorGPU(gen_img_gray)
            input_tensor = torch.cat([input_tensor, gen_tensor], axis = 0)

    input_tensor = input_tensor[1:]
    output1 = model_cas(input_tensor)

    torch.cuda.empty_cache()

    input_gen = check(1, 0, 0)
    input_img = fusion_part(input_gen, left, right, top, bottom, image_name, processed_map)
    input_tensor = img_to_tensorGPU(input_img)

    for i in range(2, 10):
        for j in range(1, 16):
            gen = check(i, j * 16, 255)
            gen_img = fusion_part(gen, left, right, top, bottom, image_name, processed_map)
            t2 = time.time()
            gen_img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
            gen_img_gray = cv2.cvtColor(gen_img_gray, cv2.COLOR_GRAY2BGR)
            gen_tensor = img_to_tensorGPU(gen_img_gray)
            input_tensor = torch.cat([input_tensor, gen_tensor], axis = 0)

    input_tensor = input_tensor[1:]
    output2 = model_cas(input_tensor)

    t2 = time.time()
    print(t2 - t1)
    output = torch.cat([output1, output2])
    max_index = torch.argmax(output, dim = 0)
    print(output.shape, max_index)

    max_value = max_index.cpu().data.numpy()[0]
    k = int(max_value / 120)
    i = int((max_value - (k * 120)) / 15)
    j = max_value - (k * 120) - (i * 15)
    print(i, j, k)

    result_color = check_color(i + 2, (j + 1) * 16, k * 255)
    '''
    #ここまで

    result_img = fusion_part(result_color, left, right, top, bottom, image_name, processed_map)

    result_hsv = cv2.cvtColor(result_img, cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv_" + image_name, result_hsv)

    path = "./hsv_" + image_name
    image = cv2.imread(path)
    image = cv2.resize(image, (240, 320))

    for i in range(200):
        cv2.imwrite("./train_image/" + str(i) + ".jpeg", image)

    torch.cuda.empty_cache()

def recommend(file_name):
    #学習用の画像が保存されたディレクトリ
    train_image_path = "./train_image/*"
    train_image = glob.glob(train_image_path)
    train_image = natsorted(train_image)

    #transformの設定
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    #datasetの設定
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, image_path, transform=None):
            self.transform = transform
            self.image_path = image_path
            self.data_num = len(image_path)

        def __len__(self):
            return self.data_num

        def __getitem__(self, idx):
            image = cv2.imread(self.image_path[idx], cv2.IMREAD_UNCHANGED)

            if self.transform:
                out_image = self.transform(image)
            else:
                out_image = image[idx]

            return out_image

    #trainloaderの設定
    BATCH_SIZE = 1
    train_dataset = MyDataset(train_image, transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model_hsv = torchvision.models.mobilenet_v2(pretrained=True)
    for param in model_hsv.parameters(): param.requires_grad = False
    num_features = model_hsv.classifier[1].in_features
    model_hsv.classifier[1] = nn.Linear(num_features, 2)
    device = torch.device('cuda')
    model_hsv = model_hsv.to(device)
    model_hsv.load_state_dict(torch.load('/home/es1video3_2/デスクトップ/pytorch/py/fashion/check/0525/hsv_0.8450.model'))
    #model_hsv.load_state_dict(torch.load('/home/es1video3_2/デスクトップ/pytorch/py/fashion/check/2020_1122/gold_0.9625.model'))
    model_hsv.eval()

    model_sat = torchvision.models.mobilenet_v2(pretrained=True)
    for param in model_sat.parameters(): param.requires_grad = False
    num_features = model_sat.classifier[1].in_features
    model_sat.classifier[1] = nn.Linear(num_features, 2)
    model_sat = model_sat.to(device)
    model_sat.load_state_dict(torch.load('/home/es1video3_2/デスクトップ/pytorch/py/fashion/check/0608/sat_0.8158.model'))
    model_sat.eval()

    model_hue = torchvision.models.mobilenet_v2(pretrained=True)
    for param in model_hue.parameters(): param.requires_grad = False
    num_features = model_hue.classifier[1].in_features
    model_hue.classifier[1] = nn.Linear(num_features, 2)
    model_hue = model_hue.to(device)
    model_hue.load_state_dict(torch.load('/home/es1video3_2/デスクトップ/pytorch/py/fashion/check/0608/hue_0.7563.model'))
    model_hue.eval()

    name = file_name
    mse = torch.nn.MSELoss()

    origin_path = "/home/es1video3_2/デスクトップ/pytorch/data/fashion/jpeg/" + name
    origin = cv2.imread(origin_path)
    origin_resize = cv2.resize(origin, (240, 320))
    origin = cv2.resize(origin, (384, 513))
    seg_path = "/home/es1video3_2/デスクトップ/pytorch/data/fashion/seg/" + name
    background = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/background/" + name)
    lower = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/lower/" + name)
    inner = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/inner/" + name)
    outer = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/outer/" + name)
    lower_resize = cv2.resize(lower, (240, 320))
    inner_resize = cv2.resize(inner, (240, 320))
    outer_resize = cv2.resize(outer, (240, 320))

    lower_mask = lower_resize[:, :, 0] / 255
    lower_tensor = torch.from_numpy(lower_mask).cuda().to(torch.float)
    inner_mask = inner_resize[:, :, 0] / 255
    inner_tensor = torch.from_numpy(inner_mask).cuda().to(torch.float)
    outer_mask = outer_resize[:, :, 0] / 255
    outer_tensor = torch.from_numpy(outer_mask).cuda().to(torch.float)

    hsv_origin = cv2.imread("./train_image/1.jpeg")
    input_tensor = transform(hsv_origin).cuda()

    #colorImage = "/home/es1video3_2/デスクトップ/pytorch/data/fashion/male/train/0/" + name
    #LH, LS, LV, IH, IS, IV, OH, OS, OV = Detect(colorImage)
    #LH, IH, OH = LH / 255, IH / 255, OH / 255

    def MyLoss(outputs, inputs):
        sigmoid = nn.Sigmoid()
        hardtanh = nn.Hardtanh(-1.0, 1.0)
        hardtanh_hue = nn.Hardtanh(0.0, 0.705)
        hardtanh_sat = nn.Hardtanh(0.0, 1.0)
        batch_loss = 0.0
        for i in range(len(outputs)):
            output = outputs[i].clone()
            input_clone = inputs[i].clone()
            sat_tensor = inputs[i].clone()
            hue_tensor = inputs[i].clone()
            add_img_hue = input_clone[0] + (lower_tensor * hardtanh(output[0] / 10)) + (inner_tensor * hardtanh(output[1] / 10)) + (outer_tensor * hardtanh(output[2] / 10))
            input_clone[0] = hardtanh_hue(add_img_hue)
            add_img_sat = input_clone[1] + (lower_tensor * hardtanh(output[3] / 10)) + (inner_tensor * hardtanh(output[4] / 10)) + (outer_tensor * hardtanh(output[5] / 10))
            input_clone[1] = hardtanh_sat(add_img_sat)
            sig_hsv_loss = sigmoid(model_hsv(input_clone.unsqueeze(0)))
            hsv_loss = 1 - sig_hsv_loss[0][1]
            hue_tensor[0] = hardtanh_hue(add_img_hue)
            hue_tensor[1] = hardtanh_hue(add_img_hue)
            hue_tensor[2] = hardtanh_hue(add_img_hue)
            sig_hue_loss = sigmoid(model_hue(hue_tensor.unsqueeze(0)))
            hue_loss = 1 - sig_hue_loss[0][1]
            sat_tensor[0] = hardtanh_sat(add_img_sat)
            sat_tensor[1] = hardtanh_sat(add_img_sat)
            sat_tensor[2] = hardtanh_sat(add_img_sat)
            sig_sat_loss = sigmoid(model_sat(sat_tensor.unsqueeze(0)))
            sat_loss = 1 - sig_sat_loss[0][1]
            total_loss = hsv_loss + hue_loss + sat_loss
            batch_loss += total_loss

        return batch_loss

    def train_visualize(output, input_img, epoch):
        hardtanh = nn.Hardtanh(-1.0, 1.0)
        hardtanh_hue = nn.Hardtanh(0.0, 0.705)
        hardtanh_sat = nn.Hardtanh(0.0, 1.0)
        input_clone = input_tensor.clone()
        add_img_hue = input_clone[0] + (lower_tensor * hardtanh(output[0] / 10)) + (inner_tensor * hardtanh(output[1] / 10)) + (outer_tensor * hardtanh(output[2] / 10))
        input_clone[0] = hardtanh_hue(add_img_hue)
        add_img_sat = input_clone[1] + (lower_tensor * hardtanh(output[3] / 10)) + (inner_tensor * hardtanh(output[4] / 10)) + (outer_tensor * hardtanh(output[5] / 10))
        input_clone[1] = hardtanh_sat(add_img_sat)
        out_np = input_clone.cpu().data.numpy()
        out_img = out_np * 255
        out_img = out_img.transpose(2, 1, 0)
        out_img = out_img.transpose(1, 0, 2)
        out_img = out_img.astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2BGR)

        con_img = cv2.hconcat([origin_resize, out_img])
        
        if(epoch == 2):
            cv2.imwrite("./result/" + name, con_img)

    #GPU
    use_gpu = torch.cuda.is_available()
    def train_model(model, optimizer, scheduler, num_epochs=50):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 100000.0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            for phase in ['train', 'valid']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)

                    running_loss = 0.0

                    for i, data in enumerate(trainloader):
                        inputs = data

                        if use_gpu:
                            inputs = Variable(inputs.cuda())
                        else:
                            inputs = Variable(inputs)

                        optimizer.zero_grad()

                        outputs = model(inputs)

                        loss = MyLoss(outputs, inputs)

                        loss.backward()
                        optimizer.step()
                        running_loss += loss.data

                    epoch_loss = running_loss / len(train_image)

                    print('{} Loss: {} '.format(phase, epoch_loss))
                    print(outputs[0])
                    train_visualize(outputs[0], inputs[0], epoch)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        #torch.save(model.state_dict(), 'check_' + str(best_loss) + '.model')

                else:
                    print("test is null")

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:.4f}'.format(best_loss))

        #load best model weights
        model.load_state_dict(best_model_wts)
        return model, outputs

    model_conv = torchvision.models.mobilenet_v2(pretrained=True)
    for param in model_conv.parameters(): param.requires_grad = False
    num_features = model_conv.classifier[1].in_features
    model_conv.classifier[1] = nn.Linear(num_features, 6)

    if use_gpu: model_conv = model_conv.cuda()
    optimizer_conv = optim.SGD(model_conv.classifier[1].parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)
    model_conv, outputs = train_model(model_conv, optimizer_conv, exp_lr_scheduler, num_epochs=4)

    origin_path = "/home/es1video3_2/デスクトップ/pytorch/data/fashion/jpeg/" + name
    origin = cv2.imread(origin_path)
    origin_resize = cv2.resize(origin, (240, 320))
    origin = cv2.resize(origin, (384, 513))
    seg_path = "/home/es1video3_2/デスクトップ/pytorch/data/fashion/seg/" + name
    background = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/background/" + name)
    lower = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/lower/" + name)
    inner = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/inner/" + name)
    outer = cv2.imread("/home/es1video3_2/デスクトップ/pytorch/data/fashion/colorization/each_parts/outer/" + name)

    #colorImage = "/home/es1video3_2/デスクトップ/pytorch/data/fashion/jpeg/" + name
    #LH, LS, LV, IH, IS, IV, OH, OS, OV = Detect(colorImage)

    hue_low = 0
    hue_inn = 0
    hue_out = 0
    sat_low = 0
    sat_inn = 0
    sat_out = 0
    bri_low = 0
    bri_inn = 0
    bri_out = 0

    def hue_sat_change(origin, background, lower, inner, outer, hue_low, hue_inn, hue_out, sat_low, sat_inn, sat_out, bri_low, bri_inn, bri_out):
        #背景生成
        origin = cv2.resize(origin, (384, 513))
        back = cv2.bitwise_and(origin, background)

        #BGR空間からHSV空間に変換
        hsv_img = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)

        #int16に
        h_img = hsv_img[:, :, 0].astype(np.int16)
        s_img = hsv_img[:, :, 1].astype(np.int16)
        b_img = hsv_img[:, :, 2].astype(np.int16)

        #色相を変化
        h_low = h_img + hue_low
        h_inn = h_img + hue_inn
        h_out = h_img + hue_out
        h_low = np.where(h_low > 180, 180, h_low)
        h_low = np.where(h_low < 0, 0, h_low)
        h_inn = np.where(h_inn > 180, 180, h_inn)
        h_inn = np.where(h_inn < 0, 0, h_inn)
        h_out = np.where(h_out > 180, 180, h_out)
        h_out = np.where(h_out < 0, 0, h_out)

        #彩度を変化
        s_low = s_img + sat_low
        s_inn = s_img + sat_inn
        s_out = s_img + sat_out
        s_low = np.where(s_low > 255, 255, s_low)
        s_low = np.where(s_low < 0, 0, s_low)
        s_inn = np.where(s_inn > 255, 255, s_inn)
        s_inn = np.where(s_inn < 0, 0, s_inn)
        s_out = np.where(s_out > 255, 255, s_out)
        s_out = np.where(s_out < 0, 0, s_out)

        #明度を変化
        b_low = b_img + bri_low
        b_inn = b_img + bri_inn
        b_out = b_img + bri_out
        b_low = np.where(b_low > 255, 255, b_low)
        b_low = np.where(b_low < 0, 0, b_low)
        b_inn = np.where(b_inn > 255, 255, b_inn)
        b_inn = np.where(b_inn < 0, 0, b_inn)
        b_out = np.where(b_out > 255, 255, b_out)
        b_out = np.where(b_out < 0, 0, b_out)

        #bgr空間に変換
        hsv_img[:, :, 0] = h_low
        hsv_img[:, :, 1] = s_low
        hsv_img[:, :, 2] = b_low
        bgr_low = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        hsv_img[:, :, 0] = h_inn
        hsv_img[:, :, 1] = s_inn
        hsv_img[:, :, 2] = b_inn
        bgr_inn = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        hsv_img[:, :, 0] = h_out
        hsv_img[:, :, 1] = s_out
        hsv_img[:, :, 2] = b_out
        bgr_out = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        #マスク
        low = cv2.bitwise_and(bgr_low, lower)
        inn = cv2.bitwise_and(bgr_inn, inner)
        out = cv2.bitwise_and(bgr_out, outer)

        #複数領域の色相を変化した画像を生成
        change_img = back + low + inn + out

        #色検出
        h_low = h_low.astype(np.uint8)
        s_low = s_low.astype(np.uint8)
        b_low = b_low.astype(np.uint8)
        h_inn = h_inn.astype(np.uint8)
        s_inn = s_inn.astype(np.uint8)
        b_inn = b_inn.astype(np.uint8)
        h_out = h_out.astype(np.uint8)
        s_out = s_out.astype(np.uint8)
        b_out = b_out.astype(np.uint8)
        h_low = cv2.bitwise_and(h_low, lower[:, :, 0])
        h_total_low = h_low.sum()
        h_low_value = h_total_low / cv2.countNonZero(h_low)
        s_low = cv2.bitwise_and(s_low, lower[:, :, 0])
        s_total_low = s_low.sum()
        s_low_value = s_total_low / cv2.countNonZero(s_low)
        b_low = cv2.bitwise_and(b_low, lower[:, :, 0])
        b_total_low = b_low.sum()
        b_low_value = b_total_low / cv2.countNonZero(b_low)
        h_inn = cv2.bitwise_and(h_inn, inner[:, :, 0])
        h_total_inn = h_inn.sum()
        h_inn_value = h_total_inn / cv2.countNonZero(h_inn)
        s_inn = cv2.bitwise_and(s_inn, inner[:, :, 0])
        s_total_inn = s_inn.sum()
        s_inn_value = s_total_inn / cv2.countNonZero(s_inn)
        b_inn = cv2.bitwise_and(b_inn, inner[:, :, 0])
        b_total_inn = b_inn.sum()
        b_inn_value = b_total_inn / cv2.countNonZero(b_inn)
        h_out = cv2.bitwise_and(h_out, outer[:, :, 0])
        h_total_out = h_out.sum()
        h_out_value = h_total_out / cv2.countNonZero(h_out)
        s_out = cv2.bitwise_and(s_out, outer[:, :, 0])
        s_total_out = s_out.sum()
        s_out_value = s_total_out / cv2.countNonZero(s_out)
        b_out = cv2.bitwise_and(b_out, outer[:, :, 0])
        b_total_out = b_out.sum()
        b_out_value = b_total_out / cv2.countNonZero(b_out)

        print(h_low_value, s_low_value, b_low_value)
        print(h_inn_value, s_inn_value, b_inn_value)
        print(h_out_value, s_out_value, b_out_value)

        #評価用
        return change_img

    origin_img = hue_sat_change(origin, background, lower, inner, outer, hue_low, hue_inn, hue_out, sat_low, sat_inn, sat_out, bri_low, bri_inn, bri_out)

    output_numpy = outputs.cpu().data.numpy()
    on = output_numpy[0]
    hue_low = int(on[0] / 10 * 255)
    hue_inn = int(on[1] / 10 * 255)
    hue_out = int(on[2] / 10 * 255)
    sat_low = int(on[3] / 10 * 255)
    sat_inn = int(on[4] / 10 * 255)
    sat_out = int(on[5] / 10 * 255)
    bri_low = 0
    bri_inn = 0
    bri_out = 0

    change_img = hue_sat_change(origin, background, lower, inner, outer, hue_low, hue_inn, hue_out, sat_low, sat_inn, sat_out, bri_low, bri_inn, bri_out)

    con_image = cv2.hconcat([origin_img, change_img])

    torch.cuda.empty_cache()
    #cv2.imwrite("./input_output/4190_21.jpeg", con_image)

    #cv2.imshow("change_img", con_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

file_list = glob.glob("/home/es1video3_2/デスクトップ/pytorch/data/fashion/jpeg/*")
file_list = natsorted(file_list)
file_list = file_list[:250]
for i in range(len(file_list)):
    file_path = file_list[i]
    file_name = file_path.split('/')[-1]
    texture_convert(file_name)
    recommend(file_name)



