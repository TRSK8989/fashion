import cv2
import numpy as np
import time
import copy


from torch import nn
from torch.autograd import Variable




def hue_sat_change(origin, background, lower, inner, outer, hue_low, hue_inn, hue_out, sat_low, sat_inn, sat_out, bri_low, bri_inn, bri_out):
    #背景生成
    #origin = cv2.resize(origin, (384, 513))
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