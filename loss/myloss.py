import torch.nn as nn



def MyLoss(outputs, inputs, label, net_c, criterion, lower_tensor, inner_tensor, outer_tensor):
    sigmoid = nn.Sigmoid()
    hardtanh = nn.Hardtanh(-1.0, 1.0)
    hardtanh_hue = nn.Hardtanh(0.0, 0.705)
    hardtanh_sat = nn.Hardtanh(0.0, 1.0)
    hardtanh_val = nn.Hardtanh(0.0, 1.0)
    batch_loss = 0.0
    for i in range(len(outputs)):
        output = outputs[i].clone()
        input_clone = inputs[i].clone()
        sat_tensor = inputs[i].clone()
        hue_tensor = inputs[i].clone()
        val_tensor = inputs[i].clone()
        
        add_img_hue = input_clone[0] + (lower_tensor * hardtanh(output[0] / 10)) + (inner_tensor * hardtanh(output[1] / 10)) + (outer_tensor * hardtanh(output[2] / 10))
        input_clone[0] = hardtanh_hue(add_img_hue)
        add_img_sat = input_clone[1] + (lower_tensor * hardtanh(output[3] / 10)) + (inner_tensor * hardtanh(output[4] / 10)) + (outer_tensor * hardtanh(output[5] / 10))
        input_clone[1] = hardtanh_sat(add_img_sat)
        add_img_val = input_clone[2] + (lower_tensor * hardtanh(output[6] / 10)) + (inner_tensor * hardtanh(output[7] / 10)) + (outer_tensor * hardtanh(output[8] / 10))
        input_clone[2] = hardtanh_val(add_img_val)
        
        class_output = net_c(input_clone.unsqueeze(0))
        class_loss = criterion(class_output, label)
        batch_loss += class_loss
        
    return batch_loss