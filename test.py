import argparse
import json

from torch.utils.data import DataLoader

from models import *
# from utils.datasets import *
from utils.utils import *

from mymodel import *

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
 
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def get_boxes(pred_boxes, pred_conf):
    n = pred_boxes.size(0)
    # pred_boxes = pred_boxes.view(n, -1, 4)
    # pred_conf = pred_conf.view(n, -1, 1)
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    p_boxes = FloatTensor(n, 4)
    # print(pred_boxes.shape, pred_conf.shape)

    for i in range(n):
        _, index = pred_conf[i].max(0)
        p_boxes[i] = pred_boxes[i][index]

    return p_boxes

def test(weights=None,
         batch_size=16,
         img_size=416,
         conf_thres=0.001,
         iou_thres=0.5,  # for nms
         save_json=False,
         single_cls=False,
         model=None,
         dataloader=None):

    device = next(model.parameters()).device  # get model device

    '''
    names = ['aa']
    path = '../dac_test1/test_data'
    # path = '../dac_test_pix/test'
    # path = '../data_training'
    '''

    # seen = 0
    model.eval()
    loss = torch.zeros(3)
    iou_sum = 0
    test_n = 0

    print(('\n' + '%10s' * 4) % ('IOU', 'l', 'Giou-l', 'obj-l'))
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))    
    for batch_i, (imgs, targets, paths, shapes) in pbar:
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        bn, _, height, width = imgs.shape  # batch size, channels, height, width
        test_n += bn

        # Disable gradients
        with torch.no_grad():
            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs
            
            
            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

            inf_out = inf_out.view(inf_out.shape[0], 6, -1)
            # ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
            inf_out_t = torch.zeros_like(inf_out[:, 0, :])
            for i in range(inf_out.shape[1]):
               inf_out_t += inf_out[:, i, :]
            inf_out_t = inf_out_t.view(inf_out_t.shape[0], -1, 6) / 6
            '''
            inf_out_t is avg of inf_out shape is (?, ?, 6)
            '''

            pre_box = get_boxes(inf_out_t[..., :4], inf_out_t[..., 4])
            # pre_box = get_boxes(inf_out[..., :4], inf_out[..., 4])

            tbox = targets[..., 2:6] * torch.Tensor([width, height, width, height]).to(device)

            # print(pre_box)
            # print(targets)
            # print(pre_box.shape, tbox.shape)
            ious = bbox_iou(pre_box, tbox)
            iou_sum += ious.sum()
            loss_o = loss / (batch_i + 1)

            iou = iou_sum / test_n
            s = ('%10.4f')*4 % (iou, loss_o.sum(), loss_o[0], loss_o[1])
            pbar.set_description(s)
    return iou, loss_o.sum(), loss_o[0], loss_o[1]