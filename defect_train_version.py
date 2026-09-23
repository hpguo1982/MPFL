import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import os


import logging

import sys

from PIL import Image

from torch.optim.lr_scheduler import CosineAnnealingLR
from model.MPFL import MPFL
from ESDI_dataloader import get_loader
import cv2
from tqdm import tqdm



def setup_logger(name, save_dir, filename="log.txt", mode='w'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)

    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - inter/(union-inter)
    iou = iou.mean()

    #mse_loss = nn.MSELoss(reduction="mean")
    #mse = mse_loss(pred, mask)

    return iou+bce
import time
def train(model_name, dataset_name):


    if dataset_name == "CrackSeg9k":
        # epoch_num = 60
        epoch_num = 80
        epoch_val = 30
    elif dataset_name == "ZJU-Leaper":
        epoch_num = 60
        epoch_val = 20
    elif dataset_name == "ESDIS-SOD":
        epoch_num = 180
        epoch_val = 100
    elif dataset_name == "Crack500":
        epoch_num = 300
        epoch_val = 1
    elif dataset_name == "TUT":
        epoch_num = 300
        epoch_val = 1
    elif dataset_name == "GAPS509":
        epoch_num = 300
        epoch_val = 1
    net = MPFL(method="pvt_v2_b2", channel=64)
    train_size = 384

    file_dir= " "

    # train_image_root = os.path.join(file_dir, dataset_name + "/train/images/")
    # train_gt_root = os.path.join(file_dir, dataset_name + "/train/gt/")
    # test_image_root = os.path.join(file_dir, dataset_name + "/test/images/")
    # test_gt_root = os.path.join(file_dir, dataset_name + "/test/gt/")


    train_image_root = os.path.join(file_dir, dataset_name + "/train_img/")
    train_gt_root = os.path.join(file_dir, dataset_name + "/train_lab/")
    test_image_root = os.path.join(file_dir, dataset_name + "/val_img/")
    test_gt_root = os.path.join(file_dir, dataset_name + "/val_lab/")

    train_loader1 = get_loader(train_image_root, train_gt_root, batchsize=16, trainsize=train_size, is_train=True)

    # ------- 3. define model --------
    if torch.cuda.is_available():
        net=net.cuda()


    # ------- 4. define optimizer --------
    print("---define optimizer...")

    optimizer = optim.Adam(net.parameters(), lr=8e-5)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=1e-7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- 5. training process --------
    print("---start training...")

    running_loss = 0.0


    best_F1=0


    for epoch in range(0, epoch_num):
        print(epoch)
        start_time = time.time()

        for i, data in enumerate(tqdm(train_loader1)):


            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            images, gts = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device),
                                                                                     requires_grad=False)
            #


            # y zero the parameter gradients
            optimizer.zero_grad()
            predictions_mask = net(images)

            mask_losses=0

            for i in range(len(predictions_mask)):
                mask_losses = mask_losses + total_loss(predictions_mask[i], gts)

            losses = mask_losses
            losses.backward()
            optimizer.step()

            running_loss += losses.item()


        end_time = time.time()
        print('Cost time: {:.4f}'.format(end_time - start_time))

        lr_scheduler.step()

        if (epoch + 1) >= epoch_val:
            # ---------------- GPU F1 验证 ----------------
            import torch.nn.functional as F
            net.eval()
            img_transform = transforms.Compose([
                transforms.Resize((train_size, train_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            images = sorted([os.path.join(test_image_root, f) for f in os.listdir(test_image_root)])
            gts = sorted([os.path.join(test_gt_root, f) for f in os.listdir(test_gt_root)])

            pred_list, gt_list = [], []

            for i in range(0, len(images), 8):  # batch size = 8
                batch_imgs = images[i:i + 8]
                batch_gts = gts[i:i + 8]

                tensors, batch_h, batch_w = [], [], []
                for img_path, gt_path in zip(batch_imgs, batch_gts):
                    ori_image = Image.open(img_path).convert("RGB")
                    tensors.append(img_transform(ori_image))
                    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                    batch_h.append(gt.shape[0])
                    batch_w.append(gt.shape[1])
                    gt_list.append(gt)

                x = torch.stack(tensors).cuda()
                with torch.no_grad():
                    preds = net(x)
                    res = torch.sigmoid(preds[-1])  # (B,1,H,W) GPU tensor

                for j in range(res.shape[0]):

                    p_tensor = res[j].unsqueeze(0)
                    p_resized = F.interpolate(
                        p_tensor,
                        size=(batch_h[j], batch_w[j]),
                        mode='bilinear',
                        align_corners=False
                    )
                    p_resized = p_resized.squeeze(0).squeeze(0)  # [H,W]

                    p_resized = (p_resized - p_resized.min()) / (p_resized.max() - p_resized.min() + 1e-8)
                    pred_list.append((p_resized.cpu().numpy() * 255).astype(np.uint8))

            # ---------------- F1 指标计算 ----------------
            from evaluatemetric import cal_prf_metrics
            Precision, Recall, F1 = cal_prf_metrics(pred_list, gt_list)

            if F1 > best_F1:
                save_path = os.path.join("./save", dataset_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(net.state_dict(), os.path.join(save_path, f"{model_name}-{dataset_name}-{F1:.4f}.pth"))
                best_F1 = F1

            print(f"F1: {F1:.4f}, best_F1: {best_F1:.4f}")
            net.train()





if __name__ == '__main__':
    model_name ='Crack500'
    dataset_name = "Crack500"
    train(model_name, dataset_name)




