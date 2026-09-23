import os
import glob
import cv2
import numpy as np
import torch
def get_image_pairs(data_dir, suffix_gt='lab', suffix_pred='pre'):
    gt_list = sorted(glob.glob(os.path.join(data_dir, f'*{suffix_gt}.png')))
    pred_list = [ll.replace(suffix_gt, suffix_pred) for ll in gt_list]
    pred_imgs, gt_imgs = [], []
    for p, g in zip(pred_list, gt_list):
        pred_imgs.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
        gt_imgs.append(cv2.imread(g, cv2.IMREAD_GRAYSCALE))
    return pred_imgs, gt_imgs

def get_statistics(pred, gt):
    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    tn = np.sum((pred==0)&(gt==0))
    return tp, fp, fn, tn

def cal_mIoU_metrics(pred_list, gt_list, thresh_step=0.01):
    final_iou = []
    for thresh in np.arange(0,1,thresh_step):
        iou_list = []
        for pred, gt in zip(pred_list, gt_list):
            pred_bin = (pred / 255 > thresh).astype(np.uint8)
            gt_bin = (gt / 255).astype(np.uint8)
            TP, FP, FN, TN = get_statistics(pred_bin, gt_bin)
            iou_1 = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0
            iou_0 = TN / (TN + FP + FN) if TN + FP + FN > 0 else 0
            iou_list.append((iou_0 + iou_1)/2)
        final_iou.append(np.mean(iou_list))
    return np.max(final_iou)

#旧的代码
def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_F1 = []
    for thresh in np.arange(0,1,thresh_step):
        F_list = []
        for pred, gt in zip(pred_list, gt_list):
            pred_bin = (pred / 255 > thresh).astype(np.uint8)
            gt_bin = (gt / 255).astype(np.uint8)
            TP, FP, FN, _ = get_statistics(pred_bin, gt_bin)
            P = 1.0 if TP+FP==0 else TP/(TP+FP)
            R = 0.0 if TP+FN==0 else TP/(TP+FN)
            F = 0.0 if P+R==0 else 2*P*R/(P+R)
            F_list.append(F)
        final_F1.append(np.mean(F_list))
    max_idx = np.argmax(final_F1)
    # 对应Precision, Recall, F1
    thresh = np.arange(0,1,thresh_step)[max_idx]
    P_list, R_list, F_list = [], [], []
    for pred, gt in zip(pred_list, gt_list):
        pred_bin = (pred / 255 > thresh).astype(np.uint8)
        gt_bin = (gt / 255).astype(np.uint8)
        TP, FP, FN, _ = get_statistics(pred_bin, gt_bin)
        P = 1.0 if TP+FP==0 else TP/(TP+FP)
        R = 0.0 if TP+FN==0 else TP/(TP+FN)
        F = 0.0 if P+R==0 else 2*P*R/(P+R)
        P_list.append(P)
        R_list.append(R)
        F_list.append(F)
    return np.mean(P_list), np.mean(R_list), np.mean(F_list)







def cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    return cal_prf_metrics(pred_list, gt_list, thresh_step)[2]

def cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        F1_list = []
        for thresh in np.arange(0,1,thresh_step):
            pred_bin = (pred / 255 > thresh).astype(np.uint8)
            gt_bin = (gt / 255).astype(np.uint8)
            TP, FP, FN, _ = get_statistics(pred_bin, gt_bin)
            P = 1.0 if TP+FP==0 else TP/(TP+FP)
            R = 0.0 if TP+FN==0 else TP/(TP+FN)
            F = 0.0 if P+R==0 else 2*P*R/(P+R)
            F1_list.append(F)
        final_F1_list.append(np.max(F1_list))
    return np.mean(final_F1_list)

def eval(results_dir, epoch=0):
    pred_list, gt_list = get_image_pairs(results_dir)
    mIoU = cal_mIoU_metrics(pred_list, gt_list)
    ODS = cal_ODS_metrics(pred_list, gt_list)
    OIS = cal_OIS_metrics(pred_list, gt_list)
    Precision, Recall, F1 = cal_prf_metrics(pred_list, gt_list)
    metrics = {
        "epoch": epoch,
        "mIoU": mIoU,
        "ODS": ODS,
        "OIS": OIS,
        "F1": F1,
        "Precision": Precision,
        "Recall": Recall
    }
    return metrics