import numpy as np
import torch
from torchvision import transforms
import os
from PIL import Image
import cv2
from tqdm import tqdm

from model.MPFL import MPFL

from evaluatemetric import eval

def eval_model(test_image_root, test_gt_root, train_size, model):

    save_root = "./results_crackfm"

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    img_transform = transforms.Compose([
        transforms.Resize((train_size, train_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    images = sorted(os.listdir(test_image_root))
    gts = sorted(os.listdir(test_gt_root))

    images = [os.path.join(test_image_root, f) for f in images]
    gts = [os.path.join(test_gt_root, f) for f in gts]

    model.eval()

    print("Start Inference...")

    for i in tqdm(range(len(images))):

        image_path = images[i]
        gt_path = gts[i]

        ori_image = Image.open(image_path).convert("RGB")
        image = img_transform(ori_image).unsqueeze(0).cuda()

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        H, W = gt.shape

        with torch.no_grad():

            pred = model(image)

            res = pred[-1]#最后一张图，也就是相加得到的那张

            res = torch.sigmoid(res).cpu().numpy().squeeze()

        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        pred_img = Image.fromarray(res * 255).convert("L")
        pred_img = pred_img.resize((W, H), resample=Image.BILINEAR)

        pred_img = np.array(pred_img)

        root_name = os.path.basename(image_path).split(".")[0]

        save_pred = os.path.join(save_root, root_name + "_pre.png")
        save_gt = os.path.join(save_root, root_name + "_lab.png")

        cv2.imwrite(save_pred, pred_img)
        cv2.imwrite(save_gt, gt)

    print("Prediction finished!")

    print("\nStart Evaluation...")


    metrics = eval(save_root, epoch=0)
    print("\n========== Evaluation Results ==========")

    for k,v in metrics.items():
        print(f"{k} : {v}")



def main(dataset_name):

    net = CrackFM(method="pvt_v2_b2", channel=64)

    train_size = 384

    if torch.cuda.is_available():
        net = net.cuda()

    model_path = " "

    net.load_state_dict(torch.load(model_path), strict=False)

    file_dir = " "

    test_image_root = os.path.join(file_dir, dataset_name + "/test_img/")
    test_gt_root = os.path.join(file_dir, dataset_name + "/test_gt/")
    eval_model(test_image_root, test_gt_root, train_size, net)



if __name__ == '__main__':

    dataset_names = ["GAPS509-normal"]

    for dataset_name in dataset_names:

        print("\n=====================================")
        print("Dataset:", dataset_name)
        print("=====================================\n")

        main(dataset_name)