"""
U-Net Testing Script for Kvasir-SEG Dataset
Same evaluation metrics as MambaDSNet for fair comparison: mDice, mIoU, mMAE
"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

import segmentation_models_pytorch as smp


class KvasirTestDataset(Dataset):
    """Test dataset that preserves original image size for accurate evaluation"""
    def __init__(self, data_root, img_size=352):
        self.data_root = data_root
        self.img_size = img_size
        
        self.image_dir = os.path.join(data_root, 'images')
        self.mask_dir = os.path.join(data_root, 'masks')
        
        self.samples = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg'))]
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        name = self.samples[idx]
        img_path = os.path.join(self.image_dir, name)
        
        m_name = name if name.endswith('.png') else name.replace('.jpg', '.png')
        if not os.path.exists(os.path.join(self.mask_dir, m_name)):
            m_name = name
            
        mask_path = os.path.join(self.mask_dir, m_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Save original size for reconstruction
        orig_w, orig_h = image.size
        
        # Original mask array for accurate evaluation
        gt_mask = np.array(mask, dtype=np.float32) / 255.0
        gt_mask[gt_mask >= 0.5] = 1.0
        gt_mask[gt_mask < 0.5] = 0.0
        
        # Resize image for network input
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize with ImageNet mean/std
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = image.transpose((2, 0, 1))
        
        image_tensor = torch.from_numpy(image).float()
        
        return {'image': image_tensor, 'gt_mask': gt_mask, 'name': name, 'orig_size': (orig_w, orig_h)}


def calc_dice(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def calc_iou(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def calc_mae(pred, target):
    return np.mean(np.abs(pred - target))


def test_unet_kvasir(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build U-Net model
    print(f"=> Building U-Net with encoder: {args.encoder}")
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,  # We'll load our trained weights
        in_channels=3,
        classes=1,
    ).to(device)
    
    if os.path.exists(args.model_path):
        print(f"=> Loading weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    
    model.eval()
    
    # Dataset & DataLoader
    test_data_dir = os.path.join(args.root_path, "TestDataset", "Kvasir")
    db_test = KvasirTestDataset(data_root=test_data_dir, img_size=args.img_size)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=4)
    print(f"=> Loaded {len(db_test)} test samples from {test_data_dir}")
    
    # Metrics Tracking
    dice_list = []
    iou_list = []
    mae_list = []
    
    # Optional: Directory to save visual predictions
    if args.save_preds:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"=> Predictions will be saved to: {args.save_dir}")
    
    # Inference
    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing U-Net", ncols=80):
            images = batch['image'].to(device)
            gt_masks = batch['gt_mask'].numpy()[0]  # Original GT shape [H, W]
            names = batch['name'][0]
            orig_w, orig_h = batch['orig_size'][0].item(), batch['orig_size'][1].item()
            
            # Forward
            outputs = model(images)
            
            # Probabilities
            preds_prob = torch.sigmoid(outputs)
            
            # Resize back to original dimensions
            preds_prob = F.interpolate(preds_prob, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            preds_prob = preds_prob.squeeze().cpu().numpy()
            
            # Binarize
            preds_binary = (preds_prob >= 0.5).astype(np.float32)
            
            # Calculate metrics
            dice = calc_dice(preds_binary, gt_masks)
            iou = calc_iou(preds_binary, gt_masks)
            mae = calc_mae(preds_prob, gt_masks)
            
            dice_list.append(dice)
            iou_list.append(iou)
            mae_list.append(mae)
            
            # Save predictions
            if args.save_preds:
                pred_img = (preds_binary * 255).astype(np.uint8)
                pred_pil = Image.fromarray(pred_img)
                pred_pil.save(os.path.join(args.save_dir, names))
    
    mDice = np.mean(dice_list)
    mIoU = np.mean(iou_list)
    mMAE = np.mean(mae_list)
    
    # Report
    print("=" * 50)
    print("         U-Net Kvasir-SEG Test Report")
    print("=" * 50)
    print(f"Encoder           : {args.encoder}")
    print(f"Model             : {args.model_path}")
    print("-" * 50)
    print(f"Mean Dice (mDice) : {mDice:.4f}")
    print(f"Mean IoU  (mIoU)  : {mIoU:.4f}")
    print(f"Mean MAE  (mMAE)  : {mMAE:.4f}")
    print("=" * 50)
    
    return mDice, mIoU, mMAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./data/Kvasir-SEG', help='root dir for dataset')
    parser.add_argument('--model_path', type=str, required=True, help='path to the model weights (.pth)')
    parser.add_argument('--encoder', type=str, default='resnet34', help='encoder backbone for U-Net')
    parser.add_argument('--img_size', type=int, default=352, help='input image size')
    parser.add_argument('--save_preds', action='store_true', help='whether to save predicted masks')
    parser.add_argument('--save_dir', type=str, default='./test_predictions/UNet_Kvasir/', help='dir to save preds')
    
    args = parser.parse_args()
    test_unet_kvasir(args)
