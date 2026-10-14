"""
U-Net Training Script for Kvasir-SEG Dataset
Using segmentation_models_pytorch library
Keeps same data augmentation, loss, and hyperparameters as MambaDSNet for fair comparison
"""
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

import segmentation_models_pytorch as smp
from datasets.dataset_kvasir import KvasirDataset
from utils import bce_dice_loss


def train_unet_kvasir(args):
    # Reproducibility
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create snapshot path
    pretrained_tag = "pretrained" if args.pretrained else "scratch"
    snapshot_path = f"./model/UNet_Kvasir/unet_{args.encoder}_{pretrained_tag}_Kvasir{args.img_size}_epo{args.max_epochs}_bs{args.batch_size}_lr{args.base_lr}_s{args.seed}_adamw_cos"
    os.makedirs(snapshot_path, exist_ok=True)
    
    # Logging
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    # Build U-Net model using segmentation_models_pytorch
    print(f"=> Building U-Net with encoder: {args.encoder}, pretrained: {args.pretrained}")
    model = smp.Unet(
        encoder_name=args.encoder,        # e.g., "resnet34"
        encoder_weights="imagenet" if args.pretrained else None,  # Pretrained or from scratch
        in_channels=3,                    # RGB input
        classes=1,                        # Binary segmentation
    ).to(device)
    
    # Dataset
    train_data_dir = os.path.join(args.root_path, "TrainDataset")
    db_train = KvasirDataset(data_root=train_data_dir, img_size=args.img_size, mode='train')
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    
    # Optimizer & Scheduler (matched to MambaDSNet for fair comparison)
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-5)
    
    # Mixed precision
    scaler = GradScaler()
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations".format(len(trainloader), max_iterations))
    
    best_loss = float('inf')
    
    iterator = tqdm(range(args.max_epochs), ncols=70)
    for epoch_num in iterator:
        model.train()
        epoch_loss = 0.0
        
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            
            # Label unsqueeze for BCE format [B, H, W] -> [B, 1, H, W]
            if label_batch.dim() == 3:
                label_batch = label_batch.unsqueeze(1)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=True):
                outputs = model(image_batch)
                loss = bce_dice_loss(outputs, label_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            iter_num += 1
            
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            
            if iter_num % 20 == 0:
                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
        
        # Calculate average loss for this epoch
        epoch_loss = epoch_loss / len(trainloader)
        logging.info('epoch %d : average loss : %f' % (epoch_num, epoch_loss))
        
        # Step scheduler (CosineAnnealing, matches MambaDSNet)
        scheduler.step()
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info("save best model to {} with loss {}".format(best_model_path, best_loss))
        
        # Save checkpoint at intervals
        save_interval = 50
        if epoch_num >= args.max_epochs - 1 or epoch_num % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
    
    writer.close()
    logging.info("Training Finished!")
    return snapshot_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./data/Kvasir-SEG', help='root dir for dataset')
    parser.add_argument('--encoder', type=str, default='resnet34', help='encoder backbone for U-Net')
    parser.add_argument('--pretrained', action='store_true', help='use ImageNet pretrained encoder')
    parser.add_argument('--img_size', type=int, default=352, help='input image size')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    
    args = parser.parse_args()
    train_unet_kvasir(args)
