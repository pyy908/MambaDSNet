import logging
import os
import random
import sys
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_kvasir import KvasirDataset
from utils import get_edge_gt, bce_dice_loss


def trainer_kvasir(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_epochs = args.max_epochs
    
    db_train = KvasirDataset(data_root=os.path.join(args.root_path, "TrainDataset"), img_size=args.img_size, mode='train')
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    
    model.train()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    # Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_iterations = max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    iterator = tqdm(range(max_epochs), ncols=70)
    best_loss = float('inf')
    
    for epoch_num in iterator:
        epoch_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            # Label unsqueeze for BCE format
            if label_batch.dim() == 3:
                label_batch = label_batch.unsqueeze(1)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=True):
                # 模型输出 (seg_logits, edge_logits)
                outputs = model(image_batch)

                if isinstance(outputs, tuple):
                    pred, edge = outputs
                    loss_seg = bce_dice_loss(pred, label_batch)

                    edge_gt = get_edge_gt(label_batch)

                    # 边缘也使用 BCE+Dice
                    loss_edge = bce_dice_loss(edge, edge_gt)

                    loss = loss_seg + args.lambda_edge * loss_edge
                else:
                    pred = outputs
                    loss_seg = bce_dice_loss(pred, label_batch)
                    loss_edge = torch.tensor(0.0)
                    loss = loss_seg
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            
            if iter_num % 20 == 0:
                logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
                
        scheduler.step()
        
        # Calculate average loss for this epoch
        epoch_loss = epoch_loss / len(trainloader)
        logging.info('epoch %d : average loss : %f' % (epoch_num, epoch_loss))
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info("save best model to {} with loss {}".format(best_model_path, best_loss))
            
        save_interval = 50
        if epoch_num >= max_epochs - 1 or epoch_num % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"
