import argparse
import logging
import sys
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from scipy.ndimage import zoom

from mamba_dsnet import MambaDSNet
from mamba import MambaAblationOnly, MambaAblationCNN
from csmi_ablation import MambaAblationCSMINoDir, MambaAblationCSMINoResidual, MambaAblationCASelf
from utils import DiceLoss, get_edge_gt

try:
    from medpy import metric as medpy_metric
    from skimage import measure
    METRIC_AVAILABLE = True
except ImportError:
    medpy_metric = None
    measure = None
    METRIC_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mambadsnet', choices=['mambadsnet', 'mamba_ab1', 'mamba_ab2', 'mamba_csmi_nodir', 'mamba_csmi_nores', 'mamba_ca_self'],
                    help='model architecture: mambadsnet, mamba_ab1/ab2, mamba_csmi_nodir/nores, mamba_ca_self')
parser.add_argument('--dataset', type=str, default='Synapse', choices=['Synapse', 'ACDC', 'Kvasir'],
                    help='dataset name: Synapse, ACDC or Kvasir')
parser.add_argument('--root_path', type=str,
                    default='./data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')


parser.add_argument('--base_lr', type=float, default=None,
                    help='segmentation network learning rate (default: 0.02 for Synapse/ACDC, 1e-4 for Kvasir)')
parser.add_argument('--warmup_epochs', type=int, default=0,
                    help='linear warmup epochs (0=disabled)')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--num_workers', type=int,
                    default=8, help='number of workers for data loading')
parser.add_argument('--print_model_stats', action='store_true',
                    help='print Params & FLOPs at input size (requires thop)')
parser.add_argument('--disable_amp', action='store_true',
                    help='temporarily disable AMP mixed precision (default: AMP enabled)')
parser.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16'],
                    help='AMP dtype when AMP is enabled: fp16 or bf16')
parser.add_argument('--clip_grad', type=float, default=0.0,
                    help='max norm for gradient clipping; <=0 disables clipping')

# 边缘感知开关与权重（lambda_edge=0 等价于关闭）
parser.add_argument('--lambda_edge', type=float, default=0.2,
                    help='edge loss weight; set 0 to disable edge awareness')

# MambaDSNet 特有参数
parser.add_argument('--base_c', type=int, default=32, help='base channel count for MambaDSNet')
parser.add_argument('--mamba_ab_optimizer', type=str, default='sgd', choices=['sgd', 'adamw'],
                    help='optimizer for mamba_ab1/mamba_ab2')
parser.add_argument('--mamba_ab_weight_decay', type=float, default=1e-4,
                    help='weight decay for mamba_ab1/mamba_ab2 optimizer')
parser.add_argument('--mamba_ab_scheduler', type=str, default='poly', choices=['poly', 'cosine'],
                    help='lr scheduler for mamba_ab1/mamba_ab2')

args = parser.parse_args()

# Kvasir 使用 AdamW，默认 lr=1e-4；Synapse/ACDC 使用 SGD，默认 lr=0.02
if args.base_lr is None:
    args.base_lr = 1e-4 if args.dataset == 'Kvasir' else 0.02


def binary_dice_loss(pred, target, eps=1e-6):
    """二值 Dice Loss，pred/target shape: [B,1,H,W]，pred 已是概率。"""
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def compute_model_stats(net, img_size, device='cuda'):
    """Compute Params/FLOPs. Unwrap tuple outputs if the model returns (seg, edge)."""
    try:
        from thop import profile, clever_format
    except ImportError:
        print("[ModelStats] thop not installed, skip FLOPs/Params stats.")
        return

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            if isinstance(out, (list, tuple)):
                return out[0]
            return out

    net_to_profile = net.module if isinstance(net, nn.DataParallel) else net
    net_to_profile = Wrapper(net_to_profile).to(device)
    dummy = torch.randn(1, 1, img_size, img_size, device=device)
    try:
        flops, params = profile(net_to_profile, inputs=(dummy,), verbose=False)
        flops, params = clever_format([flops, params], '%.3f')
        print(f"[ModelStats] Params: {params}, FLOPs@{img_size}x{img_size}: {flops}")
    except Exception as e:
        print(f"[ModelStats] Profile failed, skip stats. Error: {e}")


def calculate_metric_percase(pred, gt):
    pred = np.asarray(pred, dtype=np.uint8)
    gt = np.asarray(gt, dtype=np.uint8)
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum > 0 and gt_sum > 0:
        dice = medpy_metric.binary.dc(pred, gt)
        hd95 = medpy_metric.binary.hd95(pred, gt)
        assd = medpy_metric.binary.assd(pred, gt)
        recall = medpy_metric.binary.sensitivity(pred, gt)
        return dice, hd95, assd, recall
    elif pred_sum > 0 and gt_sum == 0:
        return 1, 0, 0, 0
    else:
        return 0, 0, 0, 0


def clean_segmentation(prediction, num_classes):
    prediction = np.asarray(prediction, dtype=np.int32)
    clean_pred = np.zeros_like(prediction)
    for c in range(1, num_classes):
        class_mask = prediction == c
        if not class_mask.any():
            continue
        labeled, num = measure.label(class_mask, return_num=True, connectivity=1)
        if num == 1:
            clean_pred[class_mask] = c
            continue
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        max_label = sizes.argmax()
        clean_pred[labeled == max_label] = c
    return clean_pred


def infer_volume_3d(image_np, net, classes, patch_size, device):
    def infer_slice(slc_np):
        x, y = slc_np.shape
        if (x, y) != tuple(patch_size):
            resized = zoom(slc_np, (patch_size[0] / x, patch_size[1] / y), order=3)
        else:
            resized = slc_np

        inp = torch.as_tensor(resized, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = net(inp)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
        if (x, y) != tuple(patch_size):
            pred = zoom(pred, (x / patch_size[0], y / patch_size[1]), order=0)
        return pred

    if len(image_np.shape) == 3:
        prediction = np.zeros_like(image_np, dtype=np.int16)
        for idx in range(image_np.shape[0]):
            prediction[idx] = infer_slice(image_np[idx])
    else:
        prediction = infer_slice(image_np)

    return clean_segmentation(prediction, classes)


def validate_acdc_3d(model, val_loader, num_classes, img_size, device):
    if not METRIC_AVAILABLE:
        raise ImportError("medpy and scikit-image are required for 3D validation metrics. Install via: pip install medpy scikit-image")

    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(val_loader):
        image = sampled_batch['image']
        label = sampled_batch['label']
        case_name = sampled_batch['case_name'][0] if 'case_name' in sampled_batch else f"case_{i_batch}"

        image_np = image.squeeze(0).cpu().detach().numpy()
        label_np = label.squeeze(0).cpu().detach().numpy()

        prediction = infer_volume_3d(image_np, model, classes=num_classes,
                                     patch_size=[img_size, img_size], device=device)

        metric_i = []
        for i in range(1, num_classes):
            metric_i.append(calculate_metric_percase(prediction == i, label_np == i))
        metric_list += np.array(metric_i)

        logging.info('val idx %d case %s mean_dice %f mean_hd95 %f mean_assd %f mean_recall %f' % (
            i_batch, case_name,
            np.mean(metric_i, axis=0)[0],
            np.mean(metric_i, axis=0)[1],
            np.mean(metric_i, axis=0)[2],
            np.mean(metric_i, axis=0)[3]))

    metric_list = metric_list / len(val_loader.dataset)
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_assd = np.mean(metric_list, axis=0)[2]
    mean_recall = np.mean(metric_list, axis=0)[3]

    for i in range(1, num_classes):
        logging.info('Val class %d mean_dice %f mean_hd95 %f mean_assd %f mean_recall %f' % (
            i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2], metric_list[i - 1][3]))

    logging.info('3D validation performance: mean_dice : %f mean_hd95 : %f mean_assd : %f mean_recall : %f' % (
        performance, mean_hd95, mean_assd, mean_recall))
    return performance, mean_hd95, mean_assd, mean_recall


def build_model(args, device):
    """构建模型，返回模型与边缘分支标记。"""
    in_channels = 3 if args.dataset == 'Kvasir' else 1
    
    model_type = args.model

    if model_type == 'mamba_ab1':
        model = MambaAblationOnly(in_ch=in_channels, num_classes=args.num_classes, base_c=args.base_c)
        supports_edge = False
        print(f"=> Building Mamba Ablation-1 (Mamba only) with in_ch: {in_channels}, num_classes: {args.num_classes}, base_c: {args.base_c}")
    elif model_type == 'mamba_ab2':
        model = MambaAblationCNN(in_ch=in_channels, num_classes=args.num_classes, base_c=args.base_c)
        supports_edge = False
        print(f"=> Building Mamba Ablation-2 (Mamba + CNN, no CSMI) with in_ch: {in_channels}, num_classes: {args.num_classes}, base_c: {args.base_c}")
    elif model_type == 'mamba_csmi_nodir':
        model = MambaAblationCSMINoDir(in_ch=in_channels, num_classes=args.num_classes, base_c=args.base_c)
        supports_edge = False
        print(f"=> Building Mamba Ablation CSMI-NoDir (cross-stream CA + residual, no dir gate) with in_ch: {in_channels}, num_classes: {args.num_classes}, base_c: {args.base_c}")
    elif model_type == 'mamba_csmi_nores':
        model = MambaAblationCSMINoResidual(in_ch=in_channels, num_classes=args.num_classes, base_c=args.base_c)
        supports_edge = False
        print(f"=> Building Mamba Ablation CSMI-NoRes (cross-stream CA + dir gate, no residual) with in_ch: {in_channels}, num_classes: {args.num_classes}, base_c: {args.base_c}")
    elif model_type == 'mamba_ca_self':
        model = MambaAblationCASelf(in_ch=in_channels, num_classes=args.num_classes, base_c=args.base_c)
        supports_edge = False
        print(f"=> Building Mamba Ablation CA-Self (standard CA self-attn + sum fusion) with in_ch: {in_channels}, num_classes: {args.num_classes}, base_c: {args.base_c}")
    else:
        model = MambaDSNet(in_ch=in_channels, num_classes=args.num_classes, base_c=args.base_c)
        supports_edge = True
        print(f"=> Building MambaDSNet with in_ch: {in_channels}, num_classes: {args.num_classes}, base_c: {args.base_c}")
    
    return model.to(device), supports_edge


def train(args):
    from datasets.dataset_synapse import RandomGenerator, ACDC_dataset
    from datasets.dataset_synapse import Synapse_dataset

    # reproducibility
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'root_path': './data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'ACDC': {
            'root_path': './data/ACDC/train',
            'list_dir': './lists/lists_ACDC',
            'num_classes': 4,
        },
        'Kvasir': {
            'root_path': './data/Kvasir-SEG',
            'list_dir': '',
            'num_classes': 1,
        }
    }

    dataset_name = args.dataset
    if dataset_name in dataset_config:
        cfg = dataset_config[dataset_name]
        args.num_classes = cfg['num_classes']
        if args.root_path == './data/Synapse/train_npz':
            args.root_path = cfg['root_path']
        if args.list_dir == './lists/lists_Synapse':
            args.list_dir = cfg['list_dir']

    model_type = args.model
    model_name = model_type

    args.exp = f"{model_name}_{dataset_name}{args.img_size}"
    exp_lr = args.base_lr
    snapshot_base = "./model/{}/{}".format(dataset_name, args.exp)
    snapshot_base = snapshot_base + '_epo' + str(args.max_epochs)
    snapshot_base = snapshot_base + '_bs' + str(args.batch_size)
    snapshot_base = snapshot_base + '_lr' + str(exp_lr)
    snapshot_base = snapshot_base + '_s' + str(args.seed)

    # device
    if torch.cuda.is_available() and args.n_gpu > 0:
        device = torch.device(f"cuda:{args.gpu_ids[0]}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # model
    model, supports_edge = build_model(args, device)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    lambda_edge = args.lambda_edge  # 支持通过参数关闭或调整边缘感知
    edge_enabled = lambda_edge > 0 and supports_edge
    edge_tag = f"edge_{lambda_edge}" if edge_enabled else "noedge"  # 包含具体的超参数值以区分敏感性实验
    snapshot_path = os.path.join(snapshot_base, edge_tag)  # 保存到带有超参数值的子目录
    os.makedirs(snapshot_path, exist_ok=True)
    
    if dataset_name == 'Kvasir':
        from trainer_kvasir import trainer_kvasir
        trainer_kvasir(args, model, snapshot_path)
        return
    
    scheduler = None
    use_poly_lr = True
    if model_type in ['mamba_ab1', 'mamba_ab2']:
        if args.mamba_ab_optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.mamba_ab_weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=args.mamba_ab_weight_decay)

        if args.mamba_ab_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.max_epochs,
                eta_min=max(args.base_lr * 0.01, 1e-6),
            )
            use_poly_lr = False
        else:
            use_poly_lr = True
        print(
            f"=> Using Mamba Ablation training strategy: {args.mamba_ab_optimizer.upper()} "
            f"(lr={args.base_lr}, wd={args.mamba_ab_weight_decay}) + {args.mamba_ab_scheduler.upper()}"
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
        print(f"=> Using MambaDSNet training strategy: SGD (lr={args.base_lr}) + PolyLR")
    
    amp_enabled = not args.disable_amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bf16' else torch.float16
    use_grad_scaler = amp_enabled and args.amp_dtype == 'fp16'
    scaler = GradScaler(enabled=use_grad_scaler)
    if amp_enabled:
        print(f"=> AMP enabled, dtype={args.amp_dtype}")
    else:
        print("=> AMP disabled, using fp32")

    # dataset
    val_loader = None
    if dataset_name == 'Synapse':
        db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                   transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    else:
        # ACDC 训练：全切片 + 过滤纯背景切片
        db_train = ACDC_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]),
                                use_all_slices=True,
                                filter_pure_background=True)
        # ACDC 验证集：全切片，保留纯背景切片（尽量贴近测试分布）
        val_root = os.path.join(os.path.dirname(args.root_path), 'valid')
        db_val = ACDC_dataset(base_dir=val_root, list_dir=args.list_dir, split="test",
                      transform=None,
                              use_all_slices=True,
                              filter_pure_background=False)
        val_loader = DataLoader(db_val, batch_size=1, shuffle=False,
                                num_workers=args.num_workers if args.n_gpu > 0 else 0, pin_memory=True)

    os.environ['WORKER_SEED'] = str(args.seed)
    trainloader = DataLoader(db_train, batch_size=args.batch_size * args.n_gpu, shuffle=True,
                             num_workers=args.num_workers if args.n_gpu > 0 else 0, pin_memory=True,
                             worker_init_fn=lambda worker_id: random.seed(int(os.environ.get('WORKER_SEED', 1234)) + worker_id))

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    best_loss = float('inf')
    best_val_loss = float('inf')
    best_val_dice = -float('inf')
    
    for epoch_num in range(max_epoch):
        model.train()
        epoch_loss = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            img = sampled_batch['image'].to(device, non_blocking=True)
            lab = sampled_batch['label'].to(device, non_blocking=True)

            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(img)
                if supports_edge:
                    seg_logits, edge_logits = outputs
                    loss_ce = ce_loss(seg_logits, lab)
                    loss_dice = dice_loss(seg_logits, lab, softmax=True)
                else:
                    seg_logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                    edge_logits = None
                    loss_ce = ce_loss(seg_logits, lab)
                    loss_dice = dice_loss(seg_logits, lab, softmax=True)

                loss_seg = 0.5 * loss_ce + 1.0 * loss_dice
                if edge_enabled and edge_logits is not None:
                    edge_gt = get_edge_gt(lab)
                    edge_pred = torch.sigmoid(edge_logits)
                    loss_edge = binary_dice_loss(edge_pred, edge_gt)
                    loss = loss_seg + lambda_edge * loss_edge
                else:
                    loss_edge = torch.tensor(0.0, device=img.device)
                    loss = loss_seg

            optimizer.zero_grad(set_to_none=True)
            if use_grad_scaler:
                scaler.scale(loss).backward()
                clip_max_norm = args.clip_grad if args.clip_grad > 0 else 0.0
                if clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_max_norm = args.clip_grad if args.clip_grad > 0 else 0.0
                if clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                optimizer.step()

            # 学习率调度 (含 warmup)
            warmup_iters = args.warmup_epochs * len(trainloader) if args.warmup_epochs > 0 else 0
            if warmup_iters > 0 and iter_num < warmup_iters:
                lr_ = args.base_lr * (iter_num + 1) / warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            elif use_poly_lr:
                lr_ = args.base_lr * (1.0 - (iter_num - warmup_iters) / max(1, max_iterations - warmup_iters)) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_ = optimizer.param_groups[0]['lr']

            iter_num += 1
            epoch_loss += loss.item()
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('info/loss_dice', loss_dice.item(), iter_num)
            writer.add_scalar('info/loss_edge', loss_edge.item(), iter_num)

        avg_epoch_loss = epoch_loss / (i_batch + 1)
        logging.info('[epoch %d] loss: %.5f' % (epoch_num + 1, avg_epoch_loss))
        
        if scheduler is not None:
            scheduler.step()

        # ACDC 验证（与测试一致：3D 体积推理指标）
        if val_loader is not None:
            avg_val_dice, avg_val_hd95, avg_val_assd, avg_val_recall = validate_acdc_3d(
                model=model,
                val_loader=val_loader,
                num_classes=args.num_classes,
                img_size=args.img_size,
                device=device,
            )
            avg_val_loss = 1.0 - avg_val_dice
            writer.add_scalar('val/total_loss', avg_val_loss, epoch_num + 1)
            writer.add_scalar('val/dice', avg_val_dice, epoch_num + 1)
            writer.add_scalar('val/hd95', avg_val_hd95, epoch_num + 1)
            writer.add_scalar('val/assd', avg_val_assd, epoch_num + 1)
            writer.add_scalar('val/recall', avg_val_recall, epoch_num + 1)
            logging.info('[epoch %d] val_proxy_loss: %.5f, val_dice: %.4f, val_hd95: %.4f, val_assd: %.4f, val_recall: %.4f' % (
                epoch_num + 1, avg_val_loss, avg_val_dice, avg_val_hd95, avg_val_assd, avg_val_recall))

            # Save best on validation
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_path = os.path.join(snapshot_path, f'best_model_val_{edge_tag}.pth')
                torch.save(model.state_dict(), best_val_path)
                logging.info(f"best val model updated: {best_val_loss:.5f}, saved to {best_val_path}")

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                best_val_dice_path = os.path.join(snapshot_path, f'best_model_valdice_{edge_tag}.pth')
                torch.save(model.state_dict(), best_val_dice_path)
                logging.info(f"best val dice updated: {best_val_dice:.4f}, saved to {best_val_dice_path}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(snapshot_path, f'best_model_{edge_tag}.pth')
            torch.save(model.state_dict(), best_path)
            logging.info(f"best model updated: {best_loss:.5f}, saved to {best_path}")

        if (epoch_num + 1) % 10 == 0 or epoch_num == max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num + 1}_{edge_tag}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training finished!"


if __name__ == "__main__":
    train(args)