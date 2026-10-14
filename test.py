import argparse
import logging
import os
import random
import sys
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import ACDC_dataset, Synapse_dataset
from mamba_dsnet import MambaDSNet
from mamba import MambaAblationOnly, MambaAblationCNN
from csmi_ablation import MambaAblationCSMINoDir, MambaAblationCSMINoResidual, MambaAblationCASelf
from medpy import metric
from scipy import stats as scipy_stats
from scipy.ndimage import zoom
from skimage import measure

# SimpleITK 仅用于保存 NIfTI，可选；缺失则在保存时报错提示安装
try:
    import SimpleITK as sitk
    sitk_available = True
except Exception:
    sitk = None
    sitk_available = False
from PIL import Image


def _load_state_dict_flexible(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            ckpt = ckpt['state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            ckpt = ckpt['model']
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    import re
    cleaned = {}
    for key, value in ckpt.items():
        new_key = key.replace('module.', '')
        if new_key.startswith('network.'):
            new_key = new_key[len('network.'):]
        # MONAI < 1.3 used sub0/sub1/sub2/sub3/subconv; MONAI >= 1.3 uses submodule
        # re.sub may skip nested matches when replacement is longer; loop until stable
        while True:
            prev = new_key
            new_key = re.sub(r'\.sub(\d+)\.', r'.submodule.\1.', new_key)
            new_key = re.sub(r'\.subconv\.', r'.submodule.conv.', new_key)
            if new_key == prev:
                break
        cleaned[new_key] = value
    return cleaned

def calculate_metric_percase(pred, gt):
    # 强制转换为标准 numpy 数组，避免奇怪的递归/类型问题
    pred = np.asarray(pred, dtype=np.uint8)
    gt = np.asarray(gt, dtype=np.uint8)

    pred[pred > 0] = 1
    gt[gt > 0] = 1

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum > 0 and gt_sum > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        assd = metric.binary.assd(pred, gt)
        recall = metric.binary.sensitivity(pred, gt)
        return dice, hd95, assd, recall
    elif pred_sum > 0 and gt_sum == 0:
        return 1, 0, 0, 0
    else:
        return 0, 0, 0, 0


def clean_segmentation(prediction, num_classes):
    """保留每个类别最大的连通域，抹除孤立噪点。"""
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
        sizes[0] = 0  # 背景置零避免被选中
        max_label = sizes.argmax()
        clean_pred[labeled == max_label] = c
    return clean_pred


def compute_model_stats(net, img_size, device='cuda', in_channels=1):
    """计算 Params 与 FLOPs。直接 print，避免日志未配置看不到。"""
    try:
        from thop import profile, clever_format
        import copy
    except ImportError:
        logging.info("[ModelStats] thop not installed, skip FLOPs/Params stats.")
        return

    net_to_profile = net.module if isinstance(net, torch.nn.DataParallel) else net
    try:
        # 深拷贝模型计算 FLOPs，防止 thop 残留 hook 导致原模型推理报错
        net_to_profile = copy.deepcopy(net_to_profile)
    except Exception:
        pass
        
    dummy = torch.randn(1, in_channels, img_size, img_size, device=device)
    try:
        flops, params = profile(net_to_profile, inputs=(dummy,), verbose=False)
        flops, params = clever_format([flops, params], '%.3f')
        logging.info(f"[ModelStats] Params: {params}, FLOPs@{img_size}x{img_size}: {flops}")
    except Exception as e:
        logging.info(f"[ModelStats] Profile failed, skip stats. Error: {e}")


# Statistical Analysis Utilities

def format_pvalue(p):
    if p < 0.001:
        return f"{p:.2e} ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f} (ns)"


def print_run_results(method_name, all_run_avg_scores, metric_names=('Dice', 'HD95', 'ASD', 'Recall'), run_seeds=None):
    """Print mean ± std for a method across N runs, and return per-metric run-level arrays.

    all_run_avg_scores: list of [num_metrics] arrays — one per run, already averaged across classes & cases.
    Returns: np.ndarray of shape [num_runs, num_metrics]
    """
    stacked = np.stack(all_run_avg_scores, axis=0)          # [N_runs, 4]
    N = stacked.shape[0]
    print(f"\n{'='*60}")
    print(f"Method: {method_name}   (N = {N} runs)")
    if run_seeds is not None:
        print(f"Seeds: {run_seeds}")
    print(f"{'='*60}")
    for i, name in enumerate(metric_names):
        mean_val = float(np.mean(stacked[:, i]))
        std_val = float(np.std(stacked[:, i], ddof=1))
        print(f"  {name:10s}: {mean_val:.4f} ± {std_val:.4f}")
    return stacked


def compare_two_methods(name_a, scores_a, name_b, scores_b,
                        metric_names=('Dice', 'HD95', 'ASD', 'Recall')):
    """Independent t-test for each metric between two methods.

    scores_a / scores_b: [N_runs, num_metrics] arrays
    """
    print(f"\n{'='*60}")
    print(f"Independent t-test:  {name_a}  vs  {name_b}")
    print(f"{'='*60}")
    for i, name in enumerate(metric_names):
        t_stat, p_val = scipy_stats.ttest_ind(scores_a[:, i], scores_b[:, i])
        print(f"  {name:10s}: t = {t_stat:+.4f},  p = {format_pvalue(p_val)}")


def paired_test_across_cases(name_a, case_metrics_a, name_b, case_metrics_b,
                             metric_names=('Dice', 'HD95', 'ASD', 'Recall')):
    """Paired t-test & Wilcoxon signed-rank test on per-case scores (same test cases, two methods).

    case_metrics_a / case_metrics_b: list of np.ndarray, each [4,] — one per test case,
                                      aligned in the same order.
    """
    arr_a = np.array(case_metrics_a)          # [N_cases, 4]
    arr_b = np.array(case_metrics_b)          # [N_cases, 4]
    if arr_a.shape != arr_b.shape:
        print(f"[WARN] Mismatched case counts: {name_a}={arr_a.shape[0]}, {name_b}={arr_b.shape[0]}. "
              "Paired test skipped.")
        return

    print(f"\n{'='*60}")
    print(f"Paired test (per-case):  {name_a}  vs  {name_b}   (N = {arr_a.shape[0]} cases)")
    print(f"{'='*60}")
    for i, mname in enumerate(metric_names):
        diffs = arr_a[:, i] - arr_b[:, i]
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1))

        # Paired t-test (parametric)
        t_stat, p_ttest = scipy_stats.ttest_rel(arr_a[:, i], arr_b[:, i])

        # Wilcoxon signed-rank test (non-parametric, preferred for small N or non-normal diffs)
        w_stat, p_wilcoxon = scipy_stats.wilcoxon(arr_a[:, i], arr_b[:, i])

        print(f"  {mname:10s}: Δ = {mean_diff:+.4f} ± {std_diff:.4f}"
              f"  |  paired-t p = {format_pvalue(p_ttest)}"
              f"  |  Wilcoxon p = {format_pvalue(p_wilcoxon)}")


def anova_across_methods(methods_scores_dict, metric_names=('Dice', 'HD95', 'ASD', 'Recall')):
    """One-way ANOVA across 2+ methods. Each value is [N_runs, num_metrics]."""
    names = list(methods_scores_dict.keys())
    arrays = list(methods_scores_dict.values())
    if len(names) < 2:
        print("ANOVA requires at least 2 methods.")
        return
    print(f"\n{'='*60}")
    print(f"One-way ANOVA across {len(names)} methods: {', '.join(names)}")
    print(f"{'='*60}")
    for i, mname in enumerate(metric_names):
        groups = [arr[:, i] for arr in arrays]
        f_stat, p_val = scipy_stats.f_oneway(*groups)
        print(f"  {mname:10s}: F = {f_stat:.4f},  p = {format_pvalue(p_val)}")


def collect_inference_metrics(args, model, test_save_path=None):
    """Run inference once and return structured metrics.

    Returns
    -------
    per_class_avg : np.ndarray  shape [num_classes-1, 4]
        Average (dice, hd95, assd, recall) for each foreground class, averaged across test cases.
    overall_avg : np.ndarray  shape [4,]
        Grand average across all classes and cases.
    avg_time_per_case : float
    case_metrics : list of np.ndarray each shape [4,]
        Per-case overall metric (dice, hd95, assd, recall) — mean across foreground classes for that case.
        Same order as the dataloader.
    """
    db_test = args.Dataset(base_dir=args.volume_path, split=args.split, list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()

    metric_list = 0.0
    total_time_all = 0.0
    case_metrics_all = []

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i, time_i = test_single_volume(image, label, model, classes=args.num_classes,
                                              patch_size=[args.img_size, args.img_size],
                                              test_save_path=test_save_path, case=case_name,
                                              z_spacing=args.z_spacing, args=args)
        metric_list += np.array(metric_i)
        total_time_all += time_i
        # Per-case overall (mean across foreground classes)
        case_overall = np.mean(np.array(metric_i), axis=0)
        case_metrics_all.append(case_overall)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_assd %f mean_recall %f time_per_slice %f' % (
            i_batch, case_name,
            case_overall[0], case_overall[1], case_overall[2], case_overall[3],
            time_i
        ))

    per_class_avg = metric_list / len(db_test)               # [num_classes-1, 4]
    overall_avg = np.mean(per_class_avg, axis=0)             # [4,]
    avg_time_per_case = total_time_all / len(db_test)

    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_assd %f mean_recall %f' % (
            i, per_class_avg[i - 1][0], per_class_avg[i - 1][1],
            per_class_avg[i - 1][2], per_class_avg[i - 1][3]))

    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_assd : %f mean_recall : %f mean_time_per_slice : %fs' % (
        overall_avg[0], overall_avg[1], overall_avg[2], overall_avg[3], avg_time_per_case))

    return per_class_avg, overall_avg, avg_time_per_case, case_metrics_all


# Single-volume test

def test_single_volume(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None, z_spacing=1, args=None):
    """MambaDSNet 单模型测试：整幅 resize -> 前向 -> resize 回原尺寸"""
    # squeeze(0) removes ONLY the batch dimension (assumed to be 0). 
    # image_np shape will be [D, H, W] if input was [1, D, H, W] (3D vol) or [1, H, W] (2D image but usually loaded as 3D vol with D=1).
    image_np, label_np = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    
    # If image is 4D [C, D, H, W] after squeeze(0) (because input was [1, C, D, H, W]), we need to be careful.
    # Usually datasets like ACDC/Synapse return [1, D, H, W] or [D, H, W].
    # But standard dataloaders often return [B, C, D, H, W] for 3D or [B, C, H, W] for 2D.
    # Based on context: Synapse/ACDC here likely returns image as [1, H, W] per slice in 2D training, 
    # BUT in inference(test_volume), it usually loads the WHOLE volume.
    # Let's assume image_np is [D, H, W].
    
    total_time = 0.0

    def infer_slice(slc_np):
        nonlocal total_time
        x, y = slc_np.shape
        resized = zoom(slc_np, (patch_size[0] / x, patch_size[1] / y), order=3) if (x, y) != tuple(patch_size) else slc_np
        device = next(net.parameters()).device
        inp = torch.as_tensor(resized, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            logits = net(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            elif isinstance(logits, list):
                logits = logits[0]
        torch.cuda.synchronize()
        t1 = time.time()
        total_time += (t1 - t0)
            
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()
        if (x, y) != tuple(patch_size):
            pred = zoom(pred, (x / patch_size[0], y / patch_size[1]), order=0)
        return pred

    if len(image_np.shape) == 3:
        prediction = np.zeros_like(label_np)
        for ind in range(image_np.shape[0]):
            prediction[ind] = infer_slice(image_np[ind])
    else:
        prediction = infer_slice(image_np)

    prediction = clean_segmentation(prediction, classes)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label_np == i))

    if test_save_path is not None:
        if not sitk_available and not args.save_png: # Only raise if no save method works
            raise ImportError("SimpleITK is required to save NIfTI; install SimpleITK or disable --is_savenii")
        
        # Save PNG: slice by slice
        if args.save_png:
            png_dir = os.path.join(test_save_path, case)
            os.makedirs(png_dir, exist_ok=True)
            
            # Define palette
            palette = np.zeros((256, 3), dtype=np.uint8)
            np.random.seed(42)
            for c in range(1, classes):
                palette[c] = np.random.randint(50, 255, size=3)
            
            def get_overlay_image(mask, bg_img, pal):
                mask_pil = Image.fromarray(mask)
                mask_pil.putpalette(pal.flatten())
                mask_rgb = mask_pil.convert("RGB")
                
                # Alpha channel: 0 for background, 100 for foreground
                alpha = np.zeros_like(mask, dtype=np.uint8)
                alpha[mask > 0] = 100 # Adjust opacity here (0-255)
                
                mask_rgba = mask_rgb.convert("RGBA")
                mask_rgba.putalpha(Image.fromarray(alpha))
                
                return Image.alpha_composite(bg_img, mask_rgba)

            for ind in range(image_np.shape[0]):
                # Rescale image to 0-255
                slc_img = image_np[ind]
                if slc_img.max() > slc_img.min():
                    slc_img = (slc_img - slc_img.min()) / (slc_img.max() - slc_img.min()) * 255
                else:
                    slc_img = slc_img * 0
                slc_img = slc_img.astype(np.uint8)
                
                # Base image (Background)
                img_bg = Image.fromarray(slc_img).convert("RGBA")
                
                # Predict Overlay
                slc_pred = prediction[ind].astype(np.uint8)
                ov_pred = get_overlay_image(slc_pred, img_bg, palette)

                # GT Overlay
                slc_gt = label_np[ind].astype(np.uint8)
                ov_gt = get_overlay_image(slc_gt, img_bg, palette)

                Image.fromarray(slc_img).save(os.path.join(png_dir, f"{ind:03d}_img.png"))
                ov_pred.save(os.path.join(png_dir, f"{ind:03d}_pred.png"))
                ov_gt.save(os.path.join(png_dir, f"{ind:03d}_gt.png"))
        
        # Save NIfTI
        if args.is_savenii and sitk_available:
            image_save = np.asarray(image_np, dtype=np.float32)
            pred_save = np.asarray(prediction, dtype=np.float32)
            label_save = np.asarray(label_np, dtype=np.float32)
            img_itk = sitk.GetImageFromArray(image_save)
            prd_itk = sitk.GetImageFromArray(pred_save)
            lab_itk = sitk.GetImageFromArray(label_save)
            img_itk.SetSpacing((1, 1, z_spacing))
            prd_itk.SetSpacing((1, 1, z_spacing))
            lab_itk.SetSpacing((1, 1, z_spacing))
            sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
            sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    
    num_slices = image_np.shape[0] if len(image_np.shape) == 3 else 1
    avg_time = total_time / num_slices
    return metric_list, avg_time


def _build_model(args, arch_override=None):
    """Build a fresh model instance based on arch args (no weight loading)."""
    arch = arch_override if arch_override is not None else args.arch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arch == 'mambadsnet':
        return MambaDSNet(in_ch=1, num_classes=args.num_classes, base_c=args.base_c).to(device)
    elif arch == 'mamba_ab1':
        return MambaAblationOnly(in_ch=1, num_classes=args.num_classes, base_c=args.base_c).to(device)
    elif arch == 'mamba_ab2':
        return MambaAblationCNN(in_ch=1, num_classes=args.num_classes, base_c=args.base_c).to(device)
    elif arch == 'mamba_csmi_nodir':
        return MambaAblationCSMINoDir(in_ch=1, num_classes=args.num_classes, base_c=args.base_c).to(device)
    elif arch == 'mamba_csmi_nores':
        return MambaAblationCSMINoResidual(in_ch=1, num_classes=args.num_classes, base_c=args.base_c).to(device)
    elif arch == 'mamba_ca_self':
        return MambaAblationCASelf(in_ch=1, num_classes=args.num_classes, base_c=args.base_c).to(device)


def _load_model_for_seed(args, model_path, arch_override=None):
    """Build model and load checkpoint, return eval model."""
    net = _build_model(args, arch_override=arch_override)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    state = _load_state_dict_flexible(model_path)
    incompatible = net.load_state_dict(state, strict=False)
    print(f"Model loaded from {model_path}")
    print(f"Missing keys: {len(incompatible.missing_keys)}, unexpected keys: {len(incompatible.unexpected_keys)}")
    net.eval()
    return net


# Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_path', type=str,
                        default='./data/Synapse', help='root dir for validation volume data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='dataset name: Synapse or ACDC')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
    parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
    parser.add_argument('--save_png', action="store_true", help='save prediction results as png')
    parser.add_argument('--test_save_dir', type=str, default='./test_predictions', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--base_c', type=int, default=32, help='base channel count for MambaDSNet')
    parser.add_argument('--model_path', type=str, default=None, 
                help='path to MambaDSNet weight; if None, uses dataset-specific default')
    parser.add_argument('--num_workers', type=int, default=1, help='num workers for dataloader')
    parser.add_argument('--arch', type=str, default='mambadsnet', choices=['mambadsnet', 'mamba_ab1', 'mamba_ab2', 'mamba_csmi_nodir', 'mamba_csmi_nores', 'mamba_ca_self'], help='model arch to test')
    parser.add_argument('--print_model_stats', action='store_true', help='print Params & FLOPs (requires thop)')
    # Multi-run & statistical analysis
    parser.add_argument('--num_runs', type=int, default=1,
                        help='number of runs (different seeds) for mean±std reporting (default: 1 = single run)')
    parser.add_argument('--seed_list', type=str, default='1234,2345,3456,4567,5678',
                        help='comma-separated seed list for multi-run (default: 1234,2345,3456,4567,5678)')
    parser.add_argument('--compare_arch', type=str, default=None,
                        help='arch to compare against with t-test (e.g. mamba_ab1)')
    parser.add_argument('--compare_model_path', type=str, default=None,
                        help='explicit baseline model path template (overrides dataset config)')
    parser.add_argument('--output_stats_file', type=str, default=None,
                        help='optional path to save aggregated stats as JSON')
    parser.add_argument('--save_case_metrics', type=str, default=None,
                        help='save main model per-case metrics to this JSON file')
    parser.add_argument('--reference_metrics', type=str, default=None,
                        help='load saved per-case metrics as main model (skip re-inference of main model)')

    args = parser.parse_args()

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
        'ACDC': {
            'Dataset': ACDC_dataset,
            'volume_path': './data/ACDC/test',
            'list_dir': './lists/lists_ACDC',
            'num_classes': 4,
            'z_spacing': 1,
            'split': 'test',
            'model_paths': {
                'mambadsnet': './model/ACDC/mambadsnet_ACDC224_epo200_bs8_lr0.01_s{seed}/edge_0.2/best_model_valdice_edge_0.2.pth',
                'mamba_ab1': './model/ACDC/mamba_ab1_ACDC224_epo200_bs8_lr0.01_s{seed}/noedge/best_model_valdice_noedge.pth',
                'mamba_ab2': './model/ACDC/mamba_ab2_ACDC224_epo200_bs8_lr0.01_s{seed}/noedge/best_model_valdice_noedge.pth',
                'mamba_csmi_nodir': './model/ACDC/mamba_csmi_nodir_ACDC224_epo200_bs8_lr0.01_s{seed}/noedge/best_model_valdice_noedge.pth',
                'mamba_csmi_nores': './model/ACDC/mamba_csmi_nores_ACDC224_epo200_bs8_lr0.01_s{seed}/noedge/best_model_valdice_noedge.pth',
                'mamba_ca_self': './model/ACDC/mamba_ca_self_ACDC224_epo200_bs8_lr0.01_s{seed}/noedge/best_model_valdice_noedge.pth',
            },
        },

        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': './data/Synapse',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
            'split': 'test_vol',
            'model_paths': {
                'mambadsnet': './model/Synapse/mambadsnet_Synapse224_epo150_bs24_lr0.02_s{seed}/edge_0.2/best_model_edge_0.2.pth',
                'mamba_ab1': './model/Synapse/mamba_ab1_Synapse224_epo150_bs24_lr0.02_s{seed}/noedge/best_model_noedge.pth',
                'mamba_ab2': './model/Synapse/mamba_ab2_Synapse224_epo150_bs24_lr0.02_s{seed}/noedge/best_model_noedge.pth',
                'mamba_csmi_nodir': './model/Synapse/mamba_csmi_nodir_Synapse224_epo150_bs24_lr0.02_s{seed}/noedge/best_model_noedge.pth',
                'mamba_csmi_nores': './model/Synapse/mamba_csmi_nores_Synapse224_epo150_bs24_lr0.02_s{seed}/noedge/best_model_noedge.pth',
                'mamba_ca_self': './model/Synapse/mamba_ca_self_Synapse224_epo150_bs24_lr0.02_s{seed}/noedge/best_model_noedge.pth',
            },

        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.split = dataset_config[dataset_name]['split']

    # Resolve model path template
    if args.model_path is None:
        model_path_template = dataset_config[dataset_name]['model_paths'][args.arch]
    else:
        model_path_template = args.model_path

    # Setup logging
    log_folder = './test_log/test'
    os.makedirs(log_folder, exist_ok=True)
    log_filename = f"{args.dataset}_{args.arch}.txt"
    logging.basicConfig(filename=os.path.join(log_folder, log_filename), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Parse seed list
    seed_list = [int(s.strip()) for s in args.seed_list.split(',')]
    if args.num_runs > 1:
        if len(seed_list) < args.num_runs:
            raise ValueError(f"seed_list has {len(seed_list)} seeds but num_runs={args.num_runs}")
        seed_list = seed_list[:args.num_runs]
    else:
        seed_list = [args.seed]

    # Determine save path (only used if saving NIfTI/PNG; multi-run disables saving to avoid overwrites)
    if (args.is_savenii or args.save_png) and args.num_runs == 1:
        test_save_path = os.path.join(args.test_save_dir, args.dataset)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        if (args.is_savenii or args.save_png) and args.num_runs > 1:
            print("[WARN] Saving disabled in multi-run mode to avoid overwriting outputs.")
        test_save_path = None

    # Run / load main method
    logging.info(str(args))
    metric_names = ('Dice', 'HD95', 'ASD', 'Recall')

    import json as _json

    if args.reference_metrics is not None:
        # Skip main model inference — load pre-saved per-case metrics
        print(f"\n--- Loading saved case metrics from {args.reference_metrics} ---")
        with open(args.reference_metrics, 'r') as f:
            ref_data = _json.load(f)
        main_case_metrics = [np.array(v) for v in ref_data['case_metrics']]
        main_overall = np.array(ref_data['overall_avg'])
        print(f"  Loaded {len(main_case_metrics)} cases  "
              f"Dice={main_overall[0]:.4f}  HD95={main_overall[1]:.4f}  "
              f"ASD={main_overall[2]:.4f}  Recall={main_overall[3]:.4f}")
        all_overall = [main_overall]
        main_scores = np.stack(all_overall, axis=0)
        _ = print_run_results(args.arch, all_overall, metric_names, run_seeds=[0])
    else:
        logging.info(f"Dataset: {args.dataset} | Arch: {args.arch} | Runs: {len(seed_list)} | Seeds: {seed_list}")

        all_overall = []  # list of overall_avg arrays [4,]
        all_times = []    # list of avg_time_per_case floats
        all_per_class = []  # list of per_class_avg arrays [num_classes-1, 4]
        main_case_metrics = None

        for run_idx, seed in enumerate(seed_list):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            model_path = model_path_template.format(seed=seed)
            print(f"\n--- Run {run_idx+1}/{len(seed_list)} (seed={seed}) ---")
            net = _load_model_for_seed(args, model_path)

            if args.print_model_stats and run_idx == 0:
                in_c = 1
                compute_model_stats(net, args.img_size, in_channels=in_c)

            per_class_avg, overall_avg, avg_time, case_metrics = collect_inference_metrics(args, net, test_save_path)
            all_overall.append(overall_avg)
            all_times.append(avg_time)
            all_per_class.append(per_class_avg)
            if run_idx == 0:
                main_case_metrics = case_metrics
            print(f"  Run {run_idx+1}  Dice={overall_avg[0]:.4f}  HD95={overall_avg[1]:.4f}  "
                  f"ASD={overall_avg[2]:.4f}  Recall={overall_avg[3]:.4f}  time={avg_time:.4f}s")

        main_scores = print_run_results(args.arch, all_overall, metric_names, run_seeds=seed_list)

        # Print time_per_slice mean ± std across runs
        if len(all_times) > 1:
            times_arr = np.array(all_times)
            print(f"  {'time/slice':10s}: {times_arr.mean():.4f} ± {times_arr.std(ddof=1):.4f} s")

        # Print per-organ (per-class) mean ± std across runs
        if len(all_per_class) > 1:
            per_class_stacked = np.stack(all_per_class, axis=0)  # [N_runs, num_classes-1, 4]
            num_cls = per_class_stacked.shape[1]
            print(f"\n  --- Per-organ metrics across {len(all_per_class)} runs ---")
            for c in range(num_cls):
                cls_scores = per_class_stacked[:, c, :]  # [N_runs, 4]
                parts = []
                for m, mname in enumerate(metric_names):
                    mu = float(np.mean(cls_scores[:, m]))
                    sd = float(np.std(cls_scores[:, m], ddof=1))
                    parts.append(f"{mname}={mu:.4f}±{sd:.4f}")
                print(f"  Organ {c+1}:  " + "  ".join(parts))

        # Save per-case metrics if requested
        if args.save_case_metrics is not None and main_case_metrics is not None:
            save_data = {
                'arch': args.arch,
                'dataset': args.dataset,
                'overall_avg': [float(v) for v in all_overall[0]],
                'case_metrics': [[float(v) for v in m] for m in main_case_metrics],
            }
            with open(args.save_case_metrics, 'w') as f:
                _json.dump(save_data, f, indent=2)
            print(f"\nPer-case metrics saved to {args.save_case_metrics}")

    # Compare with baseline (if requested)
    methods_for_anova = {args.arch: main_scores}

    if args.compare_arch is not None:
        compare_arch = args.compare_arch
        if args.compare_model_path:
            baseline_template = args.compare_model_path
        else:
            baseline_template = dataset_config[dataset_name]['model_paths'].get(compare_arch)
            if baseline_template is None:
                print(f"[WARN] No model path found for {compare_arch} in dataset config. "
                      f"Use --compare_model_path explicitly.")
                baseline_template = None

        if baseline_template is not None:
            baseline_overall = []
            baseline_times = []
            baseline_per_class = []
            baseline_case_metrics = None
            print(f"\n{'='*60}")
            print(f"Running baseline: {compare_arch}")
            print(f"{'='*60}")
            for run_idx, seed in enumerate(seed_list):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                model_path = baseline_template.format(seed=seed)
                print(f"--- Baseline run {run_idx+1}/{len(seed_list)} (seed={seed}) ---")
                net = _load_model_for_seed(args, model_path, arch_override=compare_arch)

                per_class_avg, overall_avg, avg_time, case_metrics = collect_inference_metrics(args, net, test_save_path=None)
                baseline_overall.append(overall_avg)
                baseline_times.append(avg_time)
                baseline_per_class.append(per_class_avg)
                if run_idx == 0:
                    baseline_case_metrics = case_metrics
                print(f"  Run {run_idx+1}  Dice={overall_avg[0]:.4f}  HD95={overall_avg[1]:.4f}  "
                      f"ASD={overall_avg[2]:.4f}  Recall={overall_avg[3]:.4f}")

            baseline_scores = print_run_results(compare_arch, baseline_overall, metric_names, run_seeds=seed_list)

            if len(baseline_times) > 1:
                times_arr = np.array(baseline_times)
                print(f"  {'time/slice':10s}: {times_arr.mean():.4f} ± {times_arr.std(ddof=1):.4f} s")

            if len(baseline_per_class) > 1:
                per_class_stacked = np.stack(baseline_per_class, axis=0)
                num_cls = per_class_stacked.shape[1]
                print(f"\n  --- Per-organ metrics across {len(baseline_per_class)} runs ---")
                for c in range(num_cls):
                    cls_scores = per_class_stacked[:, c, :]
                    parts = []
                    for m, mname in enumerate(metric_names):
                        mu = float(np.mean(cls_scores[:, m]))
                        sd = float(np.std(cls_scores[:, m], ddof=1))
                        parts.append(f"{mname}={mu:.4f}±{sd:.4f}")
                    print(f"  Organ {c+1}:  " + "  ".join(parts))
            compare_two_methods(args.arch, main_scores, compare_arch, baseline_scores, metric_names)
            methods_for_anova[compare_arch] = baseline_scores

            # Paired test across cases (uses first-run per-case scores)
            if main_case_metrics is not None and baseline_case_metrics is not None:
                paired_test_across_cases(args.arch, main_case_metrics,
                                         compare_arch, baseline_case_metrics,
                                         metric_names)

    # ANOVA (if 2+ methods available)
    if len(methods_for_anova) >= 2:
        anova_across_methods(methods_for_anova, metric_names)

    # Optionally save to JSON
    if args.output_stats_file:
        import json
        stats_out = {}
        for method_name, scores_arr in methods_for_anova.items():
            stats_out[method_name] = {}
            for i, mname in enumerate(metric_names):
                stats_out[method_name][mname] = {
                    'mean': float(np.mean(scores_arr[:, i])),
                    'std': float(np.std(scores_arr[:, i], ddof=1)),
                    'values': [float(v) for v in scores_arr[:, i]],
                }
        with open(args.output_stats_file, 'w') as f:
            json.dump(stats_out, f, indent=2)
        print(f"\nAggregated stats saved to {args.output_stats_file}")
