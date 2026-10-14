"""
消融实验显著性检验 (Ablation Study Significance Testing)
========================================================
对 7 个模型变体进行配对显著性检验 (paired t-test + Wilcoxon),
检验相邻变体之间的性能提升是否显著。

数据来源: test_log/test/ 下的 3-run 测试日志
- 每个变体在 3 个随机种子 (1234, 2345, 3456) 上测试相同的 12 个 Synapse 测试样本
- 对每个 case，取其 3 个种子的 Dice/HD95 均值作为该 case 的代表值
- 在 12 个 case 对上做配对检验
"""

import re
import numpy as np
from scipy import stats

# ============================================================
# 配置: 模型变体名 → (日志文件, 对应的3-run段落索引, 标签)
# ============================================================

# 变体定义 (按表格顺序)
VARIANTS = [
    {
        "name": "Mamba-only (single stream)",
        "file": "test_log/test/Synapse_mamba_ab1.txt",
        "section_idx": 1,   # 第2个Namespace段落 (第1个是单次运行, 第2个是3-run)
        "arch": "mamba_ab1",
        "ca": "", "residual": "", "dirgate": "", "eaat": "",
    },
    {
        "name": "+ Element-wise Sum",
        "file": "test_log/test/Synapse_mamba_ab2.txt",
        "section_idx": 1,   # 第2个Namespace段落是3-run
        "arch": "mamba_ab2",
        "ca": "", "residual": "", "dirgate": "", "eaat": "",
    },
    {
        "name": "+ Independent CA + Sum (self)",
        "file": "test_log/test/Synapse_mamba_ca_self.txt",
        "section_idx": 0,   # 第1个Namespace段落即为3-run
        "arch": "mamba_ca_self",
        "ca": "self", "residual": "", "dirgate": "", "eaat": "",
    },
    {
        "name": "Cross-stream CA + Residual",
        "file": "test_log/test/Synapse_mamba_csmi_nodir.txt",
        "section_idx": 0,   # 第1个Namespace段落即为3-run
        "arch": "mamba_csmi_nodir",
        "ca": "cross", "residual": "√", "dirgate": "", "eaat": "",
    },
    {
        "name": "Cross-stream CA + DirGate",
        "file": "test_log/test/Synapse_mamba_csmi_nores.txt",
        "section_idx": 0,   # 第1个Namespace段落即为3-run
        "arch": "mamba_csmi_nores",
        "ca": "cross", "residual": "", "dirgate": "√", "eaat": "",
    },
    {
        "name": "+ CSMI (full)",
        "file": "test_log/test/Synapse_mambadsnet.txt",
        "section_idx": 0,   # noedge 3-run (第一个Namespace段落)
        "arch": "mambadsnet (noedge)",
        "ca": "cross", "residual": "√", "dirgate": "√", "eaat": "",
    },
    {
        "name": "+ CSMI (full) + EAAT",
        "file": None,   # edge_0.2 数据内嵌于下方 inline_data
        "section_idx": 0,
        "arch": "mambadsnet (edge_0.2)",
        "ca": "cross", "residual": "√", "dirgate": "√", "eaat": "√",
        # 手动嵌入 edge_0.2 3-run per-case 数据
        "inline_data": [
            # Run 1 (seed=1234)
            {
                "case0008": {"dice": 0.659683, "hd95": 9.312802},
                "case0022": {"dice": 0.912697, "hd95": 2.672283},
                "case0038": {"dice": 0.785292, "hd95": 14.013696},
                "case0036": {"dice": 0.821844, "hd95": 18.690275},
                "case0032": {"dice": 0.859285, "hd95": 7.600475},
                "case0002": {"dice": 0.865888, "hd95": 8.251910},
                "case0029": {"dice": 0.789131, "hd95": 24.218243},
                "case0003": {"dice": 0.747421, "hd95": 32.754411},
                "case0001": {"dice": 0.757049, "hd95": 28.262272},
                "case0004": {"dice": 0.779928, "hd95": 13.721184},
                "case0025": {"dice": 0.779745, "hd95": 18.702826},
                "case0035": {"dice": 0.903144, "hd95": 2.618942},
            },
            # Run 2 (seed=2345)
            {
                "case0008": {"dice": 0.657213, "hd95": 9.858402},
                "case0022": {"dice": 0.908616, "hd95": 2.933629},
                "case0038": {"dice": 0.838236, "hd95": 5.263696},
                "case0036": {"dice": 0.865984, "hd95": 13.872536},
                "case0032": {"dice": 0.889387, "hd95": 4.544912},
                "case0002": {"dice": 0.857204, "hd95": 7.501845},
                "case0029": {"dice": 0.774362, "hd95": 19.108362},
                "case0003": {"dice": 0.796528, "hd95": 17.861800},
                "case0001": {"dice": 0.744544, "hd95": 25.664258},
                "case0004": {"dice": 0.778832, "hd95": 13.722352},
                "case0025": {"dice": 0.763822, "hd95": 25.595188},
                "case0035": {"dice": 0.898899, "hd95": 2.572328},
            },
            # Run 3 (seed=3456)
            {
                "case0008": {"dice": 0.671078, "hd95": 8.736159},
                "case0022": {"dice": 0.900882, "hd95": 3.275929},
                "case0038": {"dice": 0.830630, "hd95": 5.434162},
                "case0036": {"dice": 0.875880, "hd95": 14.161782},
                "case0032": {"dice": 0.795589, "hd95": 7.885382},
                "case0002": {"dice": 0.864450, "hd95": 6.139250},
                "case0029": {"dice": 0.759558, "hd95": 15.447973},
                "case0003": {"dice": 0.810363, "hd95": 27.205092},
                "case0001": {"dice": 0.742145, "hd95": 21.377449},
                "case0004": {"dice": 0.736281, "hd95": 11.211702},
                "case0025": {"dice": 0.748089, "hd95": 30.945690},
                "case0035": {"dice": 0.889118, "hd95": 3.227316},
            },
        ],
    },
]


def parse_all_runs(filepath):
    """
    解析日志文件中的全部 3-run 测试段落。
    返回: list of list of dicts
        runs[run_idx][case_name] = {"dice": float, "hd95": float}
    """
    with open(filepath) as f:
        lines = f.readlines()

    # 找到所有 Namespace 行
    ns_indices = [i for i, l in enumerate(lines) if "Namespace" in l]

    all_runs_list = []  # 每个元素是一个3-run段落

    for ns_i, ns_start in enumerate(ns_indices):
        # 确定该段落的结束位置
        ns_end = ns_indices[ns_i + 1] if ns_i + 1 < len(ns_indices) else len(lines)

        # 检查这一段是否包含 num_runs=3
        header = "".join(lines[ns_start:ns_start + 5])
        if "num_runs=3" not in header and "'num_runs': 3" not in header:
            # 可能是单次运行的前导段落，跳过
            # 但也可能是跨行的 num_runs，放宽条件
            section_text = "".join(lines[ns_start:ns_end])
            if "num_runs=3" not in section_text and "'num_runs': 3" not in section_text:
                continue

        # 提取 seed_list 确认是3-run
        section_text = "".join(lines[ns_start:min(ns_start + 10, ns_end)])

        # 提取该段落中所有的 per-case 行
        cases_in_section = []
        current_run_cases = {}
        case_order = []

        for line in lines[ns_start + 1:ns_end]:
            # 跳过新的 Namespace (不应出现但安全检查)
            if "Namespace" in line:
                break

            m = re.search(
                r"idx \d+ case (\S+) mean_dice ([\d.]+) mean_hd95 ([\d.]+)",
                line,
            )
            if m:
                case_name = m.group(1)
                dice = float(m.group(2))
                hd95 = float(m.group(3))

                if case_name not in current_run_cases:
                    current_run_cases[case_name] = {"dice": dice, "hd95": hd95}
                    if case_name not in case_order:
                        case_order.append(case_name)
                else:
                    # 遇到重复 case → 说明新的一轮开始了
                    cases_in_section.append(current_run_cases)
                    current_run_cases = {case_name: {"dice": dice, "hd95": hd95}}

            # 检测 "Testing performance" 行 (表示一轮结束的标记)
            if "Testing performance in best val model" in line:
                if len(current_run_cases) == 12:
                    cases_in_section.append(current_run_cases)
                    current_run_cases = {}

        # 收尾
        if len(current_run_cases) == 12:
            cases_in_section.append(current_run_cases)

        if len(cases_in_section) >= 3:
            all_runs_list.append(cases_in_section[:3])  # 只取前3轮

    return all_runs_list


def extract_per_case_means(runs_section):
    """
    对单个3-run段落的3轮结果，计算每个case的Dice和HD95均值。

    参数:
        runs_section: list of 3 dicts, 每个 dict 是 {case_name: {"dice": ..., "hd95": ...}}

    返回:
        case_names: list of str (12 cases)
        dice_means: np.array (12,)
        hd95_means: np.array (12,)
        dice_per_run: np.array (3, 12)
        hd95_per_run: np.array (3, 12)
    """
    # 获取case名称 (所有run应该相同)
    case_names = sorted(runs_section[0].keys())

    dice_per_run = np.zeros((3, 12))
    hd95_per_run = np.zeros((3, 12))

    for i, run in enumerate(runs_section):
        for j, case in enumerate(case_names):
            dice_per_run[i, j] = run[case]["dice"]
            hd95_per_run[i, j] = run[case]["hd95"]

    dice_means = dice_per_run.mean(axis=0)
    hd95_means = hd95_per_run.mean(axis=0)

    return case_names, dice_means, hd95_means, dice_per_run, hd95_per_run


def format_pvalue(p):
    """格式化 p 值，添加显著性标记。"""
    if p < 0.001:
        return f"{p:.2e} ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f} (ns)"


def paired_comparison(arr_a, arr_b, name_a, name_b, metric_name, case_names):
    """
    配对比较两个变体在某个指标上的表现。
    同时输出每个 case 的差值详情。
    """
    diffs = arr_a - arr_b
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(arr_a, arr_b)

    # Wilcoxon signed-rank test
    # scipy 1.10+ 用法
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(arr_a, arr_b, method="exact")
    except (TypeError, ValueError):
        # fallback for older scipy or when exact fails
        w_stat, p_wilcoxon = stats.wilcoxon(arr_a, arr_b)

    print(f"\n  {'─' * 55}")
    print(f"  {metric_name}: {name_a} vs {name_b}")
    print(f"  {'─' * 55}")
    print(f"  {'Case':<12s}  {'A':>8s}  {'B':>8s}  {'Δ (A-B)':>10s}")
    print(f"  {'─' * 42}")

    sorted_cases = sorted(zip(case_names, arr_a, arr_b, diffs), key=lambda x: x[0])
    for case, a, b, d in sorted_cases:
        direction = "↑" if d > 0 else ("↓" if d < 0 else "=")
        print(f"  {case:<12s}  {a:8.4f}  {b:8.4f}  {d:+10.4f} {direction}")

    print(f"  {'─' * 42}")
    print(f"  {'Mean Δ':<12s}  {np.mean(arr_a):8.4f}  {np.mean(arr_b):8.4f}  {mean_diff:+10.4f} ± {std_diff:.4f}")
    print(f"  {'':12s}  Paired t-test:  t = {t_stat:+.4f},  p = {format_pvalue(p_ttest)}")
    print(f"  {'':12s}  Wilcoxon:       W = {w_stat:.1f},  p = {format_pvalue(p_wilcoxon)}")

    return {
        "metric": metric_name,
        "mean_a": np.mean(arr_a),
        "mean_b": np.mean(arr_b),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "p_ttest": p_ttest,
        "w_stat": w_stat,
        "p_wilcoxon": p_wilcoxon,
    }


def print_summary_table(all_results):
    """打印汇总表格。"""
    print(f"\n\n{'=' * 90}")
    print(f"  显著性检验汇总 — Synapse Ablation Study")
    print(f"{'=' * 90}")

    print(f"\n  {'Comparison':<48s}  {'Metric':<6s}  {'t-stat':>8s}  {'p(t-test)':>12s}  {'p(Wilcoxon)':>14s}")
    print(f"  {'─' * 90}")

    for r in all_results:
        label = f"{r['name_a'][:22]} vs {r['name_b'][:22]}"
        print(f"  {label:<48s}  {r['metric']:<6s}  {r['t_stat']:+8.4f}  {format_pvalue(r['p_ttest']):>12s}  {format_pvalue(r['p_wilcoxon']):>14s}")


def main():
    print("=" * 90)
    print("  消融实验显著性检验 — Ablation Study Statistical Significance Testing")
    print("  数据集: Synapse (12 test volumes, 3 seeds each)")
    print("  方法: Per-case 均值 (跨3种子) → 配对 t-test + Wilcoxon signed-rank")
    print("=" * 90)

    # ---- 第1步: 提取所有变体的 per-case 数据 ----
    all_data = {}

    for v in VARIANTS:
        print(f"\n\n{'█' * 60}")
        print(f"  解析: {v['name']}  [{v['arch']}]")

        if v.get("inline_data"):
            # 使用内嵌数据
            runs_section = v["inline_data"]
            print(f"  数据来源: 手动嵌入 (edge_0.2 测试输出)")
        else:
            print(f"  文件: {v['file']}")
            runs_list = parse_all_runs(v["file"])
            print(f"  找到 {len(runs_list)} 个 3-run 段落")

            if v["section_idx"] >= len(runs_list):
                print(f"  警告: section_idx={v['section_idx']} 超出范围 ({len(runs_list)} 个段落), 使用最后一个")
                section_idx = len(runs_list) - 1
            else:
                section_idx = v["section_idx"]

            runs_section = runs_list[section_idx]

        case_names, dice_means, hd95_means, dice_per_run, hd95_per_run = \
            extract_per_case_means(runs_section)

        if not v.get("inline_data"):
            print(f"  使用段落 {section_idx+1}/{len(runs_list)}")
        print(f"  Per-run Dice: {np.mean(dice_per_run, axis=1)}")
        print(f"  Mean Dice: {np.mean(dice_means):.4f} ± {np.std(np.mean(dice_per_run, axis=1), ddof=1):.4f}")
        print(f"  Mean HD95: {np.mean(hd95_means):.4f} ± {np.std(np.mean(hd95_per_run, axis=1), ddof=1):.4f}")

        all_data[v["name"]] = {
            "case_names": case_names,
            "dice_means": dice_means,
            "hd95_means": hd95_means,
            "dice_per_run": dice_per_run,
            "hd95_per_run": hd95_per_run,
        }

    # ---- 第2步: 配对比较 (相邻行) ----
    print(f"\n\n{'#' * 90}")
    print(f"  #  逐对比较 (Pairwise Comparisons)")
    print(f"{'#' * 90}")

    all_results = []

    for i in range(len(VARIANTS) - 1):
        name_a = VARIANTS[i]["name"]
        name_b = VARIANTS[i + 1]["name"]

        print(f"\n{'=' * 90}")
        print(f"  比较 {i+1}: {name_a}")
        print(f"         vs {name_b}")
        print(f"{'=' * 90}")

        data_a = all_data[name_a]
        data_b = all_data[name_b]

        # 验证 case 名称对齐
        assert data_a["case_names"] == data_b["case_names"], \
            f"Case mismatch: {data_a['case_names']} vs {data_b['case_names']}"

        cases = data_a["case_names"]

        # Dice 比较
        r_dice = paired_comparison(
            data_a["dice_means"], data_b["dice_means"],
            name_a, name_b, "Dice", cases,
        )
        r_dice["name_a"] = f"[{i+1}] {name_a}"
        r_dice["name_b"] = f"[{i+2}] {name_b}"
        all_results.append(r_dice)

        # HD95 比较
        r_hd95 = paired_comparison(
            data_a["hd95_means"], data_b["hd95_means"],
            name_a, name_b, "HD95", cases,
        )
        r_hd95["name_a"] = f"[{i+1}] {name_a}"
        r_hd95["name_b"] = f"[{i+2}] {name_b}"
        all_results.append(r_hd95)

    # ---- 第3步: 汇总 ----
    print_summary_table(all_results)

    # ---- 第4步: 额外比较 - 第1行 vs 第7行 (最终模型) ----
    print(f"\n\n{'#' * 90}")
    print(f"  #  额外: Mamba-only vs Full Model (CSMI + EAAT)")
    print(f"{'#' * 90}")

    name_first = VARIANTS[0]["name"]
    name_last = VARIANTS[-1]["name"]
    data_first = all_data[name_first]
    data_last = all_data[name_last]

    r_dice_full = paired_comparison(
        data_first["dice_means"], data_last["dice_means"],
        name_first, name_last, "Dice", data_first["case_names"],
    )
    r_hd95_full = paired_comparison(
        data_first["hd95_means"], data_last["hd95_means"],
        name_first, name_last, "HD95", data_first["case_names"],
    )

    print(f"\n\n{'=' * 90}")
    print(f"  所有检验完成。")
    print(f"{'=' * 90}")

    return all_data, all_results


if __name__ == "__main__":
    all_data, all_results = main()
