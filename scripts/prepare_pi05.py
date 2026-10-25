#!/usr/bin/env python3
"""准备π0.5数据划分、训练集统计和运行配置；绝不复制或修改原始数据。"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil

REPO = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(config, rules, output, runtime, steps_override=None):
    import numpy as np
    import pyarrow.parquet as pq
    cfg = json.loads(json.dumps(config))
    root = (REPO / cfg['dataset']['root']).resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    if info['robot_type'] != 'realman' or info['features']['observation.state']['shape'] != [7] or info['features']['action']['shape'] != [7]:
        raise ValueError('此配置只接受RM65的7维state/action数据，不接受10维派生集或SO100')
    grouped = defaultdict(list)
    digest = hashlib.sha256()
    for path in sorted((root / 'data').rglob('*.parquet')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
        for row in pq.read_table(path).to_pylist():
            grouped[int(row['episode_index'])].append(row)
    if len(grouped) != info['total_episodes'] or sum(map(len, grouped.values())) != info['total_frames']:
        raise ValueError('真实数据数量与info不一致，停止准备')
    manual = set(rules['exclude_episodes'])
    review = set(rules['review_candidates']) if rules['quarantine_review_candidates'] else set()
    unknown = (manual | review) - set(grouped)
    if unknown:
        raise ValueError(f'清单中的episode不存在：{sorted(unknown)}')
    excluded = {}; eligible = {}; equivalent = defaultdict(list)
    for ep, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r['frame_index'])
        s = np.asarray([r['observation.state'] for r in rows], np.float32)
        a = np.asarray([r['action'] for r in rows], np.float32)
        reason = []
        if ep in manual: reason.append('manual_exclude')
        if ep in review: reason.append('review_quarantine_not_confirmed_failure')
        if s.shape != (len(rows), 7) or a.shape != s.shape: reason.append('bad_shape')
        elif not np.isfinite(s).all() or not np.isfinite(a).all(): reason.append('nonfinite')
        else:
            if len(rows) < rules['minimum_frames']: reason.append('short')
            if np.any(np.all(s[:, :6] == 0, axis=1)) or np.any(np.all(a[:, :6] == 0, axis=1)): reason.append('zero_joint_feedback')
            if np.any(s[:, 6] < 0) or np.any(s[:, 6] > 1) or np.any(a[:, 6] < 0) or np.any(a[:, 6] > 1): reason.append('gripper_out_of_range')
            if np.all(a[:, 6] >= rules['closed_threshold']): reason.append('never_closed_heuristic')
            if np.max(np.ptp(s[:, :6], axis=0)) < rules['minimum_joint_range_deg']: reason.append('stationary_heuristic')
        if reason:
            excluded[ep] = reason
        else:
            eligible[ep] = (s, a)
            signature = hashlib.sha256(s.tobytes() + a.tobytes()).hexdigest()
            equivalent[signature].append(ep)
    # 完全相同的数值轨迹必须分在同一侧；未声称解决所有裁剪/重采样近重复。
    groups = sorted(equivalent.values(), key=lambda g: g[0])
    random.Random(rules['seed']).shuffle(groups)
    target = max(1, round(len(eligible) * rules['validation_fraction']))
    validation = []
    for group in groups:
        if len(validation) >= target: break
        validation.extend(group)
    train = sorted(set(eligible) - set(validation))
    validation.sort()
    if not train or not validation: raise ValueError('训练或验证划分为空')
    if not 0 < rules['validation_fraction'] < 1: raise ValueError('validation_fraction必须在0与1之间')
    train_frames = sum(len(eligible[e][0]) for e in train)
    stats = json.loads((root / 'meta/stats.json').read_text())
    for key, col in [('observation.state', 0), ('action', 1)]:
        x = np.concatenate([eligible[e][col] for e in train]).astype(np.float64)
        stats[key] = {'min': x.min(0).tolist(), 'max': x.max(0).tolist(),
                      'mean': x.mean(0).tolist(), 'std': x.std(0).tolist(), 'count': [len(x)]}
        for name, percentile in [('q01', 1), ('q10', 10), ('q50', 50), ('q90', 90), ('q99', 99)]:
            stats[key][name] = np.percentile(x, percentile, axis=0).tolist()
    base = (REPO / cfg['policy']['pretrained_path']).resolve()
    tokenizer = base.parent / 'paligemma-3b-pt-224'
    if not (base / 'model.safetensors').is_file(): raise FileNotFoundError(base / 'model.safetensors')
    if not (tokenizer / 'tokenizer_config.json').is_file(): raise FileNotFoundError(tokenizer / 'tokenizer_config.json')
    global_batch = cfg['batch_size'] * 4
    train_starts = sum(max(0, len(eligible[e][0]) - cfg['policy']['chunk_size'] + 1) for e in train)
    if not train_starts: raise ValueError('没有完整动作块可训练')
    cfg['steps'] = steps_override or math.ceil(train_starts / global_batch) * int(rules['training_epochs'])
    if cfg['steps'] < 1: raise ValueError('steps必须为正')
    cfg['scheduler']['num_decay_steps'] = cfg['steps']
    cfg['scheduler']['num_warmup_steps'] = min(cfg['scheduler']['num_warmup_steps'], max(1, cfg['steps'] // 10))
    cfg['dataset'].update(root=str(root), episodes=train)
    cfg['policy']['pretrained_path'] = str(base)
    cfg['output_dir'] = str(output.resolve())
    manifest = {'dataset_root': str(root), 'dataset_sha256': digest.hexdigest(),
                'rules': rules, 'original_episodes': len(grouped), 'excluded': excluded,
                'train_episodes': train, 'validation_episodes': validation,
                'train_frames': train_frames, 'train_chunk_starts': train_starts, 'validation_frames': sum(len(eligible[e][0]) for e in validation),
                'global_batch': global_batch, 'steps': cfg['steps'], 'tokenizer': str(tokenizer),
                'limitations': '自动阈值不等于成功率；近重复、图像遮挡与任务是否成功仍需人工复核。验证集不参与训练和state/action统计；此入口不做真机评估。'}
    dump(runtime / 'train.json', cfg)
    dump(runtime / 'train_stats.json', stats)
    dump(runtime / 'data_selection.json', manifest)
    print(f"原始{len(grouped)}条 → 排除/隔离{len(excluded)}条 → 训练{len(train)}条/{train_frames}帧，验证{len(validation)}条")
    print(f"全局batch={global_batch}，optimizer steps={cfg['steps']}；输出：{cfg['output_dir']}")
    print('排除原因：', excluded)
    return cfg, manifest


def check_resources(output, check_dependencies=True):
    # 每次启动都重新检查；不终止已有任务，也不擅自换用其他卡。
    import subprocess
    errors = []
    existing = output
    while not existing.exists(): existing = existing.parent
    free_gib = shutil.disk_usage(existing).free / 2**30
    if free_gib < 110:
        errors.append(f'输出盘仅剩{free_gib:.1f}GiB，需要至少110GiB，覆盖新旧两个检查点并存峰值；请设置PI05_OUTPUT_DIR到足够大的盘')
    r = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    free = {int(a): float(b)/1024 for a,b in (line.split(',') for line in r.stdout.splitlines())}
    gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,7').split(',')]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        errors.append('要求4个不同的物理GPU编号')
    for i in gpu_ids:
        if free.get(i, 0) < 42: errors.append(f'GPU{i}空闲仅{free.get(i,0):.1f}GiB，先等待当前任务释放；不会自动换卡')
    if check_dependencies:
        from importlib.metadata import version, PackageNotFoundError
        for package, expected in [('lerobot','0.4.3'),('accelerate','1.10.1')]:
            try:
                if version(package) != expected: errors.append(f'要求{package}=={expected}，当前{version(package)}')
            except PackageNotFoundError: errors.append(f'未安装{package}=={expected}；使用独立训练环境和requirements-train-pi05.txt')
    if errors: raise RuntimeError('\n'.join(errors))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--runtime-dir', type=Path, default=REPO/'.train/pi05')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--steps', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    import yaml
    cfg = yaml.safe_load((REPO/'configs/pi05_realman.yaml').read_text())
    rules = yaml.safe_load((REPO/'configs/pi05_data_selection.yaml').read_text())
    if args.steps is not None and args.steps < 1: p.error('--steps必须为正')
    if args.resume:
        checkpoint = args.output_dir/'checkpoints/last'
        if not (checkpoint/'complete.json').is_file(): raise FileNotFoundError('没有完整的FSDP检查点，不能假装resume后重新训练')
        saved = json.loads((checkpoint/'complete.json').read_text())
        # 恢复保存的配置/划分，不用当前yaml静默覆盖原实验。
        for name in ['train.json','train_stats.json','data_selection.json']:
            dump(args.runtime_dir/name, json.loads((checkpoint/name).read_text()))
        if Path(json.loads((args.runtime_dir/'train.json').read_text())['output_dir']).resolve() != args.output_dir.resolve():
            raise ValueError('resume输出路径与检查点不一致')
        selection = json.loads((args.runtime_dir/'data_selection.json').read_text())
        digest = hashlib.sha256()
        root = Path(selection['dataset_root'])
        for path in sorted((root/'data').rglob('*.parquet')):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        if digest.hexdigest() != selection['dataset_sha256']:
            raise ValueError('原始数值数据已变化，禁止按原训练状态恢复')
        print('恢复完整检查点，step=', saved['step'])
    else:
        if args.output_dir.exists(): raise FileExistsError(f'输出目录已存在：{args.output_dir}，请换实验名或显式resume')
        prepare(cfg, rules, args.output_dir, args.runtime_dir, args.steps)
    if not args.prepare_only: check_resources(args.output_dir)
    print('运行配置：', args.runtime_dir/'train.json')

if __name__ == '__main__': main()
