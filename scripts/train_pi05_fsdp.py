#!/usr/bin/env python3
"""复用LeRobot 0.4.3的数据/PI05/处理器与优化器；显式处理FSDP全rank保存。

不改site-packages。不是重写π0.5模型。当前仅支持单机4卡、无梯度累积。
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import time


def configure_wrap(plugin, model):
    """PI05联合注意力直接访问decoder子层，不能按GemmaDecoderLayer包裹。

    只包实际经forward调用的Linear；共享权重模块留在根FSDP，防止跨单元拆开。
    """
    import torch
    owners = Counter(id(p) for m in model.modules() for p in m.parameters(recurse=False))
    targets = {m for m in model.modules() if isinstance(m, torch.nn.Linear)
               and all(owners[id(p)] == 1 for p in m.parameters(recurse=False))}
    if not targets: raise RuntimeError('未找到可分片的Linear，拒绝静默退回DDP')
    def wrap(module, recurse, nonwrapped_numel):
        return True if recurse else module in targets
    plugin.auto_wrap_policy = wrap
    return len(targets)


def save_checkpoint(accelerator, model, cfg, pre, post, step, epoch, offset, runtime):
    """所有rank参与save_state和full-state gather，rank0写出可部署BF16完整权重。

    新检查点完整落盘后才替换last并移除上一个本入口生成的检查点。
    """
    import torch
    from safetensors.torch import save_file
    root = Path(cfg.output_dir) / 'checkpoints'
    final = root / f'{step:06d}'
    staging = root / f'.incomplete-{step:06d}'
    if accelerator.is_main_process:
        if staging.exists() or final.exists():
            raise FileExistsError(f'检查点已存在，停止防止覆盖：{staging} / {final}')
        staging.mkdir(parents=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(staging / 'training_state'))
    # FULL_STATE_DICT gather是collective：必须全rank调用，不能包在rank0分支里。
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        target = staging / 'pretrained_model'
        target.mkdir()
        # 打断共享存储并转部署BF16；训练恢复仍用上面的FP32/优化器分片。
        weights = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous().clone()
                   for k, v in state.items()}
        save_file(weights, str(target / 'model.safetensors'), metadata={'format': 'pt'})
        raw = accelerator.unwrap_model(model)
        raw.config.save_pretrained(target)
        cfg.save_pretrained(target)
        pre.save_pretrained(target)
        post.save_pretrained(target)
        for name in ['train.json','train_stats.json','data_selection.json']:
            shutil.copy2(runtime / name, staging / name)
        (staging/'complete.json').write_text(json.dumps({'step': step, 'epoch': epoch,
             'offset': offset, 'world_size': accelerator.num_processes,
             'batch_size': cfg.batch_size, 'format': 'pi05-fsdp-v1'}, indent=2)+'\n')
        staging.rename(final)
        link = root / '.last-next'
        if link.is_symlink(): link.unlink()
        link.symlink_to(final.name)
        os.replace(link, root / 'last')
        for old in root.iterdir():
            if old.is_dir() and not old.is_symlink() and old.name.isdigit() and old != final and (old/'complete.json').is_file():
                shutil.rmtree(old)
        print(f'[checkpoint] {final}', flush=True)
    del state
    accelerator.wait_for_everyone()


def make_processors(cfg, dataset, tokenizer, device):
    from lerobot.policies.factory import make_pre_post_processors
    return make_pre_post_processors(
        policy_cfg=cfg.policy, pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            'device_processor': {'device': str(device)},
            'tokenizer_processor': {'tokenizer_name': tokenizer},
            'normalizer_processor': {'stats': dataset.meta.stats,
                'features': {**cfg.policy.input_features, **cfg.policy.output_features},
                'norm_map': cfg.policy.normalization_mapping}},
        postprocessor_overrides={'unnormalizer_processor': {
            'stats': dataset.meta.stats, 'features': cfg.policy.output_features,
            'norm_map': cfg.policy.normalization_mapping}})


def verify_base_loaded(model, base):
    # from_pretrained的上游异常分支可能返回随机模型。用动作头/视觉塔/语言层探针校验，
    # 只读取小张量，不额外复制14GB模型，也不把加载日志当作成功证据。
    import torch
    from safetensors import safe_open
    state = model.state_dict()
    probes = ['action_in_proj.weight','action_out_proj.weight',
              'paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight',
              'paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight']
    checked = 0
    with safe_open(str(Path(base)/'model.safetensors'),framework='pt',device='cpu') as f:
        for key in probes:
            if key not in f.keys(): continue
            dst = 'model.' + key
            if dst not in state: raise RuntimeError(f'基座映射后的参数不存在：{dst}')
            expected = f.get_tensor(key).to(state[dst].dtype)
            # 只比较少量元素，避免多进程额外跨设备复制大矩阵。
            actual = state[dst].detach().flatten()[:1024].cpu()
            if not torch.equal(actual, expected.flatten()[:1024]):
                raise RuntimeError(f'预训练权重探针不一致，停止：{key}')
            checked += 1
    if checked < 3: raise RuntimeError(f'基座格式未经验证，仅匹配到{checked}个探针')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config_path', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    runtime = args.config_path.resolve().parent
    from importlib.metadata import version
    if version('lerobot') != '0.4.3' or version('accelerate') != '1.10.1':
        raise RuntimeError('使用requirements-train-pi05.txt指定版本，不能静默混用新旧API')
    import torch
    import draccus
    from accelerate import Accelerator, DataLoaderConfiguration
    from accelerate.utils import set_seed
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies.factory import make_policy
    cfg = draccus.decode(TrainPipelineConfig, json.loads(args.config_path.read_text()))
    manifest = json.loads((runtime/'data_selection.json').read_text())
    if cfg.policy.type != 'pi05' or cfg.policy.freeze_vision_encoder or cfg.policy.train_expert_only or cfg.policy.use_peft or cfg.peft is not None:
        raise ValueError('此入口要求π0.5全参数微调，不能冻结或启用PEFT')
    accelerator = Accelerator(mixed_precision='bf16', step_scheduler_with_optimizer=False,
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=cfg.seed))
    if accelerator.distributed_type.name != 'FSDP' or accelerator.num_processes != 4:
        raise RuntimeError('此训练入口必须通过4进程FSDP启动，不接受普通DDP或单卡')
    set_seed(cfg.seed)
    checkpoint = Path(cfg.output_dir)/'checkpoints/last'
    restored = json.loads((checkpoint/'complete.json').read_text()) if args.resume else None
    if restored and (restored['world_size'] != 4 or restored['batch_size'] != cfg.batch_size):
        raise ValueError('恢复要求world_size和每卡batch与保存时一致')
    if not args.resume and Path(cfg.output_dir).exists(): raise FileExistsError(cfg.output_dir)
    with accelerator.main_process_first():
        dataset = make_dataset(cfg)
    dataset.meta.stats = json.loads((runtime/'train_stats.json').read_text())
    # 丢掉每条轨迹最后49个“起始帧”，避免官方PI05 loss把越界补齐动作当真标签；
    # 其余时刻的未来50步仍完整存在，不修改原Parquet。
    episode_ids = dataset.hf_dataset['episode_index']
    frame_ids = dataset.hf_dataset['frame_index']
    lengths = {int(e): int(n) for e,n in zip(dataset.meta.episodes['episode_index'], dataset.meta.episodes['length'])}
    starts = [i for i,(e,f) in enumerate(zip(episode_ids,frame_ids))
              if int(f) + cfg.policy.chunk_size <= lengths[int(e)]]
    if not starts: raise RuntimeError('没有完整动作块可用于训练')
    train_data = torch.utils.data.Subset(dataset, starts)
    model = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    verify_base_loaded(model, cfg.policy.pretrained_path)
    frozen = [name for name,param in model.named_parameters() if not param.requires_grad]
    if frozen: raise RuntimeError(f'仍有冻结参数，拒绝冒充全量微调：{frozen[:10]}')
    # FP32 master参数用于优化器，FSDP前向/反向使用BF16。禁止纯BF16 Adam动量静默降精度。
    model.float()
    wrapped = configure_wrap(accelerator.state.fsdp_plugin, model)
    pre, post = make_processors(cfg, dataset, manifest['tokenizer'], accelerator.device)
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, model)
    loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        **({'prefetch_factor': 2} if cfg.num_workers else {}))
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    step = epoch = offset = 0
    if args.resume:
        accelerator.load_state(str(checkpoint/'training_state'))
        step, epoch, offset = restored['step'], restored['epoch'], restored['offset']
    if accelerator.is_main_process:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        snapshot = Path(cfg.output_dir)/'run_config'
        snapshot.mkdir(exist_ok=True)
        for name in ['train.json','train_stats.json','data_selection.json']:
            shutil.copy2(runtime/name, snapshot/name)
        print(f'全参数微调：FSDP Linear单元={wrapped}，完整chunk起始样本={len(starts)}；全局batch={cfg.batch_size*4}', flush=True)
        print('训练/验证episode互斥；当前入口不运行真机或离线验证成功率。',flush=True)
        if args.resume: print('恢复模型/优化器/调度器/RNG；多worker数据预取下不承诺逐样本完全一致。',flush=True)
    accelerator.wait_for_everyone()
    runtime = Path(cfg.output_dir)/'run_config'
    model.train()
    while step < cfg.steps:
        loader.set_epoch(epoch)
        active = accelerator.skip_first_batches(loader, offset) if offset else loader
        for batch in active:
            started = time.perf_counter()
            batch = pre(batch)
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss, _ = model(batch)
            finite = accelerator.reduce(torch.isfinite(loss.detach()).to(torch.int), reduction='sum')
            if finite.item() != accelerator.num_processes: raise FloatingPointError('检测到非有限loss，所有rank停止')
            accelerator.backward(loss)
            grad = accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_norm)
            if not torch.isfinite(grad): raise FloatingPointError('非有限梯度，停止且不保存为有效检查点')
            optimizer.step()
            scheduler.step()
            step += 1
            offset += 1
            if step % cfg.log_freq == 0 or step == 1:
                mean_loss = accelerator.reduce(loss.detach().float(), reduction='mean')
                if accelerator.is_main_process:
                    with (Path(cfg.output_dir)/'metrics.jsonl').open('a') as log:
                        record = dict(step=step,loss=mean_loss.item(),grad_norm=float(grad),
                            lr=optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-started,
                            gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
                        log.write(json.dumps(record)+'\n')
                    print(record,flush=True)
            if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
                save_checkpoint(accelerator, model, cfg, pre, post, step, epoch, offset, runtime)
            if step >= cfg.steps: break
        epoch += 1
        offset = 0
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == '__main__': main()
