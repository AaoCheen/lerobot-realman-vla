"""验证不删数据的划分/归一化，以及重复轨迹不跨训练验证；不加载VLA模型。"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_pi05', REPO/'scripts/prepare_pi05.py')
prepare_module=importlib.util.module_from_spec(spec); spec.loader.exec_module(prepare_module)

class PreparationTest(unittest.TestCase):
    def test_resource_gate_uses_selected_physical_gpus(self):
        free = SimpleNamespace(free=120 * 2**30)
        # 0号卡有任务不应阻止后4卡；4号卡不足时必须拒绝启动。
        rows = '0, 1000\n1, 49000\n2, 49000\n3, 49000\n4, 49000\n5, 49000\n6, 49000\n7, 49000\n'
        with mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '4,5,6,7'}), \
             mock.patch.object(prepare_module.shutil, 'disk_usage', return_value=free), \
             mock.patch('subprocess.run', return_value=SimpleNamespace(stdout=rows)):
            prepare_module.check_resources(Path('/tmp/new-vla-test'), check_dependencies=False)
        rows = rows.replace('4, 49000', '4, 10000')
        with mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '4,5,6,7'}), \
             mock.patch.object(prepare_module.shutil, 'disk_usage', return_value=free), \
             mock.patch('subprocess.run', return_value=SimpleNamespace(stdout=rows)):
            with self.assertRaisesRegex(RuntimeError, 'GPU4'):
                prepare_module.check_resources(Path('/tmp/new-vla-test'), check_dependencies=False)

    def test_filter_split_and_training_only_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data=root/'data'; (data/'meta').mkdir(parents=True); (data/'data').mkdir()
            base=root/'models/pi05_base'; base.mkdir(parents=True); (base/'model.safetensors').write_bytes(b'fixture-only')
            tokenizer=base.parent/'paligemma-3b-pt-224'; tokenizer.mkdir(); (tokenizer/'tokenizer_config.json').write_text('{}')
            rows=[]
            for ep in range(10):
                for frame in range(70):
                    # 0/1是数值完全相同的源轨迹，必须被分在同一侧。
                    source=0 if ep==1 else ep
                    state=[(source+1)*10+frame*.1]*6+[.1 if frame%2 else .9]
                    if ep==8: state[-1]=1.0  # 全程不闭合，自动排除。
                    if ep==9: state[0]=1e6  # 人工排除，不得污染训练统计。
                    rows.append(dict(episode_index=ep,frame_index=frame,**{'observation.state':state,'action':state}))
            pq.write_table(pa.Table.from_pylist(rows),data/'data/rows.parquet')
            original=(data/'data/rows.parquet').read_bytes()
            info={'robot_type':'realman','total_episodes':10,'total_frames':700,
                  'features':{'observation.state':{'shape':[7]},'action':{'shape':[7]}}}
            (data/'meta/info.json').write_text(json.dumps(info)); (data/'meta/stats.json').write_text('{}')
            cfg=yaml.safe_load((REPO/'configs/pi05_realman.yaml').read_text())
            cfg['dataset']['root']=str(data); cfg['policy']['pretrained_path']=str(base)
            rules=yaml.safe_load((REPO/'configs/pi05_data_selection.yaml').read_text())
            rules.update(exclude_episodes=[9],review_candidates=[7],validation_fraction=.25)
            out=root/'run'; runtime=root/'runtime'
            _, m=prepare_module.prepare(cfg,rules,out,runtime)
            self.assertEqual(set(m['excluded']),{7,8,9})
            self.assertFalse(set(m['train_episodes']) & set(m['validation_episodes']))
            self.assertEqual(0 in m['train_episodes'],1 in m['train_episodes'])
            stats=json.loads((runtime/'train_stats.json').read_text())
            expected=np.asarray([r['observation.state'] for r in rows if r['episode_index'] in m['train_episodes']],np.float32).astype(np.float64)
            np.testing.assert_allclose(stats['observation.state']['mean'],expected.mean(0))
            self.assertLess(max(stats['observation.state']['max']),1e6)
            self.assertEqual(m['train_chunk_starts'],len(m['train_episodes'])*21)
            self.assertEqual((data/'data/rows.parquet').read_bytes(),original)
            first=deepcopy(m)
            _, second=prepare_module.prepare(cfg,rules,out,runtime)
            self.assertEqual(first,second)
            rules['exclude_episodes']=[999]
            with self.assertRaisesRegex(ValueError,'不存在'):
                prepare_module.prepare(cfg,rules,out,runtime)

if __name__=='__main__':unittest.main()
