#!/usr/bin/env python3
"""只读核对 realman_all 的状态/动作前缀来源；不据此断言图像相同。"""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
BASE=Path(__file__).resolve().parents[2]/'data/datasets'

def episodes(name):
    groups={}
    for path in sorted((BASE/name/'data').rglob('*.parquet')):
        table=pq.read_table(path).to_pydict()
        for i,e in enumerate(table['episode_index']):
            groups.setdefault(e,[]).append((table['frame_index'][i],table['observation.state'][i][:7],table['action'][i]))
    out={}
    for e,rr in groups.items():
        rr.sort(key=lambda r:r[0])
        out[e]=(np.asarray([r[1] for r in rr],np.float32),np.asarray([r[2] for r in rr],np.float32))
    return out

def main():
    target=episodes('realman_all')
    sources=[(n,episodes(n)) for n in ['realman_new','realman_new_after48_full']]
    matches=[]
    for e,(state,action) in target.items():
        for name,data in sources:
            for source_e,(s,a) in data.items():
                if len(s)>=len(state) and np.array_equal(state,s[:len(state)]) and np.array_equal(action,a[:len(action)]):
                    matches.append(dict(episode=e,source=name,source_episode=source_e,frames=len(state),source_frames=len(s)))
    result=dict(criterion='exact state first7 and action prefix match; image lineage not verified',matches=matches,
        unmatched_target=sorted(set(target)-{m['episode'] for m in matches}),
        unmatched_sources={name:sorted(set(data)-{m['source_episode'] for m in matches if m['source']==name}) for name,data in sources})
    Path(__file__).with_name('lineage_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('匹配条数:',len(matches)); print('来源未匹配:',result['unmatched_sources'])
    return result

if __name__=='__main__':main()
