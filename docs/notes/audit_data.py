#!/usr/bin/env python3
"""只读检查数据；仅在 docs/notes 写 JSON/CSV，不修改或删除 data 中的文件。"""
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import csv, hashlib, json, subprocess
import numpy as np
import pyarrow.parquet as pq
import h5py

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'data'
OUT = Path(__file__).resolve().parent

def read_rows(paths):
    return [row for p in sorted(paths) for row in pq.read_table(p).to_pylist()]

def probe(p):
    r = subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_frames,duration,avg_frame_rate',
        '-of','json',str(p)],capture_output=True,text=True)
    try:
        s=json.loads(r.stdout)['streams'][0]
        return str(p), s
    except (ValueError,KeyError,IndexError):
        return str(p), {'error':r.stderr[:300] or 'no video stream'}

def main():
    summaries=[]; episodes=[]; signatures=defaultdict(list); raw=[]
    for ip in sorted(DATA.rglob('meta/info.json')):
        root=ip.parent.parent; info=json.loads(ip.read_text()); name=root.name
        meta=read_rows((root/'meta/episodes').rglob('*.parquet'))
        rows=read_rows((root/'data').rglob('*.parquet'))
        groups=defaultdict(list)
        for row in rows: groups[int(row['episode_index'])].append(row)
        metaby={int(m['episode_index']):m for m in meta}
        errors=[]; warnings=[]
        if len(rows)!=info['total_frames']: errors.append('info frame count mismatch')
        if len(groups)!=info['total_episodes'] or len(meta)!=info['total_episodes']: errors.append('episode count mismatch')
        if set(groups)!=set(metaby): errors.append('episode IDs mismatch')
        indexes=[int(r['index']) for r in rows]
        if sorted(indexes)!=list(range(len(rows))): errors.append('global index not unique/contiguous')
        video_keys=[k for k,v in info['features'].items() if v['dtype']=='video']
        probes=dict(ThreadPoolExecutor(max_workers=8).map(probe,sorted((root/'videos').rglob('*.mp4'))))
        for e, rr in sorted(groups.items()):
            rr.sort(key=lambda r:r['frame_index']); n=len(rr); m=metaby.get(e,{})
            state=np.asarray([r['observation.state'] for r in rr],dtype=np.float32)
            action=np.asarray([r['action'] for r in rr],dtype=np.float32)
            ts=np.asarray([r['timestamp'] for r in rr],dtype=float)
            issues=[]; review=[]
            if not np.isfinite(state).all() or not np.isfinite(action).all(): issues.append('nonfinite_state_action')
            if state.shape!=(n,info['features']['observation.state']['shape'][0]) or action.shape!=(n,info['features']['action']['shape'][0]): issues.append('shape_mismatch')
            if [int(r['frame_index']) for r in rr]!=list(range(n)): issues.append('frame_index_not_contiguous')
            if n>1 and (np.any(np.diff(ts)<=0) or not np.allclose(np.diff(ts),1/info['fps'],atol=2e-5)): issues.append('timestamp_not_uniform_fps')
            if m.get('length')!=n: issues.append('meta_length_mismatch')
            if m and (int(m['dataset_from_index'])!=min(r['index'] for r in rr) or int(m['dataset_to_index'])!=max(r['index'] for r in rr)+1): issues.append('meta_index_range_mismatch')
            if m and len(m.get('stats/observation.state/min',[])) != state.shape[1]: review.append('episode_state_stats_dimension_stale')
            if m and m.get('stats/observation.state/count',[n])[0]!=n: review.append('episode_state_stats_count_stale')
            grip=action[:,-1]; sg=state[:,6] if info['robot_type']=='realman' else state[:,-1]
            if info['robot_type']=='realman':
                if np.any(grip<-.001) or np.any(grip>1.001) or np.any(sg<-.001) or np.any(sg>1.001): issues.append('gripper_outside_0_1')
                if np.all(grip>=.5): review.append('never_close_below_0.5')
                if grip[-1]<.5: review.append('end_closed_below_0.5')
                if np.any(np.all(np.abs(state[:,:6])<1e-7,axis=1)): review.append('all_joint_zero_state')
                if np.any(np.all(np.abs(action[:,:6])<1e-7,axis=1)): review.append('all_joint_zero_action')
            if n<60: review.append('short_under_2s')
            max_step=float(np.max(np.abs(np.diff(action[:,:6],axis=0)))) if n>1 else 0
            if info['robot_type']=='realman' and max_step>20: review.append('raw_joint_step_over_20deg_check_wrap')
            state_delta=float(np.max(np.ptp(state[:,:6],axis=0)))
            if state_delta<1: review.append('joint_range_under_1unit')
            next_diff=float(np.max(np.abs(action[:-1,:min(action.shape[1],state.shape[1])]-state[1:,:action.shape[1]]))) if n>1 else 0
            videos=[]
            for key in video_keys:
                prefix='videos/'+key
                if prefix+'/file_index' not in m: issues.append('missing_video_meta'); continue
                vp=root/info['video_path'].format(video_key=key,chunk_index=int(m[prefix+'/chunk_index']),file_index=int(m[prefix+'/file_index']))
                videos.append(str(vp.relative_to(REPO)))
                if not vp.exists(): issues.append('missing_video'); continue
                s=probes.get(str(vp),{})
                if 'error' in s or not s: issues.append('video_probe_failed'); continue
                shape=info['features'][key]['shape']
                if (s.get('height'),s.get('width'))!=tuple(shape[:2]): issues.append('video_shape_mismatch')
                available=int(s['nb_frames']) if str(s.get('nb_frames','')).isdigit() else None
                begin=float(m[prefix+'/from_timestamp']); end=float(m[prefix+'/to_timestamp'])
                if begin<0 or abs((end-begin)*info['fps']-n)>1.1: issues.append('video_meta_span_mismatch')
                if available is not None and end*info['fps']>available+1.1: issues.append('video_too_short_for_meta')
            sig=hashlib.sha256(np.ascontiguousarray(state[:,:action.shape[1]]).tobytes()+np.ascontiguousarray(action).tobytes()).hexdigest()
            signatures[sig].append({'dataset':name,'episode':e,'frames':n})
            rec=dict(dataset=name,episode=e,frames=n,file_index=m.get('data/file_index'),gripper_min=float(grip.min()),gripper_max=float(grip.max()),closed_ratio=float(np.mean(grip<.5)),max_joint_action_step=max_step,next_state_max_abs_diff=next_diff,issues=';'.join(sorted(set(issues))),review=';'.join(sorted(set(review))),videos=';'.join(videos))
            episodes.append(rec)
        tasks=read_rows([root/'meta/tasks.parquet'])
        summary=dict(dataset=name,robot=info['robot_type'],episodes=len(groups),frames=len(rows),metadata_episodes=len(meta),fps=info['fps'],state_dim=info['features']['observation.state']['shape'][0],action_dim=info['features']['action']['shape'][0],tasks=tasks,errors=errors,video_files=len(probes),probe_errors=sum('error' in p for p in probes.values()),episode_issue_counts=dict(Counter(x for r in episodes if r['dataset']==name for x in r['issues'].split(';') if x)),review_counts=dict(Counter(x for r in episodes if r['dataset']==name for x in r['review'].split(';') if x)))
        summaries.append(summary); print(json.dumps(summary,ensure_ascii=False),flush=True)
    for p in sorted((DATA/'raw_hdf5').rglob('*.hdf5')):
        try:
            with h5py.File(p,'r') as f:
                q=np.asarray(f['observations/qpos']); a=np.asarray(f['action']); t=np.asarray(f['timestamps'])
                keys=[]; f.visititems(lambda k,v:keys.append((k,list(v.shape))) if isinstance(v,h5py.Dataset) else None)
                raw.append(dict(file=str(p.relative_to(REPO)),frames=len(q),shapes=keys,action_next_state_maxdiff=float(np.max(np.abs(a[:-1]-q[1:]))),actual_fps=float(1/np.mean(np.diff(t))),max_gap_seconds=float(np.max(np.diff(t))),nonmonotonic=int(np.sum(np.diff(t)<=0)),nonfinite=int(np.sum(~np.isfinite(q))),attrs={k:str(v) for k,v in f.attrs.items()}))
        except Exception as e: raw.append(dict(file=str(p.relative_to(REPO)),error=str(e)))
    dup=[v for v in signatures.values() if len(v)>1]
    result=dict(datasets=summaries,raw_hdf5=raw,exact_state_action_duplicate_groups=dup,limitations=['Video headers and metadata checked, not full video decode or task success.','Duplicate definition is identical complete state (first action_dim columns) + action arrays, not image identity.','Review thresholds are heuristics, not deletion decisions.'])
    (OUT/'data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'episode_qc.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=episodes[0].keys(),lineterminator='\n'); w.writeheader(); w.writerows(episodes)
    print('raw',json.dumps(raw,ensure_ascii=False)); print('duplicate_groups',len(dup))
    return result

if __name__=='__main__':
    main()
