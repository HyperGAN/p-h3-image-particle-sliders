"""Sole RAW metric over full native image+audio coordinates, teacher-only set."""
import math
import torch
def score(records,*,positions):
 groups={};seen=set()
 for r in records:
  key=(r['scene'],r['seed'],r['path']);position=r['position']
  if(key,position)in seen:raise ValueError('duplicate native example/position')
  seen.add((key,position));s,n,t=[r[k].detach().cpu().double()for k in('student','neutral','teacher')]
  if s.shape!=n.shape or s.shape!=t.shape or not s.numel()or not all(torch.isfinite(v).all()for v in(s,n,t)):raise ValueError('all student/neutral/teacher coordinates must remain finite, including excluded teacherzero rows')
  delta=t-n;energy=float(delta.square().sum());sse=float((s-t).square().sum())
  groups.setdefault(key,[]).append(dict(position=position,SSE=sse,teacher_energy=energy,oracle_SSE={k:float((v-t).square().sum())for k,v in [('OFF',n),('teacher',t),('reverse',n-delta)]}))
 examples=[];valid=[];excluded=[]
 for key,rows in sorted(groups.items()):
  if sorted(r['position']for r in rows)!=list(positions):raise ValueError('incomplete fixed native position set')
  energy=math.fsum(r['teacher_energy']for r in rows);sse=math.fsum(r['SSE']for r in rows)
  if not math.isfinite(energy)or not math.isfinite(sse):raise ValueError('nonfinite pooled coordinates')
  relative=sse/energy if energy>0 else None
  (valid if energy>0 else excluded).append(list(key))
  examples.append(dict(example=list(key),SSE=sse,teacher_energy=energy,relative_SSE=relative,oracles={k:math.fsum(r['oracle_SSE'][k]for r in rows)/energy if energy>0 else None for k in('OFF','teacher','reverse')}))
 values=[x['relative_SSE']for x in examples if x['relative_SSE']is not None]
 return dict(primary=math.fsum(values)/len(values)if values else None,teacher_valid_examples=valid,teacher_excluded_zero_energy_examples=excluded,examples=examples,raw_coordinates='complete image followed by coupled audio; no scaling in metric',definition='Operand-first FP64, pool positions/coordinates within example, then equal example mean; fixed teacher-only exclusions, no epsilon or student-driven exclusion')
