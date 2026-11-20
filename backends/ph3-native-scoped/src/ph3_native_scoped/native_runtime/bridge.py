"""Native packed image+audio responses; every objective/update belongs to current core.

No ph3_hypergan/classic Game import, optimizer-rate mutation, or old state load.
The new conditional critic and batch2 are explicit changes from historical094.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import copy,hashlib,json,math
import torch
from torch import nn
from particlegan import get_recipe
from particle_sliders.engine import ConditionalParticleGAN,PG_VERSION,PG_SOURCE_PROFILE
from particle_sliders.engine import _hash
from particle_sliders.backends import AdapterParameters,PreparedResponse,bind_native_gradient_checkpointing
from .native_host import PackedState,Conditioning
from .networks import RoutedAdapter,GlobalMixCritic
from . import COHORT

@dataclass(frozen=True)
class NativeRow:
 state:PackedState
 neutral:Conditioning
 positive:Conditioning
 scene:str
 seed:int
 path:str
 position:int
 time:float
 scale:float=1.

class NativeResponse:
 family='ph3-native-image-coupled-audio'
 def __init__(self,host,adapter,*,identity):
  if host.adapter is not adapter or not identity:raise ValueError('actual owned native host/adapter/identity required')
  if adapter.particles.shape!=(128,4)or adapter.particles.dtype!=torch.float32:raise ValueError('actual complete128x4 FP32bank required')
  self.host,self.adapter,self.model_identity=host,adapter,copy.deepcopy(identity)
  self.network=AdapterParameters(adapter.named_parameters(),adapter.scaled)
  bind_native_gradient_checkpointing(host.model,self.network)
 def prepare(self,row):
  if type(row.position)is not int or not 0<=row.position<len(self.host.timesteps)or row.path not in('neutral','positive')or row.scale!=1. or row.time!=float(self.host.timesteps[row.position]):raise ValueError('declared native path/time/unipolar target required')
  state=row.state.copy();neutral=row.neutral.copy();positive=row.positive.copy()
  row=NativeRow(state,neutral,positive,row.scene,row.seed,row.path,row.position,row.time)
  with torch.no_grad(),self.network.scaled(0.):
   n=self.host.response(self.host.velocity(row.state,row.neutral,row.position)).detach().clone()
   t=self.host.response(self.host.velocity(row.state,row.positive,row.position)).detach().clone()
  if n.shape!=t.shape or n.ndim!=2 or n.shape[0]!=1 or not all(torch.isfinite(x).all()for x in(n,t)):raise ValueError('full finite native response geometry required')
  return PreparedResponse(row,t,n)
 def generated(self,prepared):
  with self.network.scaled(1.):
   value=self.host.response(self.host.velocity(prepared.row.state,prepared.row.neutral,prepared.row.position))
  if value.shape!=prepared.target.shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():raise ValueError('full native FP32 response required')
  return value
 def response(self,row,*,strength):
  if type(strength)not in(int,float)or not math.isfinite(strength)or not 0<=strength<=1:raise ValueError('finite unipolar strength required')
  with self.network.scaled(strength):return self.host.response(self.host.velocity(row.state,row.neutral,row.position))
 def metadata(self):
  return dict(cohort=COHORT,model=self.model_identity,response='native image then coupled-audio flattened without cropping/pooling',time='actual pinned native scheduler position/timestep',routed_architecture=self.adapter.identity,routed_source_sha256=hashlib.sha256(Path(__file__).with_name('networks.py').read_bytes()).hexdigest(),host_source_sha256=hashlib.sha256(Path(__file__).with_name('native_host.py').read_bytes()).hexdigest(),nativebank=[128,4])

class ConditionalCritic(nn.Module):
 """All native response coordinates followed by declared metadata conditions.

 The historical bounded critic is not used: shared095 requires unbounded raw
 logits. The original HNDL global-mix/manual-attention math stays unchanged.
 """
 score_bound=0.
 def __init__(self,response_width,condition_width,*,device='cpu'):
  super().__init__();self.response_width=response_width;self.condition_width=condition_width
  self.critic=GlobalMixCritic(response_width+condition_width,score_bound=0.,device=device)
  self.identity=dict(formulation='ph3095-unbounded-conditional-globalmix-v1',response_width=response_width,condition_width=condition_width,condition_features=['position_fraction','neutral_path','positive_path'],full_response_coordinates=True,architecture=self.critic.identity)
 def forward(self,response,condition):
  if response.ndim!=2 or response.shape[1]!=self.response_width or condition.shape!=(len(response),self.condition_width):raise ValueError('exact complete native response/condition dimensions required')
  return self.critic(torch.cat((response,condition),dim=1))

def coordinate_scale(prepared):
 """One training-only task coordinate scalar, never a recipe/noise override."""
 energy=sum(float((p.target.double()-p.neutral.double()).square().sum())for p in prepared)
 coordinates=sum(p.target.numel()for p in prepared)
 if not coordinates or not math.isfinite(energy):raise ValueError('finite full training coordinates required')
 return math.sqrt(energy/coordinates)if energy>0 else 1.

def conditions(prepared,positions,device):
 return torch.tensor([[p.row.position/max(1,positions-1),float(p.row.path=='neutral'),float(p.row.path=='positive')]for p in prepared],device=device,dtype=torch.float32)

def build_game(host,rows,*,lr,d_lr_mult,prior_lr_mult,seed=7,total_steps=128,identity):
 """Construct the actual shared095 game, with public recipe noise unchanged.

 Callers construct host/adapter/init in an owned scoped RNG context. No model
 weights, previous teachers, optimizer packets or private trainer are loaded.
 """
 if total_steps!=128 or PG_VERSION!='0.11.0'or PG_SOURCE_PROFILE!='release-0.11.0':raise ValueError('fresh official095 common128 cohort required')
 backend=NativeResponse(host,host.adapter,identity=identity);prepared=[backend.prepare(row)for row in rows]
 if len(prepared)<2:raise ValueError('batch2 requires at leasttwo declared rows')
 condition=conditions(prepared,len(host.timesteps),host.device);width=prepared[0].target.shape[1]
 critic=ConditionalCritic(width,condition.shape[1],device=host.device)
 reference=get_recipe('bispf_optimized')
 game=ConditionalParticleGAN(backend,prepared,critic,conditions=condition,particles=backend.network.particles,response_domain='continuous_vector',response_scale=coordinate_scale(prepared),sigma=reference.observation_sigma,batch_size=2,seed=seed,total_steps=128,recipe_overrides=dict(lr=lr,d_lr_mult=d_lr_mult,prior_lr_mult=prior_lr_mult),pg_source_profile='release-0.11.0',penalty_input_domain='continuous')
 if game.prior.z is not host.adapter.particles or game.prior.z.numel()!=512:raise RuntimeError('actual complete routed prior bank not owned by sharedgame')
 if game.recipe.observation_sigma!=reference.observation_sigma or game.sigma!=reference.observation_sigma:raise RuntimeError('public recipe/reference noise differs')
 return game
