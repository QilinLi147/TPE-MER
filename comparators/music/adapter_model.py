"""Architecture adapters for the music comparison."""
from dataclasses import asdict
import torch
from torch import nn
from .adapters.recent_mer_regression import build_recent_mer_baseline
from .adapters.damer_regression import DAMERJointVARegressor
METHODS=('crnn','mcrnn','damff','dsaml','cnn_transformer','damer')
class AdapterModel(nn.Module):
 FORMAT='tpe_music_controls_20261029'
 def __init__(self,config):
  super().__init__();self.config=config
  if config.kind not in METHODS:raise ValueError(config.kind)
  if config.kind=='damer':self.core=DAMERJointVARegressor(128,84,temporal_bins=87,embed_dim=128,fusion_dim=256,num_heads=4,cross_view_layers=2,dropout=.1)
  else:self.core=build_recent_mer_baseline(config.kind,model_dim=128,dropout=.1)
 @property
 def encoder(self):
  # Return only representation modules: the inherited head-only epoch puts
  # these in eval mode, leaving the selected readout modules in train mode.
  excluded={'head'} if self.config.kind!='damer' else {'fusion_projection','fusion_head','mel_head','cochleagram_head'}
  return nn.ModuleList([m for n,m in self.core.named_children() if n not in excluded])
 def head_parameters(self):
  if self.config.kind=='damer':return list(self.core.fusion_projection.parameters())+list(self.core.fusion_head.parameters())
  return list(self.core.head.shared.parameters())+list(self.core.head.mean.parameters())
 def forward(self,x,times,songs,song_slices):
  if x.ndim!=3 or x.shape[1:]!=(212,self.config.frames):raise ValueError('unexpected frozen cache dimensions')
  lengths=[s.stop-s.start for s in song_slices]
  if sum(lengths)!=len(x) or min(lengths)<1:raise ValueError('invalid song partition')
  B,T=len(lengths),max(lengths);mask=torch.zeros(B,T,dtype=torch.bool,device=x.device)
  features={n:x.new_zeros(B,T,b,self.config.frames) for n,b in [('mel',128),('cochleagram',84)]};order=[]
  for j,sl in enumerate(song_slices):
   idx=torch.argsort(times[sl],stable=True)+sl.start;order.append(idx);n=len(idx);mask[j,:n]=True
   features['mel'][j,:n]=x[idx,:128];features['cochleagram'][j,:n]=x[idx,128:]
  batch={'features':features,'feature_masks':{n:mask[:,:,None].expand(B,T,self.config.frames) for n in features},'mask':mask}
  values=self.core(batch).mean
  flat=torch.cat([values[j,:n] for j,n in enumerate(lengths)],0)
  return flat[torch.argsort(torch.cat(order))]
 def parameter_count(self):
  return {'total':sum(p.numel() for p in self.parameters()),'encoder':sum(p.numel() for p in self.encoder.parameters())}
 def checkpoint(self,extra=None):
  return dict(extra or {},format=self.FORMAT,config=asdict(self.config),state={k:v.detach().cpu() for k,v in self.state_dict().items()})
