"""Source bank and block-softmax fitting for the SEED A_scale readout."""
import numpy as np
from scipy.optimize import minimize
from .regions import channel_groups

def stable(x):
    return np.concatenate([x.mean(2,dtype=np.float64),x.std(2,dtype=np.float64)],axis=2).reshape(len(x),-1)

def select_sources(z,y,groups,count):
    chosen=[]
    z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
    centre=z.mean(0);centre/=max(np.linalg.norm(centre),1e-8)
    for i in range(count):
        pool=np.array([j for j in range(len(z)) if j not in chosen],int)
        assert len(pool),'not enough distinct FIT windows'
        if chosen:
            used_subjects={int(groups[j,0]) for j in chosen}
            fresh=np.array([j for j in pool if int(groups[j,0]) not in used_subjects],int)
            if not len(fresh):
                old={tuple(groups[j]) for j in chosen};fresh=np.array([j for j in pool if tuple(groups[j]) not in old],int)
            if len(fresh):pool=fresh
            score=(2-2*z[pool]@z[chosen].T).clip(0,4).min(1)
        else:score=z[pool]@centre
        chosen.append(int(pool[np.argmax(score)]))
    return np.asarray(chosen)

def bank(z,x,y,meta,classes):
    raw=stable(x);mu=raw.mean(0);sd=raw.std(0).clip(1e-3);v=(raw-mu)/sd
    _,regions=channel_groups('seed_raw');masks=[]
    for inds in regions:
        mask=np.zeros((62,10),bool);mask[inds]=True;masks.append(mask.ravel())
    groups=np.stack([meta[k] for k in ['subject','session','trial']],axis=1)
    rows=[];types=[]
    for d in range(classes):
        rr=[];tt=[]
        for cls in [0,1]:
            pool=np.flatnonzero((y==d)==bool(cls))
            for typ,n in [(0,8),(1,2),(2,2),(3,2)]:
                vec=z[pool] if typ==0 else v[pool][:,masks[typ-1]]
                pick=pool[select_sources(vec,y[pool],groups[pool],n)]
                rr.extend(pick.tolist());tt.extend([typ]*n)
        rows.append(rr);types.append(tt)
    rows=np.asarray(rows);types=np.asarray(types)
    return dict(rows=rows,types=types,z=z[rows].astype(np.float64),v=v[rows],mu=mu,sd=sd,masks=np.asarray(masks),source_original_index=meta['original_index'][rows],source_subject=meta['subject'][rows],source_session=meta['session'][rows],source_trial=meta['trial'][rows])

def fit_head(X, y, Xd, yd):
    X = np.asarray(X, np.float64)
    Xd = np.asarray(Xd, np.float64)
    n,D,K=X.shape;mu=X.mean(0);sd=X.std(0).clip(1e-3)
    A=np.concatenate([(X-mu)/sd,np.ones((n,D,1))],2);B=(Xd-mu)/sd
    Y=np.eye(D)[y];best=None;grid=[]
    for C in (0.001, 0.01, 0.1, 1, 10):
        reg=np.full((D,K+1),1/(C*n));reg[:,-1]=1e-10
        def fun(theta):
            w=theta.reshape(D,K+1);z=np.einsum('ndk,dk->nd',A,w);mx=z.max(1,keepdims=True)
            ez=np.exp(z-mx);pp=ez/ez.sum(1,keepdims=True)
            val=np.mean(mx[:,0]+np.log(ez.sum(1))-(Y*z).sum(1))+.5*np.sum(reg*w*w)
            grad=np.einsum('ndk,nd->dk',A,pp-Y)/n+reg*w
            return val,grad.ravel()
        opt=minimize(fun,np.zeros(D*(K+1)),jac=True,method='L-BFGS-B',options=dict(maxiter=2000,gtol=1e-9,ftol=1e-14,maxls=50))
        if not opt.success:raise RuntimeError('convex solver did not converge: '+str(opt.message))
        w=opt.x.reshape(D,K+1);zz=np.einsum('ndk,dk->nd',B,w[:,:-1])+w[:,-1]
        acc=float((np.argmax(zz, axis=1) == yd).mean());grid.append(dict(C=C,accuracy=acc,iterations=int(opt.nit)))
        if best is None or acc>best[0]:best=acc,C,w.copy()
    _,C,w=best;W=w[:,:-1]/sd;b=w[:,-1]-(W*mu).sum(1)
    return dict(W=W,b=b,C=C,grid=grid)

def logits(X,s):
    if X.ndim==2:return X@s['W'].T+s['b']
    return np.einsum('ndk,dk->nd',X,s['W'])+s['b']
