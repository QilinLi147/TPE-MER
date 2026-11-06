"""Electrode groups for the SEED adaptation."""
DEAP_CHANNELS="FP1 AF3 F3 F7 FC5 FC1 C3 T7 CP5 CP1 P3 P7 PO3 O1 OZ PZ FP2 AF4 FZ F4 F8 FC6 FC2 CZ C4 T8 CP6 CP2 P4 P8 PO4 O2".split()
SEED_CHANNELS="FP1 FPZ FP2 AF3 AF4 F7 F5 F3 F1 FZ F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCZ FC2 FC4 FC6 FT8 T7 C5 C3 C1 CZ C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 P7 P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO5 PO3 POZ PO4 PO6 PO8 CB1 O1 OZ O2 CB2".split()
TYPE_NAMES=("learned_global","frontal_band_statistics","central_temporal_band_statistics","posterior_band_statistics")

def channel_groups(dataset):
    names=DEAP_CHANNELS if dataset=="deap" else SEED_CHANNELS if dataset=="seed_raw" else None
    if names is None: raise ValueError("unregistered EEG dataset")
    groups=[[],[],[]]
    for i,n in enumerate(names):
        if n.startswith(("CP","TP","P","O","CB")):k=2
        elif n.startswith(("FP","AF","F")):k=0
        else:k=1
        groups[k].append(i)
    assert sorted(sum(groups,[]))==list(range(len(names))) and all(groups)
    return names,groups

