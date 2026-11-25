"""Preserve the original 512-update trajectory, then add one exposure."""
from .v13_protocol import learning_rate
checkpoints=(128,256,512,640,768,896,1024)

def exposure_rate(step):
    if not 1<=step<=1024:raise ValueError('No run may exceed1024updates')
    return learning_rate(min(step,512),512,5e-5,32)

def plateau_diagnostic(target_counts,last_256_losses):
    if len(target_counts)!=3 or len(last_256_losses)!=256:
        raise ValueError('Last3checkpoints and256updates required')
    before=sum(last_256_losses[:128])/128;after=sum(last_256_losses[128:])/128
    change=abs(after-before)/max(abs(before),1e-12)
    span=max(target_counts)-min(target_counts)
    return dict(plateau=span<=2 and change<=.02,target_span_of96=span,
        previous128_mean_ce=before,last128_mean_ce=after,relative_ce_change=change,
        rule='last3 target counts span<=2/96 and last128 vs preceding128 mean training CE changes<=2%',
        interpretation='descriptive finite-window plateau check, not proof of convergence')
