"""Reader-only answer, paired-assignment and unaffected-fact supervision."""

def assignment(scores):
    import torch
    if scores.shape[-2:]!=(2,2):raise ValueError('Two conditions by two candidate values required')
    a,b,c,d=scores[...,0,0],scores[...,0,1],scores[...,1,0],scores[...,1,1]
    gaps=torch.stack((a-b,d-c,a-c,d-b),dim=-1)
    lower=-.5*torch.logsumexp(-gaps/.5,dim=-1)
    return torch.nn.functional.softplus(.5-lower).mean()

def locality(student,teacher):
    import torch
    if student.shape!=teacher.shape:raise ValueError('Identical answer token alignment required')
    return torch.nn.functional.kl_div(student.log_softmax(-1),teacher.detach().softmax(-1),reduction='batchmean')

def prefix(model,item,i,j,cache,arm):
    if arm not in ('compressed','full'):raise ValueError('Known reading arm required')
    ids=item['contexts'][i];query=item['queries'][j]
    memory=cache.get(ids).to(model.device) if arm=='compressed' else None
    return model.prefix(dict(context=ids,**query),'memory_reader' if arm=='compressed' else 'full_reader',memory)

def logits(model,p,answer):
    import torch
    inputs=torch.cat((p,model.embeddings(answer).unsqueeze(0)),dim=1)
    result=model.forward_prefix(inputs,use_cache=False,return_dict=True)
    return result.logits[0,p.shape[1]-1:p.shape[1]-1+len(answer)].float()

def backward(model,item,cache,arm,method,weight=.5):
    import torch
    if method!='bottleneck' or not 0<weight<=1:raise ValueError('Fixed objective family required')
    if any(part.training for part in model.modules()):raise ValueError('Deterministic eval-mode replay required')
    pairs=[(i,j,a) for i,row in enumerate(item['answers']) for j,a in enumerate(row) if a is not None]
    prefixes={(i,j):prefix(model,item,i,j,cache,arm) for i,j,_ in pairs}
    other=(2,3) if item['family']=='probe' else (1,)
    teacher={};ce_value=0.;locality_value=0.
    for i,j,answer in pairs:
        if method=='bottleneck' and j in other:
            output=logits(model,prefixes[i,j],answer)
            ce=torch.nn.functional.cross_entropy(output,torch.tensor(answer,device=model.device))
            loss=ce*weight/len(pairs)
            if i==0:teacher[j]=output.detach()
            else:
                local=locality(output,teacher[j])/(3*len(other))
                loss=loss+.05*weight*local;locality_value+=float(local.detach())
        else:
            ce=model.answer_nll(prefixes[i,j],answer);loss=ce*weight/len(pairs)
        ce_value+=float(ce.detach())/len(pairs);loss.backward()
        del loss,ce
    assignment_value=0.
    if item['family']=='probe':
        from .benchmark import _sequential_auxiliary
        branches=[lambda i=i,j=j,a=a:-model.answer_nll(prefixes[i,j],a)
            for i in range(4) for j in (0,1) for a in item['answers'][i][:2]]
        objective=lambda *scores:assignment(torch.stack(scores).reshape(4,2,2))
        assignment_value=_sequential_auxiliary(branches,objective,model.device,.1)
    return dict(loss=weight*ce_value+.1*assignment_value+.05*weight*locality_value,
        sft_loss=ce_value,bottleneck_loss=assignment_value,locality_loss=locality_value,
        supervised_answers=len(pairs),reader_only=True)
