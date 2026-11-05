"""Run TerraTorch with offline local weights and a mandatory encoder-load audit."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import json, sys, time
from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback
from terratorch.cli_tools import build_lightning_cli, MyLightningCLI

torch.set_num_threads(4)
loads = []
original_load = torch.nn.Module.load_state_dict

def checked_load(module, state_dict, *args, **kwargs):
    if type(module).__name__ != 'PrithviViT':
        return original_load(module, state_dict, *args, **kwargs)
    try:
        result = original_load(module, state_dict, *args, **kwargs)
    except RuntimeError as exc:
        raise ValueError('Pretrained Prithvi encoder load failed; refusing random initialization') from exc
    actual = module.state_dict()
    bad = [k for k,v in state_dict.items() if k in actual and not torch.equal(actual[k],v)]
    if result.missing_keys or result.unexpected_keys or bad:
        raise ValueError('Incomplete or inconsistent pretrained Prithvi encoder load')
    loads.append({'keys':len(state_dict),'missing':result.missing_keys,'unexpected':result.unexpected_keys,'mismatched':bad})
    print('VERIFIED_PRETRAINED_ENCODER', json.dumps(loads[-1]), flush=True)
    return result

torch.nn.Module.load_state_dict = checked_load

class RequirePretrained(Callback):
    def setup(self, trainer, pl_module, stage):
        if not loads:
            raise ValueError('No verified pretrained encoder load; refusing to run')

original_instantiate = MyLightningCLI.instantiate_classes

def instantiate_checked(cli):
    original_instantiate(cli)
    cli.trainer.callbacks.append(RequirePretrained())

MyLightningCLI.instantiate_classes = instantiate_checked
start = time.time()
cli = build_lightning_cli()
summary = {'seconds':time.time()-start,'pretrained_loads':loads,'global_step':cli.trainer.global_step,
           'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
           'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,
           'callback_metrics':{k:float(v) for k,v in cli.trainer.callback_metrics.items()},
           'checkpoints':[{'best':getattr(c,'best_model_path',''),'last':getattr(c,'last_model_path','')} for c in cli.trainer.checkpoint_callbacks]}
if cli.trainer.log_dir:
    Path(cli.trainer.log_dir,'run_summary.json').write_text(json.dumps(summary,indent=2))
print('RUN_SUMMARY', json.dumps(summary,indent=2), flush=True)
