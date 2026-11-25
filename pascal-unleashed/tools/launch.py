"""Run the installed ComfyUI with the packaged DLL in its denoiser scope."""

from functools import wraps
import importlib.util
import json
import os
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'work'/'anima_p40'/'launch.py'
spec=importlib.util.spec_from_file_location('recorded_launcher',source)
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def install(mode,out,trace_latents):
    if mode=='fp32':
        return
    from comfy.model_base import BaseModel
    import execution
    from pascal_unleashed.torch import Engine,patch_linear

    modes={'fast':'fast','middle':'balanced','precise':'precise'}
    os.environ['PU_IMMUTABLE_WEIGHTS']='1'
    engine=Engine(modes[mode],cache=True,cast_fp32=True,guard=True)
    apply=BaseModel.apply_model

    @wraps(apply)
    def apply_with_pascal(*args,**kwargs):
        with patch_linear(engine):
            return apply(*args,**kwargs)

    BaseModel.apply_model=apply_with_pascal
    run=execution.PromptExecutor.execute

    @wraps(run)
    def audited(self,prompt,prompt_id,*args,**kwargs):
        engine.clear()
        before=engine.metrics.copy()
        result=run(self,prompt,prompt_id,*args,**kwargs)
        totals={key:engine.metrics[key]-before[key] for key in ('accelerated','native','fallback','logical_ops','milliseconds')}
        report=dict(totals,mode=mode,prompt_id=prompt_id,int8_linear_calls=totals['accelerated'],
                    fp32_linear_calls=totals['native'])
        with (out/'precision.jsonl').open('a',encoding='utf-8') as file:
            file.write(json.dumps(report)+'\n')
        print('Release DLL '+json.dumps(report),flush=True)
        return result

    execution.PromptExecutor.execute=audited


launcher.install=install
launcher.main()
