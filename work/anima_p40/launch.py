"""Reuse the installed FP32 Comfy launcher and add a process-local Anima hook."""

import argparse
import ast
from ctypes import c_void_p
from functools import wraps
from inspect import signature
import json
import os
from pathlib import Path
import sys


def install_latent_trace(out):
    import comfy.sample
    import numpy as np

    sample = comfy.sample.sample
    parameters = signature(sample)

    @wraps(sample)
    def trace(*args, **kwargs):
        bound = parameters.bind(*args, **kwargs)
        steps = bound.arguments['steps']
        seed = bound.arguments.get('seed')
        original = bound.arguments.get('callback')
        checkpoints = {1, 2, 4, 8, 12, 16, 20, 24, 30, 32, steps}
        snapshots = {}

        def record(step, denoised, current, total):
            if step + 1 in checkpoints:
                snapshots[f'denoised_{step+1:03d}'] = denoised.detach().clone()
            if original is not None:
                return original(step, denoised, current, total)

        bound.arguments['callback'] = record
        result = sample(*bound.args, **bound.kwargs)
        snapshots['final'] = result.detach()
        np.savez(out / f'latents-{steps}-{seed}.npz', **{key: tensor.cpu().numpy() for key,tensor in snapshots.items()})
        return result

    comfy.sample.sample = trace


def install(mode, out, trace_latents):
    if trace_latents:
        install_latent_trace(out)
    if mode == 'fp32':
        return
    import execution
    import torch
    import torch.nn.functional as functional
    if mode in ('fast','middle','precise'):
        from comfy.model_base import BaseModel
        from four_engine import FourEngine

        engine=FourEngine(os.environ['CUDA_VISIBLE_DEVICES'],mode)
        engine.self_check()
        model_class,method=BaseModel,'apply_model'
    else:
        from comfy.ldm.anima.model import Anima
        from pascal_engine import Engine

        engine=Engine(os.environ['CUDA_VISIBLE_DEVICES'])
        engine.self_check(c_void_p(torch.cuda.current_stream(0).cuda_stream))
        engine.mode=mode
        model_class,method=Anima,'forward'
    model_forward=getattr(model_class,method)
    linear = functional.linear

    @wraps(model_forward)
    def with_pascal(*args, **kwargs):
        def patched_linear(input, weight, bias=None):
            return engine.linear(linear, input, weight, bias)

        functional.linear = patched_linear
        try:
            return model_forward(*args, **kwargs)
        finally:
            functional.linear = linear

    setattr(model_class,method,with_pascal)
    execute = execution.PromptExecutor.execute

    @wraps(execute)
    def audited(self, prompt, prompt_id, *args, **kwargs):
        engine.reset()
        result = execute(self, prompt, prompt_id, *args, **kwargs)
        report = engine.report()
        report['prompt_id'] = prompt_id
        if not report['int8_linear_calls']:
            raise RuntimeError('The workflow never invoked a Pascal INT8 linear operation')
        with (out / 'precision.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(report) + '\n')
        print(f"Pascal {mode}: {report['int8_linear_calls']} DP4A GEMMs, "
              f"{report['fp32_linear_calls']} small or unaligned FP32 linears", flush=True)
        return result

    execution.PromptExecutor.execute = audited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--mode', choices=('fp32','turbo','balanced','fast','middle','precise'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--trace-latents', action='store_true')
    opts = parser.parse_args()
    original = opts.original.resolve()
    out = opts.output.resolve()
    tree = ast.parse(original.read_text(encoding='utf-8-sig'), filename=str(original))
    for index, node in enumerate(tree.body):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'start_comfyui'):
            tree.body[index:index] = ast.parse('_install_pascal(_pascal_mode, _pascal_output, _pascal_trace_latents)').body
            break
    else:
        raise RuntimeError('The installed launcher has no ComfyUI entry point')

    sys.argv = [str(original), '--port', str(opts.port), '--benchmark-output', str(out)]
    namespace = {'__file__': str(original), '__name__': '__main__', '_install_pascal': install,
                 '_pascal_mode': opts.mode, '_pascal_output': out, '_pascal_trace_latents': opts.trace_latents}
    exec(compile(ast.fix_missing_locations(tree), str(original), 'exec'), namespace)


if __name__ == '__main__':
    main()
