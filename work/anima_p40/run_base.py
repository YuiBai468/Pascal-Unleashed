"""Compare Anima Base step counts with the original Turbo example's seed and prompt."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from PIL import Image

from run_tiers import replay_session


def sampler(graph):
    return next(node['inputs'] for node in graph.values() if node['class_type']=='KSampler')


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--installation', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--reference-png', type=Path, required=True)
    cli.add_argument('--modes', nargs='+', choices=('fp32','turbo','balanced'),default=('fp32','turbo','balanced'))
    cli.add_argument('--steps', nargs='+', type=int, default=(8,40))
    cli.add_argument('--port', type=int, default=8394)
    args = cli.parse_args()
    if len(set(args.steps)) != len(args.steps) or any(step<=0 for step in args.steps):
        cli.error('Step counts must be positive and unique')
    installation, output = args.installation.resolve(), args.output.resolve()
    sys.path.insert(0,str(installation))
    import benchmark_p40 as bench

    bench.PORT = args.port
    graph,_,_,base = bench.prompt_for('base','compare',1,None,None)
    with Image.open(args.reference_png) as png:
        reference = json.loads(png.info['prompt'])
    turbo = sampler(reference)
    base['seed'] = turbo['seed']
    for kind in ('positive','negative'):
        graph[base[kind][0]]['inputs']['text'] = reference[turbo[kind][0]]['inputs']['text']

    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for steps in args.steps:
        prompt = deepcopy(graph)
        sampler(prompt)['steps'] = steps
        (output/f'workflow-{steps}.json').write_text(json.dumps(prompt,ensure_ascii=False,indent=2),encoding='utf-8')
        jobs.append((f'-{steps}',prompt))
    for mode in args.modes:
        replay_session(bench,jobs,output/mode,mode,installation/'run_fp32.py',model='base',trace_latents=True)


if __name__=='__main__':
    main()
