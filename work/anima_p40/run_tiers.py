"""Replay the FP32 baseline PNG's Comfy graph through the two saved P40 modes."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import psutil


def replay_session(bench, jobs, output, mode, original, model='turbo', trace_latents=False):
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', bench.PORT)) == 0:
            raise RuntimeError(f'Port {bench.PORT} already has a listener')

    saved={}

    def fixed_prompt(model, run_name, batch, steps, cfg):
        graph,tag=saved[run_name]
        prompt = deepcopy(graph)
        for node_id, node in prompt.items():
            if node['class_type'] == 'SaveImage':
                node['inputs']['filename_prefix'] = f'anima-{mode}{tag}'
                save_id = node_id
            elif node['class_type'] == 'EmptyLatentImage':
                latent = node['inputs']
            elif node['class_type'] == 'KSampler':
                sample = node['inputs']
        return prompt, save_id, latent, sample

    bench.prompt_for = fixed_prompt
    options = SimpleNamespace(balanced=False, batch=None, steps=None, cfg=None,
                              attention='pytorch', channels_last=False)
    command = [sys.executable, '-u', str(Path(__file__).with_name('launch.py')),
               '--original', str(original), '--mode', mode, '--output', str(output), '--port', str(bench.PORT)]
    if trace_latents:
        command.append('--trace-latents')
    config = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    log_path = output / 'comfy.log'
    if log_path.exists():
        attempt = len(list(output.glob('comfy*.log'))) + 1
        log_path = output / f'comfy-{attempt}.log'
    with log_path.open('wb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=config,
                                   creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        try:
            bench.wait_for_server(process)
            for tag,graph in jobs:
                options.name = mode+tag
                saved[options.name] = (graph,tag)
                report_path = output / 'precision.jsonl'
                prior = report_path.stat().st_size if report_path.exists() else 0
                record = bench.run_prompt(process, log_path, output, options, model)
                if mode != 'fp32':
                    with report_path.open('r', encoding='utf-8') as reports:
                        reports.seek(prior)
                        record['precision'] = json.loads(reports.readline())
                    if record['precision']['mode'] != mode or not record['precision']['int8_linear_calls']:
                        raise RuntimeError('The model did not execute the requested INT8 precision mode')
                record['log'] = str(log_path)
                (output / f'result{tag}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
                print(json.dumps(record), flush=True)
        finally:
            processes = bench.own_processes(process)
            for instance in processes:
                try:
                    instance.terminate()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(processes, timeout=10)


def replay(bench, graph, output, mode, original):
    replay_session(bench,[('',graph)],output,mode,original)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--installation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=('turbo','balanced'), default=('turbo','balanced'))
    args = parser.parse_args()
    installation, output = args.installation.resolve(), args.output.resolve()
    sys.path.insert(0, str(installation))
    import benchmark_p40 as bench

    bench.PORT = 8394
    record = json.loads((output / 'fp32' / 'results.jsonl').read_text(encoding='utf-8').strip())
    with Image.open(record['images'][0]) as img:
        graph = json.loads(img.info['prompt'])
    (output / 'workflow.json').write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding='utf-8')
    for mode in args.modes:
        replay(bench, graph, output/mode, mode, installation/'run_fp32.py')


if __name__ == '__main__':
    main()
