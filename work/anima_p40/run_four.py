"""Replay the four installed workflows with fixed prompts and sampling settings."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import psutil


MODES = ('fp32', 'fast', 'middle', 'precise')


def run_mode(bench, graphs, mode, target, installation, models):
    target.mkdir(parents=True, exist_ok=True)
    models = [key for key in models if not (target / f'{key}.json').is_file()]
    if not models:
        return
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', bench.PORT)) == 0:
            raise RuntimeError(f'Port {bench.PORT} has an existing listener')
    log_path=target/'comfy.log'
    if log_path.exists():
        log_path=target/f'comfy-{len(list(target.glob("comfy*.log")))+1}.log'
    command = [sys.executable,'-u',str(Path(__file__).with_name('launch.py')),
               '--original',str(installation/'run_fp32.py'),'--mode',mode,'--output',str(target),
               '--port',str(bench.PORT)]
    environment=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    options=SimpleNamespace(name=mode,balanced=False,batch=None,steps=None,cfg=None,
                            attention='pytorch',channels_last=False)
    bench.prompt_for=lambda model,*args: graphs[model]
    with log_path.open('wb') as log:
        process=subprocess.Popen(command,cwd=installation,stdout=log,stderr=subprocess.STDOUT,env=environment,
                                 creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        try:
            bench.wait_for_server(process)
            for model in models:
                reports=target/'precision.jsonl'
                offset=reports.stat().st_size if reports.exists() else 0
                result=bench.run_prompt(process,log_path,target,options,model)
                if mode!='fp32':
                    with reports.open('rb') as file:
                        file.seek(offset)
                        result['precision']=json.loads(file.readline())
                    precision=result['precision']
                    if precision['mode']!=mode or precision['int8_linear_calls']<=0:
                        raise RuntimeError(f'{model} did not execute {mode} DP4A linears')
                result['log']=str(log_path)
                record=target/f'{model}.json'
                record.write_text(json.dumps(result,indent=2),encoding='utf-8')
                print(json.dumps(result),flush=True)
        finally:
            processes=bench.own_processes(process)
            for item in processes:
                try:
                    item.terminate()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(processes,timeout=12)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--installation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    parser.add_argument('--models',nargs='+',choices=('base','turbo','pony','xxx'),default=('base','turbo','pony','xxx'))
    parser.add_argument('--port',type=int,default=8396)
    opts=parser.parse_args()
    target,installation=opts.output.resolve(),opts.installation.resolve()
    sys.path.insert(0,str(installation))
    import benchmark_p40 as bench

    target.mkdir(parents=True,exist_ok=True)
    workflows=target/'workflows'
    workflows.mkdir(exist_ok=True)
    bench.PORT=opts.port
    graphs={}
    for model in opts.models:
        graph,save_id,latent,sample=bench.prompt_for(model,'compare',1,None,None)
        graph[save_id]['inputs']['filename_prefix']=model
        saved=workflows/f'{model}.json'
        if saved.is_file() and json.loads(saved.read_text(encoding='utf8'))!=graph:
            raise RuntimeError(f'The saved workflow differs for {model}. Use a new output folder.')
        saved.write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
        graphs[model]=(graph,save_id,latent,sample)
    for mode in opts.modes:
        run_mode(bench,graphs,mode,target/mode,installation,opts.models)


if __name__=='__main__':
    main()
