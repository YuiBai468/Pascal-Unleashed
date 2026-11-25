"""Generate full-matrix FP32 comparison records from the release DLL."""

import argparse
import json
from pathlib import Path

from pascal_unleashed import Context, FALLBACK, OUTLIER_GUARD, devices


def main():
    args=argparse.ArgumentParser()
    args.add_argument('--output',type=Path,required=True)
    args.add_argument('--device',type=int,default=0)
    opt=args.parse_args()
    rows=[]
    scenarios=[('2048³',(2048,2048,2048),('auto','fp16','native','fast','balanced','precise')),
               ('4096³',(4096,4096,4096),('auto','native','fast','balanced','precise')),
               ('8192³',(8192,8192,8192),('auto','fast','balanced','precise')),
               ('16384×16384×8192',(16384,16384,8192),('fast','balanced','precise')),
               ('16384³',(16384,16384,16384),('fast','balanced','precise'))]
    with Context(opt.device) as runtime:
        for name,shape,modes in scenarios:
            for mode in modes:
                benchmark=runtime.benchmark(*shape,mode=mode,samples=1 if mode=='fp16' else 3)
                row=dict(shape=shape,requested=mode,guard=True,seed=2109,**benchmark.as_dict())
                print(name,mode,json.dumps(row),flush=True)
                rows.append(row)
                opt.output.write_text(json.dumps(dict(device=devices()[opt.device],reference='FP32 accumulation on FP16 operands',rows=rows),indent=2),encoding='utf-8')
        # Run the archived high-precision workload with its historical guard setting as well.
        for mode,shape in [('precise',(16384,16384,16384))]:
            benchmark=runtime.benchmark(*shape,mode=mode,flags=FALLBACK,samples=3)
            rows.append(dict(shape=shape,requested=mode,guard=False,seed=2109,**benchmark.as_dict()))
        opt.output.write_text(json.dumps(dict(device=devices()[opt.device],reference='FP32 accumulation on FP16 operands',rows=rows),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
