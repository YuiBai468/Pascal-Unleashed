"""Verify the replayed image settings and make comparison contact sheets."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


MODELS={
    'base':'Anima Base v1.0',
    'turbo':'Anima Turbo v1.1',
    'pony':'CyberRealistic Pony v18 CoreShift',
    'xxx':'XXX-Ray v1.1 DMD2',
}
MODES={'fp32':'FP32','fast':'极速档','middle':'均衡档','precise':'高精度档'}


def load(target):
    images,records={},{}
    for model in MODELS:
        graph=json.loads((target/'workflows'/f'{model}.json').read_text(encoding='utf-8'))
        images[model],records[model]={},{}
        fp32=None
        for mode in MODES:
            rec=json.loads((target/mode/f'{model}.json').read_text(encoding='utf-8'))
            if rec['model']!=model or rec['run']!=mode or len(rec['images'])!=1:
                raise ValueError(f'Unexpected render record for {model}/{mode}')
            path=Path(rec['images'][0])
            with Image.open(path) as original:
                if json.loads(original.info['prompt'])!=graph or original.size!=(rec['width'],rec['height']):
                    raise ValueError(f'Workflow or output dimensions differ for {model}/{mode}')
                image=original.convert('RGB')
            arr=np.asarray(image,dtype=np.float32)
            if not np.isfinite(arr).all() or arr.std()<5:
                raise ValueError(f'Invalid image for {model}/{mode}')
            images[model][mode]=image
            rec['file']=path.relative_to(target).as_posix()
            if mode=='fp32':
                fp32=arr
            else:
                err=float(np.sqrt(np.mean((arr-fp32)**2)))
                rec['rgb_rmse']=err
                rec['psnr_db']=20*math.log10(255/err) if err else None
                precision=rec['precision']
                if precision['mode']!=mode or precision['int8_linear_calls']<=0:
                    raise ValueError(f'Missing actual INT8 denoiser work for {model}/{mode}')
                quantized,other=(precision[k] for k in ('int8_linear_flops','fp32_linear_flops'))
                rec['linear_coverage']=quantized/(quantized+other)
            records[model][mode]=rec
    return images,records


def font(size):
    try:
        return ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
    except OSError:
        return ImageFont.load_default(size)


def sheet(models,images,records,output):
    margin, gap, width, height = 18, 10, 354, 374
    row_height=height+72
    canvas=Image.new('RGB',(margin*2+4*width+3*gap,margin*2+len(models)*row_height),'#14171a')
    draw=ImageDraw.Draw(canvas)
    heading, caption=font(22),font(16)
    for row,model in enumerate(models):
        y=margin+row*row_height
        draw.text((margin,y),MODELS[model],fill='#f4f6f9',font=heading)
        for col,(mode,label) in enumerate(MODES.items()):
            x=margin+col*(width+gap)
            frame=ImageOps.contain(images[model][mode],(width,height),Image.Resampling.LANCZOS)
            canvas.paste(frame,(x+(width-frame.width)//2,y+32+(height-frame.height)//2))
            r=records[model][mode]
            detail=f"{label}   {r['sampling_seconds']:.1f} s"
            if mode!='fp32' and r['psnr_db'] is not None:
                detail+=f"   {r['psnr_db']:.1f} dB"
            draw.text((x+8,y+height+38),detail,fill='#d5dee5',font=caption)
    canvas.save(output,optimize=True)


def report(target,records):
    lines=['# P40 生图四档对照','',
           '各模型保留原工作流的分辨率、步数、采样器和引导系数。每个模型的四张图使用完全相同的提示词、种子和初始尺寸。',
           '',
           '采样时间包含该次去噪用到的权重首次打包。端到端时间还包含模型加载、文本编码、VAE 解码及队列检查。三档在去噪器符合条件的线性层使用 FP16 输入、INT8 DP4A 和 FP32 重构。其他运算使用 FP32。',
           'DP4A 占比以记录的去噪线性层乘加量为分母。PSNR 越高，表示成图像素越接近 FP32 基线。',
           '']
    for model,title in MODELS.items():
        baseline=records[model]['fp32']
        lines.extend([f'## {title}','',
            f"{baseline['width']}×{baseline['height']}，{baseline['steps']} 步，CFG {baseline['cfg']}，种子 {baseline['seed']}。",'',
            '[四图并排](compare-'+model+'.png) · [工作流](workflows/'+model+'.json)','',
            '| 档位 | 原图 | 采样耗时 | 端到端 | 显存峰值 | DP4A 线性计算占比 | 对 FP32 的 PSNR |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
        for mode,label in MODES.items():
            r=records[model][mode]
            coverage=f"{r['linear_coverage']:.3%}" if mode!='fp32' else '—'
            psnr=f"{r['psnr_db']:.2f} dB" if r.get('psnr_db') is not None else '—'
            lines.append(f"| {label} | [PNG]({r['file']}) | {r['sampling_seconds']:.2f} s | {r['wall_seconds']:.2f} s | {r['vram_peak_gib']:.2f} GiB | {coverage} | {psnr} |")
        lines.append('')
    check=target/'implementation-check.json'
    if check.is_file():
        rows=json.loads(check.read_text(encoding='utf-8'))
        lines.extend(['## 矩阵复核','',
                      '在 P40 上用 256×8192 乘 8192×256 的 FP16 矩阵复核两套实现。异常值实验在输入的相同三列放大 32 倍。',
                      '', '| 档位 | 数据分布 | 原版相对误差 | 生图接口相对误差 | 两实现输出相对差异 |',
                      '| --- | --- | ---: | ---: | ---: |'])
        for row in rows:
            distribution='普通正态' if row['case']=='normal' else '同列异常值'
            lines.append(f"| {MODES[row['mode']]} | {distribution} | {row['archived_error']:.4%} | {row['adapted_error']:.4%} | {row['implementation_delta']:.6%} |")
        lines.extend(['','这个实验复现了高精度档对同列异常值的敏感性。四组图像的输入分布尚未逐层测量。',''])
    (target/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    opts=parser.parse_args()
    target=opts.output.resolve()
    images,records=load(target)
    sheet(MODELS,images,records,target/'comparison.png')
    for model in MODELS:
        sheet((model,),images,records,target/f'compare-{model}.png')
    (target/'comparison.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    report(target,records)


if __name__=='__main__':
    main()
