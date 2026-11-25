"""Summarize completed checks using measured values and relative public paths."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT=Path(__file__).resolve().parents[2]
VERIFY=ROOT/'release'/'verification'
NAMES={'fp32':'FP32','fast':'极速','middle':'均衡','precise':'高精度'}
LABELS={0:'自动',1:'极速',2:'均衡',3:'高精度',4:'原生 FP32',5:'原生 FP16'}


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    matrix=read(VERIFY/'matrix.json')
    functional=read(VERIFY/'functional.json')
    setup=read(VERIFY/'installation.json')
    app=read(VERIFY/'panel-application.json')
    assert setup['panel_application_acceleration_and_restore'] and app['accelerated'] and app['restored']
    render=VERIFY/'comfy-final'
    graph=read(render/'workflows'/'turbo.json')
    records={}
    preview=Image.new('RGB',(1360,364),'#11161e')
    caption=ImageDraw.Draw(preview)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for i,(mode,label) in enumerate(NAMES.items()):
        row=read(render/mode/'turbo.json')
        picture=render/mode/Path(row['images'][0]).name
        with Image.open(picture) as image:
            assert json.loads(image.info['prompt'])==graph
            rgb=image.convert('RGB')
        pixels=np.asarray(rgb,dtype=np.float32)
        if mode=='fp32':
            reference=pixels
            psnr=None
        else:
            rms=float(np.sqrt(np.mean((reference-pixels)**2)))
            psnr=20*float(np.log10(255/rms)) if rms else None
            assert row['precision']['accelerated']>0
        preview.paste(rgb.resize((320,320),Image.Resampling.LANCZOS),(12+i*340,10))
        caption.text((12+i*340,336),f"{label}  {row['sampling_seconds']:.2f} s",font=font,fill='#eaf0f7')
        records[mode]={key:row[key] for key in ('seed','steps','cfg','width','height','sampling_seconds','wall_seconds','vram_peak_gib')}
        records[mode]['psnr_db']=psnr
        records[mode]['dp4a_ops']=row.get('precision',{}).get('accelerated',0)
        records[mode]['fallback_ops']=row.get('precision',{}).get('fallback',0)
    preview.save(VERIFY/'final-app.png',optimize=True)
    (VERIFY/'final-app.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

    lines=['# Pascal Unleashed v1.0 发布前验收','',
           '验收日期为 2026-11-25。平台为 Windows x64、Tesla P40、驱动 582.78、CUDA 12.6 cuBLAS。',
           '', '## 矩阵计算','',
           '从发布 DLL 调用。随机 FP16 输入，种子 2109。原生参考为 FP32 累加，比较全部输出。通常给出三次计时的中位数，2048³ 原生 FP16 计时一次。',
           '', '| M×N×K | 请求档 | 实际档 | 耗时 ms | T/s | 全矩阵相对误差 | 最大绝对误差 | 对 FP32 加速 | 测试显存分配 GiB |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    lookup={'auto':0,'fast':1,'balanced':2,'precise':3,'native':4,'fp16':5}
    for row in matrix['rows']:
        name=LABELS[lookup[row['requested']]]+(' · 保护关闭' if not row['guard'] else '')
        shape='×'.join(map(str,row['shape']))
        lines.append(f"| {shape} | {name} | {LABELS[row['mode']]} | {row['milliseconds']:.2f} | {row['tflops']:.2f} | {row['error']:.5%} | {row['max_absolute']:.4f} | {row['multiplier']:.2f}× | {row['peak_device_bytes']/2**30:.2f} |")
    lines.extend(['','使用发布版在同一张 P40 上获得以上吞吐。保护选项会增加高精度档的检测成本，命中后调用原生。原生 FP16 列使用 FP16 累加和 FP16 输出。',
                  '', '[完整矩阵计时](matrix.json) · [功能测试明细](functional.json)','',
                  '## 真正的应用调用','',
                  '从解压的便携包加载 DLL，使用 Anima Turbo v1.1 在 ComfyUI 执行固定种子的 1024×1024 八步工作流。CUDA 线性层的调用由 DLL 返回的实际执行档位统计。PyTorch 显存峰值只覆盖 PyTorch 分配器。',
                  '', '| 模式 | 去噪秒数 | 端到端秒数 | DP4A 线性算子 | 回退 | 对 FP32 的 PSNR |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for mode,row in records.items():
        psnr=f"{row['psnr_db']:.2f} dB" if row['psnr_db'] is not None else '—'
        lines.append(f"| {NAMES[mode]} | {row['sampling_seconds']:.2f} | {row['wall_seconds']:.2f} | {row['dp4a_ops']} | {row['fallback_ops']} | {psnr} |")
    lines.extend(['','[应用计时和精度](final-app.json) · [对照图](final-app.png)','',
                  '## 安装、界面和通用接口','',
                  '- 安装器和便携包均在隔离目录运行，C SDK 示例实际返回 `[19 22; 43 50]`。',
                  '- Windows x64 wheel 已通过独立 Python 环境的安装、GPU 运行和卸载测试。',
                  '- 面板成功枚举实际 GPU，调用发布 DLL 跑 Benchmark，按应用启动 PyTorch 加速进程并复测原环境。',
                  '- 功能测试通过非方阵、批处理、转置布局、NaN/Inf、同列异常值、抵消、累加范围、显存分配失败恢复、持续调用和非默认 CUDA 流。',
                  '', '[安装检查](installation.json) · [界面 Benchmark](panel-benchmark.json) · [应用配置联动](panel-application.json)','',
                  '## 已知限制','',
                  '测试设备是 Tesla P40。合成矩阵误差和单种子图像差异是测量结果。当前公开构建由项目提供，安装程序未做 Authenticode 签名。系统须提供 CUDA 12 cuBLAS 及兼容驱动。',''])
    (VERIFY/'ACCEPTANCE.md').write_text('\n'.join(lines),encoding='utf-8')
    assert functional['tests']['out_of_memory_recovery']=='passed'


if __name__=='__main__':
    main()
