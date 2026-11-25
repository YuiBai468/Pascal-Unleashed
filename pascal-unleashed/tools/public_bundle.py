"""Create the allowlisted public source tree. Raw snapshots are always local."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from zipfile import ZIP_DEFLATED, ZipFile


ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'pascal-unleashed'
TEXT={'.py','.cu','.h','.c','.cpp','.cs','.md','.txt','.ps1','.json','.csv'}
UUID=re.compile(r'(?i)GPU-[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}')
AUTH=re.compile(r'(?i)(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})')
PRIVATE_PATH=re.compile(r'(?i)[A-Z]:(?:\\+|/)(?:Users|gpt|Python\d+|Work)(?:(?:\\+|/)[^\s"\'<>|,;\)\]}]*)*')
PRIVATE_IP=re.compile(r'\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b')


def scrub(text):
    text=AUTH.sub('<redacted-token>',text)
    text=UUID.sub('GPU-REPLACE-WITH-LOCAL-UUID',text)
    text=PRIVATE_PATH.sub('<local-path>',text)
    text=PRIVATE_IP.sub('<local-address>',text)
    for name in {os.environ.get('USERNAME',''),'<local-user>'}:
        if name and len(name)>=3:
            text=re.sub(re.escape(name),'<local-user>',text,flags=re.IGNORECASE)
    return text


def read(path):
    for coding in ('utf-8-sig','gb18030','cp1252'):
        try: return path.read_text(encoding=coding)
        except UnicodeError: continue
    raise ValueError(f'Unknown text encoding for {path.name}')


def write(stage,path,relative=None):
    relative=relative or path.relative_to(ROOT)
    output=stage/relative
    output.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix in TEXT or path.name=='LICENSE':
        output.write_text(scrub(read(path)),encoding='utf-8',newline='\n')
    else:
        shutil.copyfile(path,output)


def scoped_files(folder):
    for path in sorted(folder.rglob('*')):
        parts=path.relative_to(folder).parts
        if path.is_file() and not any(part in {'build','__pycache__','.git'} for part in parts):
            yield path


def bench_record(stage):
    data=stage/'research'/'four-models.json'
    source=ROOT/'outputs'/'p40-four-models'/'comparison.json'
    selected=('width','height','steps','seed','sampler_name','scheduler','cfg','sampling_seconds','wall_seconds',
              'vram_peak_gib','linear_coverage','psnr_db','rgb_rmse')
    records=json.loads(source.read_text(encoding='utf-8'))
    compact={model:{mode:{key:rec[key] for key in selected if key in rec} for mode,rec in modes.items()} for model,modes in records.items()}
    data.write_text(json.dumps(compact,indent=2),encoding='utf-8')
    from PIL import Image
    with Image.open(source.with_name('comparison.png')) as original:
        original.convert('RGB').save(data.with_name('four-models.png'),optimize=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,type=Path)
    parser.add_argument('--zip',required=True,type=Path)
    args=parser.parse_args()
    stage=args.stage.resolve()
    if stage.exists(): raise FileExistsError(stage)
    stage.mkdir(parents=True)
    for path in scoped_files(PROJECT):
        if path.suffix in TEXT or path.name=='LICENSE': write(stage,path)
    for path in scoped_files(ROOT/'work'):
        if path.suffix in TEXT or (path.suffix=='.npz' and path.name in {'asymmetric-stage10-3p45.npz','folded-lane-0.npz'}):
            write(stage,path)
    research=stage/'research'
    research.mkdir()
    for name in ('Pascal-Unleashed-P40-Tiers.md','Pascal-Unleashed-Mixed-FP32-0.1.json',
                 'Pascal-Unleashed-P40-stream-overlap.json','Pascal-Unleashed-Stage2-Pareto.csv'):
        write(stage,ROOT/'outputs'/name,Path('research')/name)
    bench_record(stage)
    for path in scoped_files(ROOT/'release'/'verification'):
        if path.parent.name=='verification' and path.suffix in ('.json','.md'):
            write(stage,path,Path('verification')/path.name)
    from PIL import Image
    for name in ('panel.png','final-app.png'):
        source=ROOT/'release'/'verification'/name
        if source.is_file():
            target=stage/'verification'/name
            target.parent.mkdir(exist_ok=True)
            with Image.open(source) as picture:
                picture.convert('RGB').save(target,optimize=True)
    intro=read(PROJECT/'README.md')
    intro=re.sub(r'\]\(([A-Z][A-Z.]+(?:\.md)?)\)',lambda m:f'](pascal-unleashed/{m[1]})',intro)
    intro+='\n## 研究档案\n\n[历代基准](research/Pascal-Unleashed-P40-Tiers.md) · [真实生图对比](research/four-models.png) · [逐组数据](research/four-models.json)\n'
    intro+='\n## 发布验收\n\n[安装与 GPU 运行实测](verification/ACCEPTANCE.md) · [控制面板](verification/panel.png) · [发布版 ComfyUI 样本](verification/final-app.png)\n'
    (stage/'README.md').write_text(intro,encoding='utf-8')
    shutil.copyfile(PROJECT/'LICENSE',stage/'LICENSE')
    (stage/'.gitignore').write_text('release/\n**/build/\n**/__pycache__/\n*.pyc\n',encoding='ascii')
    sources=list(scoped_files(stage))
    for path in sources:
        if path.suffix in TEXT or path.name=='LICENSE':
            text=read(path)
            if UUID.search(text) or AUTH.search(text) or PRIVATE_PATH.search(text) or PRIVATE_IP.search(text):
                raise ValueError(f'Sensitive pattern remained in {path.relative_to(stage)}')
    with ZipFile(args.zip,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sources:
            archive.write(path,'Pascal-Unleashed-v1.0/'+path.relative_to(stage).as_posix())
    with ZipFile(args.zip) as archive:
        assert archive.testzip() is None
    print(f'Public source and research files: {len(sources)}')


if __name__=='__main__':
    main()
