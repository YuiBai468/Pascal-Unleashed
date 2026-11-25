"""Audit the matching prompts, compare final pixels and make a contact sheet."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_image(record):
    with Image.open(record['images'][0]) as image:
        prompt = json.loads(image.info['prompt'])
        for node in prompt.values():
            if node['class_type'] == 'SaveImage':
                node['inputs'].pop('filename_prefix')
        return image.convert('RGB'), prompt


def compare(reference, candidate):
    delta = np.asarray(candidate, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
    pixel_rmse = float(np.sqrt(np.mean(delta**2,dtype=np.float64)))
    return {'rgb_mae_255': float(np.mean(np.abs(delta),dtype=np.float64)), 'rgb_rmse_255': pixel_rmse,
            'psnr_db': float(20*np.log10(255/pixel_rmse)) if pixel_rmse else None,
            'fraction_rgb_channels_off_by_16': float(np.mean(np.abs(delta)>16))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    dest = args.output.resolve()
    records = {'fp32': json.loads((dest/'fp32'/'results.jsonl').read_text(encoding='utf-8').strip())}
    for mode in ('turbo','balanced'):
        records[mode] = json.loads((dest/mode/'result.json').read_text(encoding='utf-8'))
    baseline, graph = load_image(records['fp32'])
    fields = ('seed','width','height','batch_size','steps','cfg','sampler_name','scheduler')
    settings = {key: records['fp32'][key] for key in fields}
    canvas = Image.new('RGB',(1536,562), '#171b21')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('segoeui.ttf',22)
    small = ImageFont.truetype('segoeui.ttf',16)
    for i, (mode,title) in enumerate((('fp32','FP32'),('turbo','Fast / INT8'),('balanced','Balanced / INT8'))):
        record = records[mode]
        image, prompt = load_image(record)
        assert prompt == graph, f'Comfy graph changed for {mode}'
        assert {key: record[key] for key in fields} == settings
        assert image.size == baseline.size == (settings['width'],settings['height'])
        if mode != 'fp32':
            report = record['precision']
            record['linear_int8_flop_fraction'] = report['int8_linear_flops'] / (
                report['int8_linear_flops'] + report['fp32_linear_flops'])
            record['image_difference_from_fp32'] = compare(baseline,image)
        image.thumbnail((512,512), Image.Resampling.LANCZOS)
        canvas.paste(image,(i*512,0))
        draw.text((i*512+12,514),title,fill='white',font=font)
        draw.text((i*512+12,540),f"Sampler {record['sampling_seconds']:.2f} s  |  Peak {record['vram_peak_gib']:.2f} GiB",
                  fill='#c3d0dd',font=small)
    canvas.save(dest/'comparison.png')
    report = {'model': 'Anima-Turbo v1.1', 'settings': settings, 'same_comfy_graph': True,
              'runs': records, 'scope': 'Pascal encoding for eligible Anima denoiser linear projections. '
              'Encoders, attention products, other operators and VAE follow the original FP32 workflow.'}
    (dest/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for mode,record in records.items():
        print(mode,record['sampling_seconds'],record['wall_seconds'],record['vram_peak_gib'],
              record.get('linear_int8_flop_fraction'),record.get('image_difference_from_fp32'))


if __name__=='__main__':
    main()
