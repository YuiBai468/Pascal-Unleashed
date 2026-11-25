"""Compare same-schedule images and latent differences from the FP32 trajectory."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare import compare, load_image


MODES = (('fp32','FP32'),('turbo','Fast / INT8'),('balanced','Balanced / INT8'))


def normalized_graph(graph):
    prompt = deepcopy(graph)
    sample = next(node['inputs'] for node in prompt.values() if node['class_type']=='KSampler')
    sample.pop('steps')
    return prompt


def normalized_error(reference,candidate):
    a,b = reference.astype(np.float64),candidate.astype(np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.linalg.norm((a-b).ravel()) / max(1e-30,np.linalg.norm(a.ravel())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', nargs='+', type=int, default=(8,40))
    args = parser.parse_args()
    dest = args.output.resolve()
    reference_graph = None
    baseline_images = {}
    results = {}
    canvas = Image.new('RGB',(1536,566*len(args.steps)), '#171b21')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('segoeui.ttf',22)
    small = ImageFont.truetype('segoeui.ttf',16)
    for lane,steps in enumerate(args.steps):
        runs = {}
        reference_latents = None
        for col,(mode,title) in enumerate(MODES):
            mode_dir = dest/mode
            record = json.loads((mode_dir/f'result-{steps}.json').read_text(encoding='utf-8'))
            image,graph = load_image(record)
            assert record['steps']==steps
            if reference_graph is None:
                reference_graph = normalized_graph(graph)
            assert normalized_graph(graph)==reference_graph
            with np.load(mode_dir/f"latents-{steps}-{record['seed']}.npz",allow_pickle=False) as data:
                latents = {key:data[key] for key in data.files}
            assert {'denoised_001',f'denoised_{steps:03d}','final'} <= set(latents)
            if mode=='fp32':
                reference_latents = latents
                first = image.copy()
                baseline_images[steps] = first
            else:
                assert latents.keys()==reference_latents.keys()
                record['latent_relative_frobenius'] = {
                    key:normalized_error(reference_latents[key],latents[key]) for key in latents}
                record['image_difference_from_fp32'] = compare(first,image)
                pol = record['precision']
                record['linear_int8_flop_fraction']=pol['int8_linear_flops'] / (
                    pol['int8_linear_flops']+pol['fp32_linear_flops'])
            image.thumbnail((512,512),Image.Resampling.LANCZOS)
            x,y=col*512,lane*566
            canvas.paste(image,(x,y))
            draw.text((x+12,y+514),f'{steps} steps  |  {title}',font=font,fill='white')
            draw.text((x+12,y+541),f"Sampler {record['sampling_seconds']:.2f} s  |  Peak {record['vram_peak_gib']:.2f} GiB",
                      font=small,fill='#c3d0dd')
            runs[mode]=record
        results[steps]=runs
    canvas.save(dest/'comparison.png')
    baseline_differences = {str(steps):compare(baseline_images[args.steps[0]],baseline_images[steps])
                            for steps in args.steps[1:]}
    report = {'model':'Anima Base v1.0','same_graph_except_steps_and_filename':True,'runs':results,
              'fp32_differences_from_shortest_schedule':baseline_differences,
              'timing':'Each precision mode starts a new process. Models warm up on the first schedule. '
                       'The second schedule reuses the model and, for INT8, its packed weight cache.',
              'scope':'INT8 for eligible Anima denoiser linear projections. Attention products, text encoder, '
                      'normalization and VAE use the original FP32 workflow.'}
    (dest/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for steps,runs in results.items():
        for mode,r in runs.items():
            print(steps,mode,r['sampling_seconds'],r['wall_seconds'],r.get('image_difference_from_fp32',{}),
                  r.get('latent_relative_frobenius',{}))


if __name__=='__main__':
    main()
