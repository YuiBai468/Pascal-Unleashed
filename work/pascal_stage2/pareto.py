import csv
import json
from pathlib import Path


def main():
    path=Path(__file__).resolve().parent/'results'
    configs={}
    for name in ('research.json','pop-compensation.json'):
        for record in json.loads((path/name).read_text(encoding='utf-8')):
            if record['shape']!=[4096]*3 or record['data']!='normal':continue
            quality=record['quality']; perf=record['performance']; c=perf['C_fp16']
            label=record['spec']['name']
            configs.setdefault(label,dict(configuration=label,error_percent=100*quality['relative_frobenius'],
                end_to_end_TFLOPS=c['equivalent_TFLOPS'],median_ms=c['median_ms'],p95_ms=c['p95_ms'],
                main_ms=perf['A_main']['median_ms'],prepacked_ms=perf['B_prepacked']['median_ms'],
                fixed_B_ms=perf['D_fixed_B']['median_ms'],host_roundtrip_ms=perf['E_host_roundtrip']['median_ms'],
                half_bitmatch_percent=100*quality['fp16_bitmatch']))
    rows=sorted(configs.values(),key=lambda row:row['error_percent'])
    best=-1
    for row in rows:
        speed=row['end_to_end_TFLOPS']
        row['pareto']=speed>best
        best=max(speed,best)
        row['quality_target']=row['error_percent']<=.05
        row['joint_target']=row['quality_target'] and speed>=20
    with (path/'pareto.csv').open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:
        if row['pareto']:
            print(row['configuration'],round(row['end_to_end_TFLOPS'],4),round(row['error_percent'],6))


if __name__=='__main__':main()
