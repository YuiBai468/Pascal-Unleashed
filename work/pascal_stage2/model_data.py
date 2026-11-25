"""Capture GEMM operands from a trained network on genuine handwritten images."""
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    torch.manual_seed(71)
    torch.set_num_threads(3)
    images,labels=load_digits(return_X_y=True)
    train_x,test_x,train_y,test_y=train_test_split(images,labels,test_size=0.25,random_state=71,stratify=labels)
    x=torch.tensor(train_x,dtype=torch.float32)/16; y=torch.tensor(train_y)
    test=torch.tensor(test_x,dtype=torch.float32)/16
    network=torch.nn.Sequential(torch.nn.Linear(64,512),torch.nn.GELU(),
        torch.nn.Linear(512,512),torch.nn.GELU(),torch.nn.Linear(512,10)).cpu()
    optim=torch.optim.AdamW(network.parameters(),lr=0.002)
    for _ in range(20):
        for batch in torch.randperm(len(x)).split(128):
            optim.zero_grad();torch.nn.functional.cross_entropy(network(x[batch]),y[batch]).backward();optim.step()
    with torch.no_grad():
        accuracy=float((network(test).argmax(1)==torch.tensor(test_y)).float().mean())
    h=network[:2](x[:256]); pre=network[2](h); pre.retain_grad()
    torch.nn.functional.cross_entropy(network[3:](pre),y[:256]).backward()
    f16=lambda t:np.ascontiguousarray(t.detach().numpy().astype(np.float16))
    out=Path(__file__).with_name('digits-operands.npz')
    np.savez_compressed(out,inference_a=f16(network[:2](test[:256])),inference_b=f16(network[2].weight),
                       gradient_a=f16(pre.grad.T),gradient_b=f16(h.T))
    print(f'CPU trained digits MLP held-out accuracy={accuracy:.6f}; forward and weight-gradient operands saved')


if __name__=='__main__':main()
