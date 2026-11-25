import numpy as np

from research import capacity


def inputs(a,b):
    a11,a12=np.hsplit(a[:len(a)//2],2); a21,a22=np.hsplit(a[len(a)//2:],2)
    b11,b12=np.hsplit(b[:len(b)//2],2); b21,b22=np.hsplit(b[len(b)//2:],2)
    return [(a11+a22,b11+b22),(a21+a22,b11),(a11,b12-b22),(a22,b21-b11),
            (a11+a12,b22),(a21-a11,b11+b12),(a12-a22,b21+b22)]


def collapse(v):
    a,b,c,d,e,f,g=v
    return np.block([[a+d-e+g,c+e],[b+d,a-b+c+f]])


def shared_carry(h,g,d):
    lo=-np.floor_divide(-(np.maximum(h,g)-127),d)
    hi=(np.minimum(h,g)+127)//d
    if np.any(lo>hi):raise ValueError('No legal shared shift')
    carry=np.maximum(lo,np.minimum(0,hi))
    return h-d*carry,g-d*carry,carry


def encoded(q,base):
    h=np.rint(q/base).astype(np.int64)
    return h,q-(base-1)*h


def main():
    rng=np.random.default_rng(7); k=4096
    for base in (16,22,26,30,32,38,44,64,128):
        for digit in (63,127):
            cap=capacity(base,digit); q=np.arange(-cap,cap+1,dtype=np.int64)
            h,g=encoded(q,base); r=g-h
            assert max(abs(h).max(),abs(g).max())<=digit
            assert max(abs(r).max(),1)<=127
            assert k*digit**2<=2**31-1 and k*int(np.max(r*r))<=2**31-1
            assert base*base*k*digit**2+(base-1)*k*int(np.max(r*r))<2**63
            print(f'base={base} digit_limit={digit} cap={cap} max_abs_r={max(abs(r))}')
        a=rng.integers(-cap,cap+1,(48,48),dtype=np.int64); b=rng.integers(-cap,cap+1,(48,48),dtype=np.int64)
        ha,ga=encoded(a,base); hb,gb=encoded(b,base)
        assert np.array_equal(a@b,base*((base-1)*(ha@hb)+ga@gb)-(base-1)*((ga-ha)@(gb-hb)))
        for step in (128,192):
            if step>254-base:continue
            # Exhaust h/r combinations at the broader one-level Strassen bounds.
            h=np.arange(-254,255)[:,None]; g=h+np.arange(-base,base+1)
            hp,gp,carry=shared_carry(h,g,step)
            assert np.max(np.abs(hp))<=127 and np.max(np.abs(gp))<=127
            assert np.array_equal(gp-hp,g-h)
            products=[]
            for (h1,h2),(g1,g2) in zip(inputs(ha,hb),inputs(ga,gb)):
                h1p,g1p,c1=shared_carry(h1,g1,step);h2p,g2p,c2=shared_carry(h2,g2,step)
                q1p=(base-1)*h1p+g1p;q2=(base-1)*h2+g2
                z=(base-1)*(h1p@h2p)+g1p@g2p+step*(c1@q2+q1p@c2)
                assert np.array_equal(z,(base-1)*(h1@h2)+g1@g2)
                products.append(z)
            assert np.array_equal(collapse(products),(base-1)*(ha@hb)+ga@gb)
            print(f'  SCC Strassen step={step}: exhaustive carry grid + exact INT64 matrix identity checked')


if __name__=='__main__':main()
