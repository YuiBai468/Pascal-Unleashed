param(
    [Parameter(Mandatory=$true)][string]$Gpu,
    [ValidateSet('Quick','Full')][string]$Mode='Quick'
)
$ErrorActionPreference='Stop'
Push-Location -LiteralPath $PSScriptRoot
function Invoke-Python {
    & python @args
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
}
try {
    Invoke-Python -m unittest -v test_stage2
    Invoke-Python proofs.py
    if ($Mode -eq 'Quick') {
        Invoke-Python research.py --gpu $Gpu --shapes 1024 --specs AR2-16 exact-128 --samples 5 --output results/quick.json
    } else {
        Invoke-Python microbench.py --gpu $Gpu
        Invoke-Python streamdot.py --gpu $Gpu
        Invoke-Python sparsebench.py --gpu $Gpu
        Invoke-Python fused_lab.py --gpu $Gpu
        Invoke-Python research.py --gpu $Gpu --shapes 1024 2048 4096
        Invoke-Python research.py --gpu $Gpu --shapes 1024 --cases uniform finite_range outliers co_spike span cancellation --specs AR2-16 exact-128 OP-16 --samples 5 --output results/stress-reproduced.json
        Invoke-Python research.py --gpu $Gpu --shapes 256x4096x2048 32x4096x4096 --specs AR2-16 exact-128 LRC-16-unclipped --output results/aspect.json
        Invoke-Python lowbit_lab.py --gpu $Gpu
        Invoke-Python strassen_lab.py --gpu $Gpu
        Invoke-Python research.py --gpu $Gpu --shapes 4096 --specs OP-3 OP-3-POPC LRC-2-POPC OP-2-POPC --samples 5 --output results/pop-compensation.json
        foreach ($Seed in 29,101) {
            Invoke-Python research.py --gpu $Gpu --shapes 4096 --specs AR2-16 exact-128 LRC-16-unclipped --seed $Seed --output "results/gaussian-$Seed.json"
        }
        Invoke-Python model_data.py
        Invoke-Python research.py --gpu $Gpu --model-data digits-operands.npz --specs AR2-16 exact-128 LRC-16-unclipped --output results/model.json
        Invoke-Python pareto.py
    }
} finally { Pop-Location }
