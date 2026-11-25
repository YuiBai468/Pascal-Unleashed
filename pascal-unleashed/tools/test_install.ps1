param([string]$Python='python',[string]$Cuda12Directory=$env:PASCAL_CUDA_DLLS)
$ErrorActionPreference='Stop'
$release=Resolve-Path (Join-Path $PSScriptRoot '..\..\release')
$verification=Join-Path $release 'verification'
$work=Join-Path $env:TEMP ('PascalUnleashed-verify-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
$install=Join-Path $work 'Installed'
$portable=Join-Path $work 'Portable'
$setup=Join-Path $release 'Pascal-Unleashed-v1.0-win-x64-setup.exe'
$wheel=Join-Path $release 'pascal_unleashed-1.0.0-py3-none-win_amd64.whl'
$registration='HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\PascalUnleashed'
$state=Join-Path $work 'State'
$env:PU_STATE_DIR=$state
$env:PASCAL_CUDA_DLLS=$Cuda12Directory
Remove-Item Env:PASCAL_UNLEASHED_DLL -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
$tests=[ordered]@{}

function Invoke-Waited([string]$Executable,[string[]]$Arguments) {
    $process=Start-Process -FilePath $Executable -ArgumentList $Arguments -WindowStyle Hidden -PassThru -Wait
    if ($process.ExitCode) { throw "$(Split-Path $Executable -Leaf) exited with $($process.ExitCode)" }
}

$register=!(Test-Path -LiteralPath $registration)
$args=@('--quiet','--dir',('"'+$install+'"'),'--log',('"'+(Join-Path $work 'setup-error.txt')+'"'))
if (!$register) { $args+='--no-register' }
try {
    Invoke-Waited $setup $args
    if (!(Test-Path (Join-Path $install 'PascalUnleashed.dll'))) { throw 'Core DLL missing after installation' }
    $tests.installed=$true
    if ($register) {
        $record=Get-ItemProperty -LiteralPath $registration
        if ($record.InstallLocation -ne $install) { throw 'Application registration did not match test install' }
    }
    $tests.windows_registration=$register
    Expand-Archive -LiteralPath (Join-Path $release 'Pascal-Unleashed-v1.0-win-x64-portable.zip') -DestinationPath $portable
    $tests.portable=$true
    $panel=Join-Path $install 'PascalUnleashedPanel.exe'
    $probe=Join-Path $verification 'panel-probe.json'
    $shot=Join-Path $verification 'panel.png'
    Invoke-Waited $panel @('--probe',('"'+$probe+'"'))
    Invoke-Waited $panel @('--render',('"'+$shot+'"'))
    if (!(Test-Path -LiteralPath $shot)) { throw 'Panel screenshot missing' }
    $tests.panel_rendered=$true
    $panelbench=Join-Path $verification 'panel-benchmark.json'
    Invoke-Waited (Join-Path $portable 'PascalUnleashedPanel.exe') @('--benchmark',('"'+$panelbench+'"'),'0','2048','2048','2048','2')
    $benchmark=Get-Content $panelbench -Raw | ConvertFrom-Json
    if ($benchmark.Run.Accelerated -lt 1) { throw 'Panel benchmark did not reach the GPU acceleration backend' }
    $tests.panel_gpu_benchmark=$true
    $appReport=Join-Path $verification 'panel-application.json'
    $pythonApp=(Get-Command $Python -ErrorAction Stop).Source
    $appScript=Join-Path $PSScriptRoot 'app_smoke.py'
    Invoke-Waited $panel @('--verify-app',('"'+$appReport+'"'),('"'+$pythonApp+'"'),('"'+$appScript+'"'))
    $tests.panel_application_acceleration_and_restore=$true
    & (Join-Path $install 'PascalSdkExample.exe')
    if ($LASTEXITCODE) { throw 'C SDK installed example failed' }
    $tests.c_sdk_example=$true
    & (Join-Path $install 'PascalSdkCppExample.exe')
    if ($LASTEXITCODE) { throw 'C++ SDK installed example failed' }
    $tests.cpp_sdk_example=$true
    & $Python -m venv (Join-Path $work 'Python')
    if ($LASTEXITCODE) { throw 'Cannot create isolated Python environment' }
    $isolated=Join-Path $work 'Python\Scripts\python.exe'
    & $isolated -m pip --disable-pip-version-check install --no-deps $wheel
    if ($LASTEXITCODE) { throw 'Wheel installation failed' }
    & $isolated -c 'import pascal_unleashed as p; ctx=p.Context(); result=ctx.benchmark(256,256,256,mode="fast",samples=1); assert result.run.accelerated==1; ctx.close()'
    if ($LASTEXITCODE) { throw 'Installed Python wheel did not execute CUDA' }
    & $isolated -m pip --disable-pip-version-check uninstall -y pascal-unleashed
    if ($LASTEXITCODE) { throw 'Wheel uninstall failed' }
    & $isolated -c 'import importlib.util; assert importlib.util.find_spec("pascal_unleashed") is None'
    if ($LASTEXITCODE) { throw 'Wheel remained importable after uninstall' }
    $tests.python_wheel_install_run_uninstall=$true
    New-Item -ItemType Directory -Path $state -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $state 'configuration-probe.txt') -Value 'preserve'
    Invoke-Waited $setup @('--uninstall','--quiet','--dir',('"'+$install+'"'))
    if (Test-Path -LiteralPath $install) { throw 'Installed directory remained after uninstall' }
    if ($register -and (Test-Path -LiteralPath $registration)) { throw 'Application registration remained after uninstall' }
    if ((Get-Content (Join-Path $state 'configuration-probe.txt')) -ne 'preserve') { throw 'Uninstall modified profile state' }
    $tests.uninstall_preserves_configuration=$true
    $tests | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $verification 'installation.json')
} finally {
    if (Test-Path (Join-Path $install '.pascal-unleashed-installed')) {
        Invoke-Waited $setup @('--uninstall','--quiet','--dir',('"'+$install+'"'))
    }
}
