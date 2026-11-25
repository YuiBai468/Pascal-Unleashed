param(
    [string]$Python = 'python',
    [string]$Zig = 'zig',
    [string]$Cuda12Directory = $env:PASCAL_CUDA_DLLS,
    [switch]$CompileOnly
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8='1'
if ($Cuda12Directory) { $env:PASCAL_CUDA_DLLS = (Resolve-Path -LiteralPath $Cuda12Directory).Path }
$csc = Join-Path ([Runtime.InteropServices.RuntimeEnvironment]::GetRuntimeDirectory()) 'csc.exe'
if (!(Test-Path -LiteralPath $csc)) { $csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe' }
if (!(Test-Path -LiteralPath $csc)) { throw 'The .NET Framework 4.x C# compiler is required.' }
Get-Command $Python,$Zig -ErrorAction Stop | Out-Null

& $Python build_kernels.py build/bin
if ($LASTEXITCODE) { throw 'CUDA kernel compilation failed.' }
& $Zig cc -target x86_64-windows-gnu -O3 -std=c11 -shared -Iinclude -Ibuild/bin native/pascal_unleashed.c -o build/bin/PascalUnleashed.dll '-Wl,--strip-all' '-Wl,--out-implib,build/bin/PascalUnleashed.lib'
if ($LASTEXITCODE) { throw 'Native DLL compilation failed.' }
& $Zig cc -target x86_64-windows-gnu -O2 -Iinclude tools/sdk_example.c build/bin/PascalUnleashed.lib -o build/bin/PascalSdkExample.exe '-Wl,--strip-all'
if ($LASTEXITCODE) { throw 'Native SDK example compilation failed.' }
& $Zig cc -target x86_64-windows-gnu -x c++ -std=c++17 -fno-exceptions -O2 -Iinclude -c tools/sdk_cpp_example.cpp -o build/bin/sdk_cpp_example.o
if ($LASTEXITCODE) { throw 'C++ SDK example compilation failed.' }
& $Zig cc -target x86_64-windows-gnu build/bin/sdk_cpp_example.o build/bin/PascalUnleashed.lib -o build/bin/PascalSdkCppExample.exe '-Wl,--strip-all'
if ($LASTEXITCODE) { throw 'C++ SDK example linking failed.' }
& $csc /nologo /target:winexe /platform:x64 /optimize+ /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.Web.Extensions.dll '/out:build\bin\PascalUnleashedPanel.exe' 'panel\Panel.cs'
if ($LASTEXITCODE) { throw 'Control panel compilation failed.' }
if ($CompileOnly) { return }
& $Python package_release.py
if ($LASTEXITCODE) { throw 'Packaging failed.' }
& $csc /nologo /target:winexe /platform:x64 /optimize+ /reference:System.Windows.Forms.dll /reference:System.IO.Compression.dll /reference:System.IO.Compression.FileSystem.dll /reference:Microsoft.CSharp.dll '/resource:build\installer-payload.zip,release.zip' '/out:..\release\Pascal-Unleashed-v1.0-win-x64-setup.exe' 'panel\Setup.cs'
if ($LASTEXITCODE) { throw 'Installation program compilation failed.' }
