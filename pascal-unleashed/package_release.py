"""Package Windows binaries, an installer payload and a platform Python wheel."""

import base64
import hashlib
from pathlib import Path
import shutil
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT=Path(__file__).resolve().parent
DIST=ROOT.parent/'release'
BUILD=ROOT/'build'
BIN=BUILD/'bin'
PORTABLE='Pascal-Unleashed-v1.0-win-x64-portable.zip'
DIST_INFO='pascal_unleashed-1.0.0.dist-info'


def add(archive,name,data):
    entry=ZipInfo(name,date_time=(2026,11,25,0,0,0))
    entry.compress_type=ZIP_DEFLATED
    archive.writestr(entry,data,compresslevel=6)


def package():
    DIST.mkdir(exist_ok=True)
    docs={path.name:path for path in ROOT.glob('*.md')}
    binaries={name:BIN/name for name in ('PascalUnleashedPanel.exe','PascalUnleashed.dll','pascal_sm61.ptx','PascalSdkExample.exe','PascalSdkCppExample.exe')}
    package_files={f'python/{path.relative_to(ROOT/"python").as_posix()}':path
                   for path in (ROOT/'python').rglob('*.py')}
    resources=docs|binaries|package_files|{'LICENSE':ROOT/'LICENSE','sdk/PascalUnleashed.lib':BIN/'PascalUnleashed.lib',
        'sdk/pascal_unleashed.h':ROOT/'include'/'pascal_unleashed.h','sdk/sdk_example.c':ROOT/'tools'/'sdk_example.c',
        'sdk/sdk_cpp_example.cpp':ROOT/'tools'/'sdk_cpp_example.cpp'}
    manifest=[]
    with ZipFile(DIST/PORTABLE,'w') as archive:
        for name,path in sorted(resources.items()):
            data=path.read_bytes()
            add(archive,name,data)
            manifest.append(f'{hashlib.sha256(data).hexdigest()}  {name}')
        add(archive,'manifest.sha256',('\n'.join(manifest)+'\n').encode('ascii'))
    shutil.copyfile(DIST/PORTABLE,BUILD/'installer-payload.zip')

    records=[]
    path=DIST/'pascal_unleashed-1.0.0-py3-none-win_amd64.whl'
    with ZipFile(path,'w') as wheel:
        files={name.removeprefix('python/'):path.read_bytes() for name,path in package_files.items() if name.startswith('python/pascal_unleashed/')}
        files.update({f'pascal_unleashed/{name}':binaries[name].read_bytes() for name in ('PascalUnleashed.dll','pascal_sm61.ptx')})
        files[f'{DIST_INFO}/licenses/LICENSE']=resources['LICENSE'].read_bytes()
        files[f'{DIST_INFO}/METADATA']=(
            'Metadata-Version: 2.4\nName: pascal-unleashed\nVersion: 1.0.0\n'
            'Summary: Opt-in FP16 GEMM for Pascal CUDA devices\nRequires-Python: >=3.9\nLicense-Expression: MIT\n'
            'License-File: LICENSE\n\n').encode()
        files[f'{DIST_INFO}/WHEEL']=b'Wheel-Version: 1.0\nGenerator: Pascal Unleashed\nRoot-Is-Purelib: false\nTag: py3-none-win_amd64\n'
        for name,data in sorted(files.items()):
            add(wheel,name,data)
            checksum=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            records.append(f'{name},sha256={checksum},{len(data)}')
        records.append(f'{DIST_INFO}/RECORD,,')
        add(wheel,f'{DIST_INFO}/RECORD',('\n'.join(records)+'\n').encode())


if __name__=='__main__':
    package()
