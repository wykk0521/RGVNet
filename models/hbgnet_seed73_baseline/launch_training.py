"""Persistent seed73 baseline launcher; validation only, never auto-test."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN = 'hbgnet_protocol_b_seed73_baseline'
LOG = Path('/root/wyk5/RGVNet/z_1_trainning_log/AI4Boundaries_1m') / RUN
WEIGHTS = Path('/root/wyk5/RGVNet/z_3_weights/AI4Boundaries_1m') / RUN
LAUNCH = ROOT / 'formal_launch_01'


def main():
    with (ROOT/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if LOG.exists() or WEIGHTS.exists() or LAUNCH.exists():
            raise FileExistsError('Fresh launch refuses existing experiment or launch evidence')
        LAUNCH.mkdir()
        env = dict(os.environ)
        env.update(PYTHONPATH=f'{ROOT}/_deps:/root/wyk5/RGVNet/.python_packages:/root/wyk5/RGVNet',
                   CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', '-u', str(ROOT/'train.py'),
                   '--config', str(ROOT/'config_protocol_b.yaml'), '--device', 'cuda:0']
        with (LAUNCH/'console.log').open('x') as console:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=console, stderr=subprocess.STDOUT)
            (LAUNCH/'launch.json').write_text(json.dumps({
                'run_name': RUN, 'pid': process.pid, 'supervisor_pid': os.getpid(),
                'command': command, 'cuda_visible_devices': '0',
                'started_unix': time.time(), 'resume': False, 'auto_test': False
            }, indent=2) + '\n')
            linked = False
            while process.poll() is None:
                if LOG.is_dir() and not linked:
                    (LOG/'console.log').symlink_to(LAUNCH/'console.log')
                    for source in (ROOT/'verification.json', ROOT/'pretrained/pvt_v2_b2.json', LAUNCH/'launch.json'):
                        target = LOG/source.name
                        if target.exists():
                            raise FileExistsError(f'Refuse existing evidence: {target}')
                        shutil.copy2(source, target)
                    linked = True
                time.sleep(1)
            code = process.wait()
        (LAUNCH/'exit_code.txt').write_text(str(code) + '\n')
        if LOG.is_dir():
            (LOG/'exit_code.txt').write_text(str(code) + '\n')
        return code


if __name__ == '__main__':
    sys.exit(main())
