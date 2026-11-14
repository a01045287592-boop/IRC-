#!/usr/bin/env python3
"""Run the second tray4 demonstration with its dedicated taught coordinates."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
SETUP = PROJECT_ROOT / 'raccoon_dataset/tray4_setups/setup_20261113_second_demo.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyACM1')
    args = parser.parse_args()
    if not SETUP.is_file():
        parser.error(f'시연 설정 파일이 없습니다: {SETUP}')

    env = os.environ.copy()
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(HERE) + (os.pathsep + existing if existing else '')
    command = [
        sys.executable, str(HERE / 'tray4.py'), 'run',
        '--setup', str(SETUP),
        '--port', args.port,
        '--execute', '--demo-only',
    ]
    raise SystemExit(subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode)


if __name__ == '__main__':
    main()
