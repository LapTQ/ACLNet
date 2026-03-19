from pathlib import Path

ACLNET_DIR = Path(__file__).parent.parent

ls_path_cfg = [
    str(ACLNET_DIR / "configs/fs26" / "v219--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2.py"),
    str(ACLNET_DIR / "configs/fs26" / "v220--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--b.py"),
    str(ACLNET_DIR / "configs/fs26" / "v221--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--jm.py"),
    str(ACLNET_DIR / "configs/fs26" / "v222--satudora_veo3_awlrecord--split-14-class--nodistinct--2s-15frames--v2--bm.py"),
    # str(ACLNET_DIR / "configs/fs26" / "j.py"),
]
num_models = 10
ls_devices = [5, 5, 5, 4, 4, 4, 3, 3, 3, 2]

import subprocess
import multiprocessing as mp
import concurrent.futures

import time

for path_cfg in ls_path_cfg:
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(
                subprocess.run,
                args=f"CUDA_VISIBLE_DEVICES={device} bash tools/dist_train.sh {path_cfg} 1 --validate --id_model {id_model}",
                cwd=str(ACLNET_DIR),
                shell=True,
                check=True,
                text=True,
            )
            for id_model, device in zip(range(num_models), ls_devices)
        ]
        ls_trained_model = [f.result() for f in futures]
