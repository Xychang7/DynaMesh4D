"""
独立 worker 脚本：只在 mhr_clean (Python 3.12 + pymomentum) 环境里运行。
职责单一：读取主进程(sam4d环境)写出的 MHR 输出 pickle，调用官方 Conversion
做 MHR -> SMPL 转换，把结果写回另一个 pickle，由主进程读取。

不 import 任何 sam3 / sam_3d_body / huggingface_hub / mamba_ssm 等重型依赖，
所以这个环境只需要装：pymomentum, mhr, smplx, torch, numpy, trimesh。

用法（由主进程通过 subprocess 调用，也可手动测试）：
    python mhr_worker.py --input /tmp/mhr_in.pkl --output /tmp/mhr_out.pkl \
        --smpl_model_path /path/to/SMPL_NEUTRAL.pkl \
        --mhr_assets_path /path/to/MHR/assets --device cpu
"""

import sys
import os
import argparse
import pickle
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MHR_ROOT = os.environ.get(
    "DYNAMESH4D_MHR_SOURCE_ROOT",
    os.path.join(PROJECT_ROOT, "third_party", "MHR-main"),
)
MHR_TOOLS = os.path.join(MHR_ROOT, "tools", "mhr_smpl_conversion")
MHR_ASSETS_ROOT = os.environ.get("DYNAMESH4D_MHR_ASSETS_ROOT", "")

if MHR_ROOT not in sys.path:
    sys.path.insert(0, MHR_ROOT)
if MHR_TOOLS not in sys.path:
    sys.path.insert(0, MHR_TOOLS)

import torch
import numpy as np
from mhr.mhr import MHR
import smplx

from conversion import Conversion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="主进程写出的 MHR 输出 pickle 路径")
    parser.add_argument("--output", type=str, required=True, help="转换结果要写到哪个 pickle 路径")
    parser.add_argument("--smpl_model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda；pymomentum 后端只支持 cpu")
    parser.add_argument("--mhr_assets_path", type=str, default=MHR_ASSETS_ROOT)
    args = parser.parse_args()

    print(f"[mhr_worker] 读取输入: {args.input}", flush=True)
    with open(args.input, "rb") as f:
        mhr_outputs = pickle.load(f)

    print(f"[mhr_worker] 共 {len(mhr_outputs)} 帧，开始加载模型...", flush=True)
    device = torch.device(args.device)

    if not args.mhr_assets_path or not os.path.isdir(args.mhr_assets_path):
        raise FileNotFoundError(
            "Set --mhr_assets_path or DYNAMESH4D_MHR_ASSETS_ROOT to the external MHR assets directory."
        )
    mhr_model = MHR.from_files(folder=Path(args.mhr_assets_path), lod=1, device=device)
    smpl_model = smplx.SMPL(model_path=args.smpl_model_path).to(device)

    converter = Conversion(
        mhr_model=mhr_model,
        smpl_model=smpl_model,
        method="pytorch",
    )

    print(f"[mhr_worker] 开始转换...", flush=True)
    conversion_results = converter.convert_sam3d_output_to_smpl(
        sam3d_outputs=mhr_outputs,
        return_smpl_meshes=False,
        return_smpl_parameters=True,
        return_smpl_vertices=True,
        return_fitting_errors=False,
    )

    smpl_vertices = conversion_results.result_vertices
    smpl_parameters = conversion_results.result_parameters

    def to_numpy(x):
        """兼容 conversion_results 里的字段可能是 torch.Tensor 也可能
        已经是 numpy.ndarray（比如走了优化求解路径时返回的就是 numpy）。"""
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    smpl_outputs = []
    for i in range(len(mhr_outputs)):
        out_dict = {**mhr_outputs[i]}

        if smpl_vertices is not None:
            out_dict['pred_vertices'] = to_numpy(smpl_vertices[i])

        if smpl_parameters is not None:
            out_dict['smpl_betas'] = to_numpy(smpl_parameters['betas'][i])
            out_dict['smpl_body_pose'] = to_numpy(smpl_parameters['body_pose'][i])
            out_dict['smpl_global_orient'] = to_numpy(smpl_parameters['global_orient'][i])
            if 'transl' in smpl_parameters:
                out_dict['smpl_transl'] = to_numpy(smpl_parameters['transl'][i])

        smpl_outputs.append(out_dict)

    print(f"[mhr_worker] 转换完成，写出结果到: {args.output}", flush=True)
    with open(args.output, "wb") as f:
        pickle.dump(smpl_outputs, f)

    print("[mhr_worker] 完成。", flush=True)


if __name__ == "__main__":
    main()