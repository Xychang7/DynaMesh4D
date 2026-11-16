"""
跨进程版 MHRtoSMPLAdapter (实时进度条版)
"""
import glob
import os
import sys
import pickle
import tempfile
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MHR_CLEAN_PYTHON = os.environ.get("DYNAMESH4D_MHR_CLEAN_PYTHON", "")
MHR_CLEAN_ENV_ROOT = (
    os.path.dirname(os.path.dirname(MHR_CLEAN_PYTHON)) if MHR_CLEAN_PYTHON else ""
)
MHR_WORKER_SCRIPT = os.environ.get(
    "DYNAMESH4D_MHR_WORKER", os.path.join(PROJECT_ROOT, "mhr_worker.py")
)
MHR_SOURCE_ROOT = os.environ.get(
    "DYNAMESH4D_MHR_SOURCE_ROOT", os.path.join(PROJECT_ROOT, "third_party", "MHR-main")
)
MHR_TOOLS_DIR = os.path.join(MHR_SOURCE_ROOT, "tools", "mhr_smpl_conversion")
MHR_ASSETS_ROOT = os.environ.get("DYNAMESH4D_MHR_ASSETS_ROOT", "")

class MHRtoSMPLAdapter:
    def __init__(self, smpl_model_path, device='cuda'):
        self.smpl_model_path = smpl_model_path
        self.worker_device = "cpu"

        if not MHR_CLEAN_PYTHON or not os.path.isfile(MHR_CLEAN_PYTHON):
            raise FileNotFoundError(
                "[Adapter] Set DYNAMESH4D_MHR_CLEAN_PYTHON to the mhr_clean Python executable."
            )
        if not os.path.exists(MHR_WORKER_SCRIPT):
            raise FileNotFoundError(f"[Adapter] 找不到 worker: {MHR_WORKER_SCRIPT}")
        if not os.path.isdir(MHR_TOOLS_DIR):
            raise FileNotFoundError(f"[Adapter] 找不到转换源码: {MHR_TOOLS_DIR}")
        if not MHR_ASSETS_ROOT or not os.path.isdir(MHR_ASSETS_ROOT):
            raise FileNotFoundError(
                "[Adapter] Set DYNAMESH4D_MHR_ASSETS_ROOT to the external MHR assets directory."
            )

        print(f"    ⚙️ [Adapter] 已配置跨进程 MHR→SMPL 转换 (worker: {MHR_CLEAN_PYTHON})", flush=True)

    def process_video_outputs(self, mhr_outputs):
        print(f"    🔄 [Adapter] 正在推送 {len(mhr_outputs)} 帧数据至物理引擎...", flush=True)

        mapped_outputs = []
        for frame in mhr_outputs:
            mapped_frame = frame.copy()
            if "vertices" in frame:
                mapped_frame["pred_vertices"] = frame["vertices"]
            if "pred_cam_t" not in frame:
                mapped_frame["pred_cam_t"] = [0.0, 0.0, 0.0]
            mapped_outputs.append(mapped_frame)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "mhr_in.pkl")
            output_path = os.path.join(tmpdir, "mhr_out.pkl")

            with open(input_path, "wb") as f:
                pickle.dump(mapped_outputs, f)

            torch_lib_dirs = glob.glob(
                os.path.join(MHR_CLEAN_ENV_ROOT, "lib", "python*", "site-packages", "torch", "lib")
            )
            env_lib_dir = f"{MHR_CLEAN_ENV_ROOT}/lib"
            
            child_env = os.environ.copy()
            existing_ld_path = child_env.get("LD_LIBRARY_PATH", "")
            ld_entries = torch_lib_dirs + [env_lib_dir]
            if existing_ld_path:
                ld_entries.append(existing_ld_path)
            child_env["LD_LIBRARY_PATH"] = ":".join(ld_entries)
            child_env["DYNAMESH4D_MHR_SOURCE_ROOT"] = MHR_SOURCE_ROOT
            child_env["DYNAMESH4D_MHR_ASSETS_ROOT"] = MHR_ASSETS_ROOT

            cmd = [
                MHR_CLEAN_PYTHON, MHR_WORKER_SCRIPT,
                "--input", input_path,
                "--output", output_path,
                "--smpl_model_path", self.smpl_model_path,
                "--device", self.worker_device,
                "--mhr_assets_path", MHR_ASSETS_ROOT,
            ]
            
            # =========================================================
            # ⚠️ 核心修复：拆除“消音器”！
            # 去掉 capture_output=True，让 tqdm 进度条直接实时打印到屏幕
            # =========================================================
            result = subprocess.run(
                cmd, env=child_env, cwd=MHR_TOOLS_DIR
            )

            if result.returncode != 0:
                raise RuntimeError(f"[Adapter] worker 子进程执行崩溃 (退出码={result.returncode})。")

            if not os.path.exists(output_path):
                raise RuntimeError("[Adapter] worker 没有生成输出文件。")
            with open(output_path, "rb") as f:
                smpl_outputs = pickle.load(f)

        print("    ✅ [Adapter] 转换完成！", flush=True)
        return smpl_outputs