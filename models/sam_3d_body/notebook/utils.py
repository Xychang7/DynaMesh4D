import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sam_3d_body.data.utils.io import load_image
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from sam_3d_body.utils import recursive_to

# ==========================================
# 1. Bi-Mamba Motion Smoother 模型定义
# ==========================================
try:
    from mamba_ssm import Mamba
except ImportError:
    print("⚠️ [Warning] mamba_ssm not found. Please install via 'pip install mamba-ssm'")
    Mamba = None

# 【消融实验全局开关】
# 设为 True: 屏蔽反向扫描，强制降级为单向 Uni-Mamba
# 设为 False: 正常的双向 Bi-Mamba (Ours)
UNI_MAMBA_MODE = False  

class BiMambaSmoother(nn.Module):
    def __init__(self, input_dim=266, d_model=256, n_layers=4):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm module is missing.")
            
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 正向 Mamba
        self.layers_fwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        # 反向 Mamba
        self.layers_bwd = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Dim)
        residual = x
        x_embed = self.embedding(x)
        
        # 1. Forward Pass (始终执行，提取因果历史信息)
        x_f = x_embed
        for layer in self.layers_fwd:
            x_f = layer(x_f)
            
        # 2. Backward Pass (根据开关决定是否屏蔽)
        if UNI_MAMBA_MODE:
            # 【核心消融设计】：强行用前向特征替代后向特征，切断未来时序演化
            x_b = x_f 
        else:
            x_b = torch.flip(x_embed, dims=[1])
            for layer in self.layers_bwd:
                x_b = layer(x_b)
            x_b = torch.flip(x_b, dims=[1])
        
        # 3. Fusion
        x_merged = self.norm(x_f + x_b)
        correction = self.output_head(x_merged)
        
        return residual + correction    


SMOOTHER_MODEL = None

def load_smoother_model(device):
    global SMOOTHER_MODEL
    
    # return None
    
    if SMOOTHER_MODEL is None:
        try:
            model = BiMambaSmoother(input_dim=266, d_model=256, n_layers=4).to(device)
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..")
            )
            default_ckpt_path = os.path.join(project_root, "mamba_smoother_epoch_15.pth")
            ckpt_path = os.environ.get("DYNAMESH4D_MAMBA_CHECKPOINT", default_ckpt_path)
            
            if os.path.exists(ckpt_path):
                print(f"🐍 [Mamba] Loading weights from {ckpt_path}...")
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                model.eval()
                SMOOTHER_MODEL = model
            else:
                print(f"⚠️ [Mamba] Checkpoint not found at {ckpt_path}. Skipping时序平滑.")
                SMOOTHER_MODEL = "DISABLED"
        except Exception as e:
            print(f"❌ [Mamba] Error loading model: {e}")
            SMOOTHER_MODEL = "DISABLED"
            
    return SMOOTHER_MODEL if SMOOTHER_MODEL != "DISABLED" else None


# ==========================================
# 2. 核心处理与工具函数
# ==========================================
def dummy_transform(x):
    return x

def collate_batches(batch_list):
    collated_batch = {}
    if len(batch_list) == 0: return collated_batch
    keys = batch_list[0].keys()
    for key in keys:
        values = [b[key] for b in batch_list]
        if key == 'img' and isinstance(values[0], torch.Tensor):
            collated_batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], torch.Tensor):
            try: collated_batch[key] = torch.cat(values, dim=0)
            except: collated_batch[key] = values 
        elif isinstance(values[0], np.ndarray):
            collated_batch[key] = np.concatenate(values, axis=0)
        elif isinstance(values[0], list):
            flattened = []
            for v in values: flattened.extend(v)
            collated_batch[key] = flattened
        else: collated_batch[key] = values
    return collated_batch

def process_image_with_mask(
    model, img_path_list, mask_path_list, idx_path, idx_dict, 
    mhr_shape_scale_dict, occ_dict
):
    # 1. 图像预处理
    img_list = []
    for img_path in img_path_list:
        img = load_image(img_path, backend="cv2", image_format="bgr")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_list.append(img)

    mini_batches = []
    real_transform = getattr(model, 'transform', None) or dummy_transform

    for i, mask_path in enumerate(mask_path_list):
        mask = cv2.imread(mask_path, 0)
        if mask.shape[:2] != img_list[i].shape[:2]:
            mask = cv2.resize(mask, (img_list[i].shape[1], img_list[i].shape[0]), interpolation=cv2.INTER_NEAREST)
        
        y, x = np.where(mask > 0)
        bbox = np.array([[np.min(x)-10, np.min(y)-10, np.max(x)+10, np.max(y)+10]]) if len(y)>0 else np.array([[0,0,10,10]])
        mini_batches.append(prepare_batch(img_list[i], real_transform, bbox, img_com_dict={}))
    
    batch = recursive_to(collate_batches(mini_batches), "cuda")
    
    # 暴力维度清洗 (Robust Dimension Cleaning)
    if 'img' in batch:
        img_tensor = batch['img']
        while img_tensor.dim() > 4:
            if img_tensor.shape[1] == 1: 
                img_tensor = img_tensor.squeeze(1)
            else:
                img_tensor = img_tensor.squeeze(0)
        
        if img_tensor.dim() == 4 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(0, 3, 1, 2)
            
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.unsqueeze(1)
            
        batch['img'] = img_tensor

    id_batch = [0] * len(img_list)
    infer_model = model.model if hasattr(model, 'model') else model

    with torch.no_grad():
        infer_model._initialize_batch(batch)
        outputs = infer_model.run_inference_batch(
            img_list, batch, inference_type="full", idx_path=idx_path, 
            idx_dict=idx_dict, mhr_shape_scale_dict=mhr_shape_scale_dict, 
            id_batch=id_batch, occ_dict=occ_dict
        )
        
        pose_output = outputs[0]
        # print("🔍 pose_output 完整键名:", list(pose_output.keys()))
        mhr_output = pose_output["mhr"]
        # print("🔍 mhr_output 完整键名:", list(mhr_output.keys()))
        
        # =========================================================
        # 动作平滑 (Bi-Mamba) & Mesh 动力学重建区
        # =========================================================
        smoother = load_smoother_model(device="cuda")
        
        if smoother is not None and "pred_pose_raw" in mhr_output:
            raw_pose = mhr_output["pred_pose_raw"] # (B, 266)
            seq_input = raw_pose.unsqueeze(0)      # (1, B, 266)
            
            # 执行约束状态空间平滑演化
            smoothed_pose = smoother(seq_input).squeeze(0) # (B, 266)
            mhr_output["pred_pose_raw"] = smoothed_pose
            
            # 严格遵循 mhr_head 动力学矩阵解码
            try:
                import roma
                from models.sam_3d_body.sam_3d_body.models.modules.geometry_utils import rot6d_to_rotmat
                from models.sam_3d_body.sam_3d_body.models.modules.mhr_utils import compact_cont_to_model_params_body, mhr_param_hand_mask
                
                head = getattr(infer_model, 'head_pose', infer_model)
                if hasattr(head, 'head_module'): head = head.head_module
                
                if hasattr(head, 'mhr_forward'):
                    # (a) 分离 6D 旋转和连续姿态
                    global_rot_6d = smoothed_pose[:, :6]
                    pred_pose_cont = smoothed_pose[:, 6:]
                    
                    # (b) 转换旋转矩阵
                    global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
                    global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)
                    global_trans = torch.zeros_like(global_rot_euler)
                    
                    # (c) 转换连续姿态并应用骨架纠偏 Mask
                    pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)
                    pred_pose_euler[:, mhr_param_hand_mask] = 0
                    pred_pose_euler[:, -3:] = 0
                    
                    # (d) 正向投影解算 3D 空间骨骼顶点
                    verts = head.mhr_forward(
                        global_trans=global_trans,
                        global_rot=global_rot_euler,
                        body_pose_params=pred_pose_euler,
                        hand_pose_params=mhr_output.get("hand"),
                        scale_params=mhr_output.get("scale"),
                        shape_params=mhr_output.get("shape"),
                        expr_params=mhr_output.get("face"),
                        do_pcblend=True,
                        return_keypoints=False,
                        return_joint_coords=False,
                        return_model_params=False,
                        return_joint_rotations=False,
                    )
                    
                    # (e) 坐标系镜像反转
                    verts[..., [1, 2]] *= -1
                    mhr_output["pred_vertices"] = verts
                    print(f"✅ [Debug] verts updated, shape={verts.shape}, mean={verts.float().mean():.4f}")
                    
                else:
                    print("⚠️ [Mamba] Head lacks mhr_forward method.")
            except Exception as e:
                print(f"❌ [Decoder Error] MHR 解算失败: {e}")
                import traceback
                traceback.print_exc()
                print("⚠️ [Fallback] 使用原始vertices，pose已平滑")

    # ==========================================
    # 3. 结果封装 (附带保存姿态参数供体外追踪使用)
    # ==========================================
    pred_vertices = mhr_output["pred_vertices"]
    # print(f"[Debug2] pred_vertices shape={pred_vertices.shape}, type={type(pred_vertices)}")
    if pred_vertices.dim() == 4: pred_vertices = pred_vertices[:, 0]
    pred_vertices = pred_vertices.detach().cpu().numpy()
    
    pose_t = mhr_output.get("pred_pose_raw", mhr_output.get("pred_pose"))
    shape_t = mhr_output.get("pred_shape_raw", mhr_output.get("pred_shape"))
    exp_t = mhr_output.get("pred_exp_raw", mhr_output.get("pred_exp"))

    # 补充提取 convert_sam3d_output_to_smpl 真正需要的字段，这些在
    # mhr_output 里本来就真实存在，之前没有被摘取出来。
    pred_cam_t_t = mhr_output.get("pred_cam_t")
    mhr_model_params_t = mhr_output.get("mhr_model_params")
    shape_params_t = mhr_output.get("shape")
    expr_params_t = mhr_output.get("face")

    full_output_list = []
    for i, v in enumerate(pred_vertices):
        out_dict = {"vertices": v, "pred_vertices": v}
        if pose_t is not None: out_dict["pose"] = pose_t[i].detach().cpu().numpy()
        if shape_t is not None: out_dict["shape"] = shape_t[i].detach().cpu().numpy()
        if exp_t is not None: out_dict["exp"] = exp_t[i].detach().cpu().numpy()

        if pred_cam_t_t is not None:
            out_dict["pred_cam_t"] = pred_cam_t_t[i].detach().cpu().numpy()
        if mhr_model_params_t is not None:
            out_dict["mhr_model_params"] = mhr_model_params_t[i].detach().cpu().numpy()
        if shape_params_t is not None:
            out_dict["shape_params"] = shape_params_t[i].detach().cpu().numpy()
        if expr_params_t is not None:
            out_dict["expr_params"] = expr_params_t[i].detach().cpu().numpy()

        full_output_list.append(out_dict)
        
    return full_output_list, [0]*len(img_list), []

def save_mesh_results(outputs, faces, save_dir, image_path, id_current):
    import trimesh
    if outputs is None: return
    vertices = outputs["vertices"]
    frame_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(save_dir, f"{frame_name}.ply")
    trimesh.Trimesh(vertices=vertices, faces=faces).export(save_path)