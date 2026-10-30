#!/usr/bin/env python
"""Reconstruct the VGGT-Omega world point cloud and color each point by its S4VY instance id
(background = 0 -> black).  Same frames as demo/forward_video.py; the depth + camera heads come
from the same s4vy checkpoint (it bundles backbone + LoRA + decoder + geometry heads).

Outputs (under --out, per video):
  <name>_pc.mp4    orbit view of the instance-colored point cloud (rendered with pyrender-free
                   numpy depth sort; no extra deps)
  <name>_pc.ply    the full point cloud (xyz + instance-color RGB)

Usage:
  python demo/pointcloud_video.py --video clip.mp4 --out out/pc [--nframes 16]
"""
import os, sys, argparse, glob
os.environ.setdefault("SEG_NO_FRAMEPOS", "1")
import numpy as np, cv2, torch
from PIL import Image
from huggingface_hub import hf_hub_download
from vggt_omega.utils.pose_enc import encoding_to_camera

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.forward_video import build_model, load_frames, model_input_hw, PAL, refine_and_overlay

# --- point cloud from the same backbone (verbatim from vggt-omega/demo_gradio.py) -----------
def unproject(depth_map, extrinsic, intrinsic):
    depth = depth_map[..., 0]
    S, H, W = depth.shape
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    fx = intrinsic[:, 0, 0][:, None, None]; fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]; cy = intrinsic[:, 1, 2][:, None, None]
    cam = np.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1)  # S,H,W,3
    R = extrinsic[:, :3, :3].transpose(0, 2, 1)   # world-from-cam = inverse of cam-from-world
    T = extrinsic[:, :3, 3]
    world = np.einsum("sij,sHWC->sHWC", R, cam) + T[:, None, None]
    return world


def render_orbit(pc, rgb, n_frames=64, res=480):
    """Simple orbit: rotate pc around the scene center, perspective-project, painter's sort."""
    c = pc.mean(0)
    pc_c = pc - c
    scale = np.abs(pc_c).max(0)
    scale = max(scale.max() * 0.5, 1e-3)
    f = res * 1.2
    vids = []
    for i in range(n_frames):
        th, ph = 2 * np.pi * i / n_frames, np.pi / 6
        eye = scale * np.array([np.cos(th) * np.cos(ph), np.sin(th) * np.cos(ph), np.sin(ph) * 0.8]) + c
        fvec, rvec = (eye - c) / np.linalg.norm(eye - c), np.array([0, 0, 1.0])
        up = np.cross(fvec, rvec); up /= np.linalg.norm(up)
        rvec = np.cross(up, fvec)
        Rv = np.stack([rvec, up, fvec], 1)
        cam = (Rv @ pc_c.T).T + eye
        z = cam[:, 2]
        ok = z > 1e-3
        u = (cam[:, 0] / z * f + res / 2).astype(int)
        v = (cam[:, 1] / z * f + res / 2).astype(int)
        m = (u >= 0) & (u < res) & (v >= 0) & (v < res) & ok
        u, v, z, rgbm = u[m], v[m], z[m], rgb[m]
        order = np.argsort(-z)                                       # far first
        img = np.zeros((res, res, 3), np.uint8)
        img[v[order], u[order]] = rgbm[order]
        vids.append(img)
    return np.stack(vids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nframes", type=int, default=16)
    ap.add_argument("--long_side", type=int, default=512)
    ap.add_argument("--max_points", type=int, default=200000)
    args = ap.parse_args()

    imgs, origs = load_frames(args.video, args.nframes, args.long_side)   # imgs (S,3,H,W) 0-1, origs (S,Ho,Wo,3)
    S, Ho, Wo = origs.shape[:3]
    Hr, Wr = imgs.shape[-2:]
    print(f"[pc] {S} frames, model={Wr}x{Hr} orig={Wo}x{Ho}", flush=True)

    model = build_model("s4vy")
    from segomega.engine.decode import masks_to_idmap
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(imgs.cuda().unsqueeze(0), run_depth=True, run_camera=True)
    idmap = masks_to_idmap(out["objectness"][0], out["mask_logits"][0], (Hr, Wr), score_thr=0.3)   # (S,Hr,Wr)
    depth = out["depth"][0].float().cpu().numpy()[:, :, :, 0]                                     # (S,Hr,Wr)
    ext, intr = encoding_to_camera(out["pose_enc"][0].float().cpu(), (Hr, Wr))
    pc = unproject(depth, ext.numpy(), intr.numpy())                                              # (S,Hr,Wr,3) world
    print(f"[pc] world points {pc.shape}, depth range {depth[depth > 0].min():.2f}-{depth[depth > 0].max():.2f}", flush=True)

    # instance id -> colour per pixel; background 0 -> black
    flat_id = idmap.reshape(-1)
    keep = (depth.reshape(-1) > 0) & np.isfinite(pc.reshape(-1, 3).all(-1)).all(-1)
    idx = np.where(keep)[0]
    if len(idx) > args.max_points:
        idx = idx[np.random.RandomState(0).choice(len(idx), args.max_points, replace=False)]
    cols = np.zeros((len(idx), 3), np.uint8)
    for k in np.unique(flat_id[idx]):
        if k == 0:
            continue
        sel = idx[flat_id[idx] == k]
        cols[sel] = PAL[int(k) % 1024]
    pts = pc.reshape(-1, 3)[idx]
    np.savez_compressed(os.path.join(args.out, "pc.npz"), xyz=pts, rgb=cols, depth=depth, idmap=idmap)

    # colored single-view png (front orbit frame) + orbit mp4
    os.makedirs(args.out, exist_ok=True)
    name = os.splitext(os.path.basename(args.video))[0]
    orbit = render_orbit(pts, cols)
    Image.fromarray(orbit[0]).save(os.path.join(args.out, f"{name}_pc_front.png"))
    vw = cv2.VideoWriter(os.path.join(args.out, f"{name}_pc.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (480, 480))
    for fr in orbit:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()

    # PLY (xyz + rgb)
    def write_ply(path, xyz, rgb):
        n = len(xyz)
        with open(path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\n")
            f.write(f"element vertex {n}\n".encode())
            f.write(b"property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            d = np.zeros(n, dtype=np.float32)
            for i, a in enumerate(xyz):
                d[i * 6:i * 6 + 3] = a
            r = np.zeros(n, dtype=np.uint8)
            for i, c in enumerate(rgb):
                r[i * 3:i * 3 + 3] = c
            f.write(np.ascontiguousarray(d).tobytes())
            f.write(np.ascontiguousarray(r).tobytes())
    write_ply(os.path.join(args.out, f"{name}_pc.ply"), pts, cols)
    print(f"[pc] wrote {name}_pc.mp4/.ply + front png -> {args.out}  ({len(pts)} points)", flush=True)


if __name__ == "__main__":
    main()
