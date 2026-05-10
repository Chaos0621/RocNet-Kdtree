import numpy as np
import torch
import os
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from ROctNetmodel_32_kdtree import ROctEncoder, ROctDecoder, encode_structure, decode_structure

class KDNode:
    def __init__(self, points, axis, left=None, right=None):
        # Internal nodes keep structure only; leaves keep gaussian points.
        self.points = points
        self.axis = axis
        self.left = left
        self.right = right
    
    def is_leaf(self):
        return self.left is None and self.right is None


class KDTree:
    def __init__(self, root):
        self.root = root


def _point_key(p):
    if isinstance(p, np.ndarray):
        return tuple(p.tolist())
    return tuple(p)

def build_kdtree(points, depth=0, split_dims=3, max_leaf_size=None):

    if len(points) == 0:
        return None
    if max_leaf_size is not None and len(points) <= max_leaf_size:
        return KDNode(points=points, axis=depth % split_dims)

    axis = depth % split_dims

    points = sorted(points, key=lambda x: (x[axis], _point_key(x)))

    mid = len(points)//2
    left_points = points[:mid]
    right_points = points[mid:]

    return KDNode(
        points=[],
        axis=axis,
        left=build_kdtree(left_points, depth+1, split_dims=split_dims, max_leaf_size=max_leaf_size),
        right=build_kdtree(right_points, depth+1, split_dims=split_dims, max_leaf_size=max_leaf_size)
    )

def build_kdtree_tree(points, split_dims=3, max_leaf_size=None):
    return KDTree(build_kdtree(points, split_dims=split_dims, max_leaf_size=max_leaf_size))


def serialize_kdtree(node):
    if node is None:
        return None
    points_sorted = tuple(sorted([_point_key(p) for p in node.points]))
    return (
        node.axis,
        points_sorted,
        serialize_kdtree(node.left),
        serialize_kdtree(node.right),
    )

def cache_leaf_tensors(node, device):
    if node is None:
        return
    if node.is_leaf():
        if not torch.is_tensor(node.points):
            node.points_tensor = torch.tensor(node.points, dtype=torch.float32, device=device)
        else:
            node.points_tensor = node.points.to(device)
        return
    cache_leaf_tensors(node.left, device)
    cache_leaf_tensors(node.right, device)

def load_gaussian_ply(ply_path, compute_colors=True):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("plyfile is required: pip install plyfile") from exc
    try:
        from hotdog.extract_gaussians import SphericalHarmonicsCalculator
    except Exception:
        SphericalHarmonicsCalculator = None

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    names = set(vertex.data.dtype.names or [])

    def _get(name, default=None):
        if name in names:
            return np.asarray(vertex[name])
        return default

    xyz = np.stack([_get("x"), _get("y"), _get("z")], axis=1)

    opacity_raw = _get("opacity")
    if opacity_raw is None:
        opacity_raw = _get("alpha")
    if opacity_raw is None:
        opacity = np.ones((xyz.shape[0], 1), dtype=np.float32)
    else:
        opacity = np.asarray(opacity_raw, dtype=np.float32).reshape(-1, 1)
        if opacity.size > 0 and opacity.max() > 1.0:
            opacity = opacity / 255.0

    scale_0 = _get("scale_0")
    scale_1 = _get("scale_1")
    scale_2 = _get("scale_2")
    if scale_0 is None or scale_1 is None or scale_2 is None:
        scales = np.ones((xyz.shape[0], 3), dtype=np.float32)
    else:
        scales = np.stack([scale_0, scale_1, scale_2], axis=1)

    rot_0 = _get("rot_0")
    rot_1 = _get("rot_1")
    rot_2 = _get("rot_2")
    rot_3 = _get("rot_3")
    if rot_0 is None or rot_1 is None or rot_2 is None or rot_3 is None:
        rots = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (xyz.shape[0], 1))
    else:
        rots = np.stack([rot_0, rot_1, rot_2, rot_3], axis=1)

    f_dc_0 = _get("f_dc_0")
    f_dc_1 = _get("f_dc_1")
    f_dc_2 = _get("f_dc_2")
    if f_dc_0 is None or f_dc_1 is None or f_dc_2 is None:
        sh_dc = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    else:
        sh_dc = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)

    sh_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_rest_")])
    if sh_rest_names:
        sh_rest = np.stack([np.asarray(vertex[name]) for name in sh_rest_names], axis=1)
    else:
        sh_rest = np.zeros((xyz.shape[0], 0), dtype=np.float32)

    sh_coeffs = np.concatenate([sh_dc, sh_rest], axis=1).astype(np.float32)
    if sh_coeffs.shape[0] != xyz.shape[0]:
        raise ValueError(f"SH coeff count {sh_coeffs.shape[0]} != xyz count {xyz.shape[0]}")

    red = _get("red")
    green = _get("green")
    blue = _get("blue")
    has_vertex_rgb = red is not None and green is not None and blue is not None
    if has_vertex_rgb:
        colors = np.stack([red, green, blue], axis=1).astype(np.float32)
        if colors.size > 0 and colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = None

    # Compute RGB colors from SH + normals if vertex RGB is unavailable.
    has_normals = all(n in vertex.data.dtype.names for n in ["nx", "ny", "nz"])
    if colors is None and compute_colors and SphericalHarmonicsCalculator is not None and has_normals:
        try:
            normals = np.stack([np.asarray(vertex["nx"]), np.asarray(vertex["ny"]), np.asarray(vertex["nz"])], axis=1)
            normals = np.asarray(normals, dtype=np.float32).reshape(-1)
            n = xyz.shape[0]
            print(f"[DEBUG] xyz: {xyz.shape} normals(raw_flat): {normals.shape} sh_coeffs: {sh_coeffs.shape}", flush=True)
            if normals.size % 3 == 0:
                if normals.size == n * 3:
                    normals = normals.reshape(n, 3)
                elif normals.size % (n * 3) == 0:
                    factor = normals.size // (n * 3)
                    normals = normals.reshape(n, factor, 3).mean(axis=1)
                else:
                    normals = None
            else:
                normals = None
            if normals is None or normals.shape[0] != n or normals.shape[1] != 3:
                normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n, 1))
            print(f"[DEBUG] normals(fixed): {normals.shape}", flush=True)
            colors = SphericalHarmonicsCalculator.recover_color_from_sh(normals, sh_coeffs, sh_degree=3)
        except Exception:
            colors = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    elif colors is None and compute_colors and SphericalHarmonicsCalculator is not None:
        # Fallback: use view direction along +Z if normals are missing
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (xyz.shape[0], 1))
        colors = SphericalHarmonicsCalculator.recover_color_from_sh(normals, sh_coeffs, sh_degree=3)
    elif colors is None:
        colors = np.zeros((xyz.shape[0], 3), dtype=np.float32)

    # Concatenate full gaussian attributes per point:
    # [xyz(3), color(3), opacity(1), scale(3), rotation(4), sh_coeffs(48)]
    all_params = np.concatenate([xyz, colors, opacity, scales, rots, sh_coeffs], axis=1).astype(np.float32)
    return all_params


def normalize_features(all_params, xyz_margin=0.05):
    arr = all_params.astype(np.float32)

    xyz = arr[:, :3]
    pmin = xyz.min(axis=0)
    pmax = xyz.max(axis=0)
    center = (pmin + pmax) / 2.0
    scale = (pmax - pmin).max() / 2.0 / (1.0 - xyz_margin)
    xyz_norm = (xyz - center) / (scale + 1e-8)

    rest = arr[:, 3:]
    if rest.shape[1] > 0:
        rmin = rest.min(axis=0)
        rmax = rest.max(axis=0)
        rscale = rmax - rmin
        rscale[rscale == 0] = 1.0
        rest_norm = (rest - rmin) / rscale
        rest_norm = rest_norm * 2.0 - 1.0
    else:
        rmin = rmax = rscale = np.zeros((0,), dtype=np.float32)
        rest_norm = rest

    normalized = np.concatenate([xyz_norm, rest_norm], axis=1)
    stats = {
        "xyz_center": center,
        "xyz_scale": scale,
        "rest_min": rmin,
        "rest_scale": rscale,
    }
    return normalized, stats


if __name__ == "__main__":
    data = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, default="/data/23010572/roc/RocNet/hotdog/hotdog/gaussians_iter30000.ply", help="Hotdog point_cloud.ply path")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--keep-list", type=int, nargs="+", default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--leaf-point-count", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--xyz-margin", type=float, default=0.05)
    parser.add_argument("--save", type=str, default=data+".pt")
    parser.add_argument("--no-color", action="store_true", help="Disable SH->RGB color computation")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--loss-reduction", type=str, default="mean", choices=["mean", "sum"], help="Leaf reconstruction loss reduction")
    parser.add_argument("--tb-logdir", type=str, default="runs", help="TensorBoard log directory (relative to RocNet dir if not absolute)")
    args = parser.parse_args()

    def print_tree(node, depth=0):
        if node is None:
            return
        indent = "  " * depth
        print(f"{indent}axis={node.axis} points={node.points}")
        print_tree(node.left, depth + 1)
        print_tree(node.right, depth + 1)

    def build_xyz_64():
        points = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    points.append((float(x), float(y), float(z)))
        return points
    
    def normalize_points(points, low=-1.0, high=1.0):
        arr = np.array(points, dtype=np.float32)
        minv = arr.min(axis=0)
        maxv = arr.max(axis=0)
        scale = maxv - minv
        scale[scale == 0] = 1.0
        arr = (arr - minv) / scale
        arr = arr * (high - low) + low
        return [tuple(p.tolist()) for p in arr]

    norm_stats = None
    if args.ply:
        all_params = load_gaussian_ply(args.ply, compute_colors=not args.no_color)
        all_params, norm_stats = normalize_features(all_params, xyz_margin=args.xyz_margin)
        points = [all_params[i] for i in range(all_params.shape[0])]
        tree = build_kdtree_tree(points, split_dims=3, max_leaf_size=args.leaf_point_count)
        feature_size = all_params.shape[1]
        sh_dim = feature_size - (3 + 3 + 1 + 3 + 4)
        loss_weights = (
            [10.0] * 3 +
            [1.0] * 3 +
            [1.0] * 1 +
            [1.0] * 3 +
            [1.0] * 4 +
            [0.1] * max(sh_dim, 0)
        )
    else:
        points = normalize_points(build_xyz_64())
        tree = build_kdtree_tree(points, split_dims=3, max_leaf_size=args.leaf_point_count)
        feature_size = 3
        loss_weights = None

    class SimpleConfig:
        leaf_code_size = feature_size
        hidden_size = args.hidden_size
        feature_size = feature_size
        leaf_point_count = args.leaf_point_count
        loss_weights = loss_weights

    if not args.ply:
        print("Original KD-Tree")
        print_tree(tree.root)
    else:
        print(f"Loaded gaussian points: {len(points)}")

    encoder = ROctEncoder(SimpleConfig)
    decoder = ROctDecoder(SimpleConfig)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Check your environment/driver.")
    device = torch.device(args.device if args.device in ["cuda", "cpu"] else "cuda")
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print(f"Using device: {device}")
    cache_leaf_tensors(tree.root, device)

    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, "model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    tb_logdir = args.tb_logdir if os.path.isabs(args.tb_logdir) else os.path.join(base_dir, args.tb_logdir)
    writer = SummaryWriter(log_dir=tb_logdir)
    print(f"TensorBoard logdir: {tb_logdir}")
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.5)


    for step in range(args.steps + 1):
        root_code = encode_structure(tree, encoder)
        recon_loss = decode_structure(
            root_code,
            tree,
            decoder,
            keep_list=args.keep_list,
            temperature=args.temperature,
            reduction=args.loss_reduction,
        ).mean()

        optim.zero_grad()
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 
            max_norm=1.0  # 梯度裁剪，防止震荡
        )
        optim.step()
        if step % 200 == 0 and step > 0:
            scheduler.step()

        if step % 5 == 0:
            print(f"Step {step:03d} recon_loss={recon_loss.item():.6f}")
        if step % 10 == 0:
            writer.add_scalar("loss/train", recon_loss.item(), step)
        if  step % 400 == 0 and step > 0:
            checkpoint_name = os.path.splitext(os.path.basename(args.save))[0]
            checkpoint_ext = os.path.splitext(os.path.basename(args.save))[1] or ".pt"
            step_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{checkpoint_name}_step_{step:06d}{checkpoint_ext}",
            )
            torch.save(
                {
                    "encoder_state": encoder.state_dict(),
                    "decoder_state": decoder.state_dict(),
                    "config": {
                        "leaf_code_size": SimpleConfig.leaf_code_size,
                        "hidden_size": SimpleConfig.hidden_size,
                        "feature_size": SimpleConfig.feature_size,
                        "leaf_point_count": SimpleConfig.leaf_point_count,
                        "loss_reduction": args.loss_reduction,
                    },
                    "norm_stats": norm_stats,
                },
                step_checkpoint_path,
            )

    print(f"Final reconstruction loss: {recon_loss.item():.6f}")
    writer.close()

    def decode_points(node, feature):
        if node.is_leaf():
            out, _score = decoder.leafDecoder(feature)
            out = out.detach().cpu().numpy().squeeze()
            pts = out.reshape(SimpleConfig.leaf_point_count, SimpleConfig.feature_size)
            target_n = len(node.points)
            pts = pts[:target_n]
            return [tuple(p.tolist()) for p in pts]
        left_feat, right_feat = decoder.NodeDecoder(feature)
        pts = decode_points(node.left, left_feat)
        pts.extend(decode_points(node.right, right_feat))
        return pts

    root_feature = decoder.sampleDecoder(root_code)
    strict_structure = not args.ply or len(points) <= 4096

    if strict_structure:
        def chamfer_loss(pred, gt):
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
            dists = torch.sum((pred.unsqueeze(2) - gt.unsqueeze(1)) ** 2, dim=-1)
            loss1 = torch.min(dists, dim=2)[0].mean()
            loss2 = torch.min(dists, dim=1)[0].mean()
            return loss1 + loss2

        def eval_leaf_errors(node, feat):
            if node.is_leaf():
                pred, _score = decoder.leafDecoder(feat)
                pred = pred.view(SimpleConfig.leaf_point_count, SimpleConfig.feature_size)
                gt = torch.tensor(node.points, dtype=pred.dtype, device=pred.device)
                return chamfer_loss(pred, gt)
            left_feat, right_feat = decoder.NodeDecoder(feat)
            return eval_leaf_errors(node.left, left_feat) + eval_leaf_errors(node.right, right_feat)

        total_chamfer = eval_leaf_errors(tree.root, root_feature)
        print("Structure match after encode/decode (strict): True")
        print(f"Total Chamfer (strict structure): {total_chamfer.item():.6f}")
    else:
        print("Skip structure check for large dataset.")
    
