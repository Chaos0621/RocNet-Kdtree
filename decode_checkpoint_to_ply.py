import argparse
import os

import numpy as np
import torch

from ROctNetmodel_32_kdtree import ROctEncoder, ROctDecoder, encode_structure
from data_kd import build_kdtree_tree, load_gaussian_ply, normalize_features


class Config:
    leaf_code_size = 17
    hidden_size = 128
    feature_size = 17
    leaf_point_count = 16
    loss_weights = None


def cache_node_tensors(node, device):
    if node is None:
        return
    if not torch.is_tensor(node.points):
        pts = np.asarray(node.points, dtype=np.float32)
        node.points_tensor = torch.as_tensor(pts, dtype=torch.float32, device=device)
    else:
        node.points_tensor = node.points.to(device)
    cache_node_tensors(node.left, device)
    cache_node_tensors(node.right, device)


def collect_decoded_features(node, decoder, feature, config):
    if node.is_leaf():
        out = decoder.leafDecoder(feature).detach().cpu().numpy().squeeze()
        pts = out.reshape(config.leaf_point_count, config.feature_size)
        return pts[:len(node.points)]
    _self_feat, left_feat, right_feat = decoder.NodeDecoder(feature)
    left = collect_decoded_features(node.left, decoder, left_feat, config)
    right = collect_decoded_features(node.right, decoder, right_feat, config)
    return np.concatenate([left, right], axis=0)


def collect_internal_original_features(node):
    if node is None or node.is_leaf():
        return []
    points = list(node.points)
    points.extend(collect_internal_original_features(node.left))
    points.extend(collect_internal_original_features(node.right))
    return points


def inverse_normalize_features(normalized, stats):
    normalized = np.asarray(normalized, dtype=np.float32)
    xyz = normalized[:, :3] * (float(stats["xyz_scale"]) + 1e-8) + np.asarray(stats["xyz_center"], dtype=np.float32)

    rest = normalized[:, 3:]
    if rest.shape[1] == 0:
        return xyz

    rest_min = np.asarray(stats["rest_min"], dtype=np.float32)
    rest_scale = np.asarray(stats["rest_scale"], dtype=np.float32)
    rest = (rest + 1.0) / 2.0
    rest = rest * rest_scale + rest_min
    return np.concatenate([xyz, rest], axis=1).astype(np.float32)


def write_gaussian_ply(path, features):
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:
        raise ImportError("plyfile is required: pip install plyfile") from exc

    features = np.asarray(features, dtype=np.float32)
    if features.shape[1] < 17:
        raise ValueError(f"Expected at least 17 decoded features, got {features.shape[1]}")

    xyz = features[:, 0:3]
    colors = np.clip(features[:, 3:6], 0.0, 1.0)
    opacity = features[:, 6]
    scales = features[:, 7:10]
    rots = features[:, 10:14]
    sh = features[:, 14:]

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    dtype.extend([(f"f_dc_{i}", "f4") for i in range(min(3, sh.shape[1]))])
    dtype.extend([(f"f_rest_{i}", "f4") for i in range(max(sh.shape[1] - 3, 0))])

    vertex = np.zeros(features.shape[0], dtype=dtype)
    vertex["x"] = xyz[:, 0]
    vertex["y"] = xyz[:, 1]
    vertex["z"] = xyz[:, 2]
    vertex["nz"] = 1.0
    vertex["red"] = (colors[:, 0] * 255.0).astype(np.uint8)
    vertex["green"] = (colors[:, 1] * 255.0).astype(np.uint8)
    vertex["blue"] = (colors[:, 2] * 255.0).astype(np.uint8)
    vertex["alpha"] = (np.clip(opacity, 0.0, 1.0) * 255.0).astype(np.uint8)
    vertex["opacity"] = opacity
    vertex["scale_0"] = scales[:, 0]
    vertex["scale_1"] = scales[:, 1]
    vertex["scale_2"] = scales[:, 2]
    vertex["rot_0"] = rots[:, 0]
    vertex["rot_1"] = rots[:, 1]
    vertex["rot_2"] = rots[:, 2]
    vertex["rot_3"] = rots[:, 3]

    for i in range(min(3, sh.shape[1])):
        vertex[f"f_dc_{i}"] = sh[:, i]
    for i in range(max(sh.shape[1] - 3, 0)):
        vertex[f"f_rest_{i}"] = sh[:, i + 3]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Decode a KDTree ROct checkpoint back to a gaussian PLY")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(base_dir, "model", "2026-04-29 01:52:23.pt"),
        help="Checkpoint with encoder_state/decoder_state/config",
    )
    parser.add_argument(
        "--input-ply",
        type=str,
        default=os.path.join(base_dir, "hotdog", "hotdog", "gaussians_iter30000.ply"),
        help="The same PLY used to build the KDTree during training",
    )
    parser.add_argument("--output", type=str, default=os.path.join(base_dir, "decoded_checkpoint.ply"))
    parser.add_argument("--xyz-margin", type=float, default=0.05)
    parser.add_argument("--no-color", action="store_true", help="Match training runs that used --no-color")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--internal-points",
        choices=["original", "drop"],
        default="original",
        help="Internal KDTree pivot points are not trained by the current decoder loss. "
             "'original' appends their original values to keep the input point count; 'drop' writes pure leaf decoder output.",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    Config.leaf_code_size = cfg.get("leaf_code_size", Config.leaf_code_size)
    Config.hidden_size = cfg.get("hidden_size", Config.hidden_size)
    Config.feature_size = cfg.get("feature_size", Config.feature_size)
    Config.leaf_point_count = cfg.get("leaf_point_count", Config.leaf_point_count)

    if Config.feature_size < 17:
        raise ValueError(f"This script expects a gaussian checkpoint. Got feature_size={Config.feature_size}")

    raw_features = load_gaussian_ply(args.input_ply, compute_colors=not args.no_color)
    normalized, computed_stats = normalize_features(raw_features, xyz_margin=args.xyz_margin)
    stats = ckpt.get("norm_stats") or computed_stats
    if normalized.shape[1] != Config.feature_size:
        raise ValueError(
            f"Input PLY feature size {normalized.shape[1]} does not match checkpoint feature_size={Config.feature_size}"
        )

    points = [normalized[i] for i in range(normalized.shape[0])]
    tree = build_kdtree_tree(points, split_dims=3, max_leaf_size=Config.leaf_point_count)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    device = torch.device(args.device)

    encoder = ROctEncoder(Config).to(device)
    decoder = ROctDecoder(Config).to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    encoder.eval()
    decoder.eval()
    cache_node_tensors(tree.root, device)

    with torch.no_grad():
        root_code = encode_structure(tree, encoder)
        root_feature = decoder.sampleDecoder(root_code)
        decoded_normalized = collect_decoded_features(tree.root, decoder, root_feature, Config)

    internal_count = 0
    if args.internal_points == "original":
        internal_points = collect_internal_original_features(tree.root)
        internal_count = len(internal_points)
        if internal_points:
            decoded_normalized = np.concatenate(
                [decoded_normalized, np.asarray(internal_points, dtype=np.float32)],
                axis=0,
            )

    decoded_features = inverse_normalize_features(decoded_normalized, stats)
    write_gaussian_ply(args.output, decoded_features)
    print(f"Decoded points: {decoded_features.shape[0]}")
    if internal_count:
        print(f"Included original internal KDTree points: {internal_count}")
    print(f"Feature size: {decoded_features.shape[1]}")
    print(f"Saved decoded PLY: {args.output}")


if __name__ == "__main__":
    main()
