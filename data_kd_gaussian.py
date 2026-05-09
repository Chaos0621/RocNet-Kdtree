import argparse
import os
import numpy as np
import torch

from ROctNetmodel_32_kdtree import ROctEncoder, ROctDecoder, encode_structure, decode_structure


class KDNode:
    def __init__(self, points, axis, left=None, right=None):
        self.points = points
        self.axis = axis
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left is None and self.right is None


class KDTree:
    def __init__(self, root):
        self.root = root


def build_kdtree(points, depth=0, split_dims=3):
    if len(points) == 0:
        return None

    axis = depth % split_dims
    points = sorted(points, key=lambda x: x[axis])

    mid = len(points) // 2
    pivot = points[mid]

    left_points = []
    right_points = []
    same_points = []

    def same_point(a, b):
        return np.array_equal(a, b)

    for p in points:
        if same_point(p, pivot):
            same_points.append(p)
            continue
        if p[axis] < pivot[axis]:
            left_points.append(p)
        elif p[axis] > pivot[axis]:
            right_points.append(p)
        else:
            # Tie-break by full vector to avoid one-sided splits
            if tuple(p.tolist()) < tuple(pivot.tolist()):
                left_points.append(p)
            else:
                right_points.append(p)

    if len(left_points) == 0 or len(right_points) == 0:
        return KDNode(points=same_points + left_points + right_points, axis=axis)

    return KDNode(
        points=same_points,
        axis=axis,
        left=build_kdtree(left_points, depth + 1, split_dims=split_dims),
        right=build_kdtree(right_points, depth + 1, split_dims=split_dims),
    )


def build_kdtree_tree(points, split_dims=3):
    return KDTree(build_kdtree(points, split_dims=split_dims))


def load_gaussian_ply(ply_path):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("plyfile is required: pip install plyfile") from exc

    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    xyz = np.stack([np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])], axis=1)
    opacity = np.asarray(vertex["opacity"]).reshape(-1, 1)
    scales = np.stack(
        [np.asarray(vertex["scale_0"]), np.asarray(vertex["scale_1"]), np.asarray(vertex["scale_2"])],
        axis=1,
    )
    rots = np.stack(
        [np.asarray(vertex["rot_0"]), np.asarray(vertex["rot_1"]), np.asarray(vertex["rot_2"]), np.asarray(vertex["rot_3"])],
        axis=1,
    )
    sh_dc = np.stack(
        [np.asarray(vertex["f_dc_0"]), np.asarray(vertex["f_dc_1"]), np.asarray(vertex["f_dc_2"])],
        axis=1,
    )

    sh_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_rest_")])
    if sh_rest_names:
        sh_rest = np.stack([np.asarray(vertex[name]) for name in sh_rest_names], axis=1)
    else:
        sh_rest = np.zeros((xyz.shape[0], 0), dtype=np.float32)

    all_params = np.concatenate([xyz, opacity, scales, rots, sh_dc, sh_rest], axis=1).astype(np.float32)

    return all_params


def normalize_features(all_params, xyz_margin=0.05):
    arr = all_params.astype(np.float32)

    # Normalize xyz to roughly [-1, 1]
    xyz = arr[:, :3]
    pmin = xyz.min(axis=0)
    pmax = xyz.max(axis=0)
    center = (pmin + pmax) / 2.0
    scale = (pmax - pmin).max() / 2.0 / (1.0 - xyz_margin)
    xyz_norm = (xyz - center) / (scale + 1e-8)

    # Min-max normalize other dimensions to [-1, 1]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True, help="Path to point_cloud.ply")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--leaf-point-count", type=int, default=8)
    parser.add_argument("--xyz-margin", type=float, default=0.05)
    parser.add_argument("--save", type=str, default="model/kdtree_gaussian_checkpoint.pt")
    args = parser.parse_args()

    all_params = load_gaussian_ply(args.ply)
    all_params, stats = normalize_features(all_params, xyz_margin=args.xyz_margin)

    points = [all_params[i] for i in range(all_params.shape[0])]
    tree = build_kdtree_tree(points, split_dims=3)

    class Config:
        leaf_code_size = all_params.shape[1]
        hidden_size = args.hidden_size
        feature_size = all_params.shape[1]
        leaf_point_count = args.leaf_point_count

    encoder = ROctEncoder(Config)
    decoder = ROctDecoder(Config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, args.save)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    for step in range(args.steps + 1):
        root_code = encode_structure(tree, encoder)
        recon_loss = decode_structure(root_code, tree, decoder).sum()

        optim.zero_grad()
        recon_loss.backward()
        optim.step()

        if step % 100 == 0:
            print(f"Step {step:04d} recon_loss={recon_loss.item():.6f}")

    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "config": {
                "leaf_code_size": Config.leaf_code_size,
                "hidden_size": Config.hidden_size,
                "feature_size": Config.feature_size,
                "leaf_point_count": Config.leaf_point_count,
            },
            "norm_stats": stats,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
