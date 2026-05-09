#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取高斯Splatting属性并封装为Tensor类。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

try:
    from hotdog.extract_gaussians import GaussianSplatExtractor
except Exception:  # pragma: no cover - only for optional import behavior
    GaussianSplatExtractor = None


@dataclass
class GaussianSplatTensors:
    """高斯点属性的Tensor封装"""

    xyz: torch.Tensor
    colors: Optional[torch.Tensor]
    opacity: torch.Tensor
    scales: torch.Tensor
    rotation: torch.Tensor
    sh_coeffs: torch.Tensor
    f_dc: torch.Tensor
    f_rest: torch.Tensor

    @classmethod
    def from_gaussians(
        cls,
        gaussians: Dict[str, "object"],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "GaussianSplatTensors":
        """从extractor输出的gaussians字典构建Tensor类"""

        def to_tensor(value):
            t = torch.as_tensor(value, dtype=dtype)
            if device is not None:
                t = t.to(device)
            return t

        colors = gaussians.get("colors")
        return cls(
            xyz=to_tensor(gaussians["xyz"]),
            colors=to_tensor(colors) if colors is not None else None,
            opacity=to_tensor(gaussians["opacity"]),
            scales=to_tensor(gaussians["scales"]),
            rotation=to_tensor(gaussians["rotation"]),
            sh_coeffs=to_tensor(gaussians["sh_coeffs"]),
            f_dc=to_tensor(gaussians["f_dc"]),
            f_rest=to_tensor(gaussians["f_rest"]),
        )

    @classmethod
    def from_root(
        cls,
        root_path: str,
        iteration: int = 30000,
        compute_colors: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "GaussianSplatTensors":
        """从hotdog根目录读取PLY并构建Tensor类"""
        if GaussianSplatExtractor is None:
            raise ImportError(
                "无法导入GaussianSplatExtractor，请确认hotdog/extract_gaussians.py可用"
            )

        extractor = GaussianSplatExtractor(root_path, iteration)
        gaussians = extractor.extract_from_ply(compute_colors=compute_colors)
        return cls.from_gaussians(gaussians, device=device, dtype=dtype)

    def as_dict(self) -> Dict[str, Optional[torch.Tensor]]:
        """导出为Tensor字典"""
        return {
            "xyz": self.xyz,
            "colors": self.colors,
            "opacity": self.opacity,
            "scales": self.scales,
            "rotation": self.rotation,
            "sh_coeffs": self.sh_coeffs,
            "f_dc": self.f_dc,
            "f_rest": self.f_rest,
        }

    def to(self, device: torch.device) -> "GaussianSplatTensors":
        """返回移动到指定设备的新实例"""
        return GaussianSplatTensors(
            xyz=self.xyz.to(device),
            colors=self.colors.to(device) if self.colors is not None else None,
            opacity=self.opacity.to(device),
            scales=self.scales.to(device),
            rotation=self.rotation.to(device),
            sh_coeffs=self.sh_coeffs.to(device),
            f_dc=self.f_dc.to(device),
            f_rest=self.f_rest.to(device),
        )


if __name__ == "__main__":
    # 简单自测用法
    tensors = GaussianSplatTensors.from_root(
        "/data/23010572/roc/RocNet/hotdog/hotdog",
        iteration=30000,
        compute_colors=True,
    )
    print({k: (v.shape if v is not None else None) for k, v in tensors.as_dict().items()})
