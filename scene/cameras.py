#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#
import typing as t
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor

from utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrixShift,
)
from PIL import Image
from jaxtyping import Float


class Camera(nn.Module):

    def __init__(
        self,
        *,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        fx: float,  # Reason: 添加水平焦距，用于计算主点偏移
        fy: float,  # Reason: 添加垂直焦距，用于计算主点偏移
        cx: float,  # Reason: 添加主点横坐标，用于偏移视锥体
        cy: float,  # Reason: 添加主点纵坐标，用于偏移视锥体
        image_width: int,
        image_height: int,
        image_name: str,
        image_path: t.Optional[Path] = None,
        mask_path: Path,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_width = image_width
        self.image_height = image_height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrixShift(
                znear=self.znear,
                zfar=self.zfar,
                focal_x=self.fx,
                focal_y=self.fy,
                cx=self.cx,
                cy=self.cy,
                width=self.image_width,
                height=self.image_height,
                fovX=self.FoVx,
                fovY=self.FoVy,
            )
            .transpose(0, 1)
            .cuda()
        )  # Reason: 调整为列主序
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def extra_repr(self) -> str:
        return (
            f"colmap_id: {self.colmap_id}, "
            f"image_name: {self.image_name}, "
            f"image_path: {self.image_path}, "
            f"mask_path: {self.mask_path}, "
            f"image_width: {self.image_width}, "
            f"image_height: {self.image_height}, "
            f"fx: {self.fx}, "
            f"fy: {self.fy}, "
            f"cx: {self.cx}, "
            f"cy: {self.cy}"
        )

    @property
    def original_image(self) -> Float[Tensor, "3 H W"]:
        image_path = self.image_path
        assert image_path.exists()
        with Image.open(image_path) as img:
            img = img.resize(
                (self.image_width, self.image_height),
                resample=Image.Resampling.BILINEAR,
            )
            image = np.array(img).astype(np.float32, copy=False) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1).contiguous().cuda()

    @property
    def mask(self) -> Float[Tensor, "H W"]:
        mask_path = self.mask_path
        assert mask_path.exists()
        with Image.open(mask_path) as img:
            img = img.convert("L")
            img = img.resize(
                (self.image_width, self.image_height), resample=Image.Resampling.NEAREST
            )
            mask = np.array(img).astype(np.float32, copy=False)
        return torch.from_numpy(mask).cuda()


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
