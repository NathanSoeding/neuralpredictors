'''
code is adjusted from the Wang et al 2025 "Foundation model of neural activity predicts response to new stimulus types" implementation
Specifically
* https://github.com/cajal/fnn/blob/main/fnn/model/pixels.py
* https://github.com/cajal/fnn/blob/main/fnn/model/perspectives.py
'''
import torch
from torch import nn

def angles_to_rmat3d(angles):
    """
    Convert batches of Euler angles (x, y, z) to 3D rotation matrices.

    Args:
        angles (torch.Tensor): Tensor of shape (N, 3), angles in radians.

    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3), rotation matrices (Rz * Ry * Rx).
    """
    x, y, z = torch.unbind(angles, axis=-1)
    N = len(x)

    A = torch.eye(3, device=x.device).repeat(N, 1, 1)
    B = torch.eye(3, device=x.device).repeat(N, 1, 1)
    C = torch.eye(3, device=x.device).repeat(N, 1, 1)

    cos_z = torch.cos(z)
    sin_z = torch.sin(z)

    A[:, 0, 0] = cos_z
    A[:, 0, 1] = -sin_z
    A[:, 1, 0] = sin_z
    A[:, 1, 1] = cos_z

    cos_y = torch.cos(y)
    sin_y = torch.sin(y)

    B[:, 0, 0] = cos_y
    B[:, 0, 2] = sin_y
    B[:, 2, 0] = -sin_y
    B[:, 2, 2] = cos_y

    cos_x = torch.cos(x)
    sin_x = torch.sin(x)

    C[:, 1, 1] = cos_x
    C[:, 1, 2] = -sin_x
    C[:, 2, 1] = sin_x
    C[:, 2, 2] = cos_x

    return A @ B @ C


class PixelTransform(nn.Module):
    """
    Nonlinear pixel intensity transform with learnable power, scale, and offset.
    """
    def __init__(self, max_power=1, init_scale=1, init_offset=0, eps=1e-5):
        super().__init__()

        self.max_power = max_power
        self.eps = eps

        self.logit = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.full([1], init_scale, dtype=torch.float32))
        self.offset = nn.Parameter(torch.full([1], init_offset, dtype=torch.float32))

    @property
    def power(self):
        return self.logit.sigmoid() * self.max_power

    def forward(self, pixels):
        return pixels.add(self.eps).pow(self.power).mul(self.scale).add(self.offset)
    

class Retina(nn.Module):
    """
    Models a retina that maps pupil centers to 3D rays via an MLP.

    Args:
        degree (float): Field of view in degrees.
        height, width (int): Retina grid resolution.
        dim_in, dim_out (int): Input/output dimensions for MLP.
        mlp_features (int): Hidden feature size.
        mlp_layers (int): Number of MLP layers.
        max_angle (float): Maximum rotation angle in degrees.
    """
    def __init__(
        self,
        degree=50,
        height=36,
        width=64,
        dim_in=2,
        dim_out=2,
        mlp_features=16,
        mlp_layers=3,
        max_angle=30,
    ):
        super().__init__()

        grid = self.create_grid(height, width, degree)
        self.register_buffer("grid", grid)
        self.max_angle = max_angle / torch.pi * 180

        layers = []

        features = [dim_in] + [mlp_features] * mlp_layers + [dim_out]
        non_linearities = [nn.GELU()] * mlp_layers + [None]

        for in_features, out_features, nonlinear in zip(features[:-1], features[1:], non_linearities):
            linear = nn.Linear(in_features, out_features)
            linear = nn.utils.parametrizations.weight_norm(linear)
            nn.init.zeros_(linear.bias)
            layers.append(linear)

            if nonlinear is not None:
                layers.append(nonlinear)

        self.mlp = nn.Sequential(*layers)

    def create_grid(self, height, width, degree):
        # Create isotropic grid of retina
        x_axis = torch.linspace(-1, 1, width)
        y_axis = torch.linspace(-1, 1, height) * height / width
        scale = (width - 1) / width

        x, y = torch.meshgrid(
            x_axis * scale,
            y_axis * scale,
            indexing="xy",
        )

        # Convert to grid of 3D rays corresponding to retina pixels
        radians = degree / 180 * torch.pi

        r = torch.sqrt(x.pow(2) + y.pow(2)).mul(radians).clip(0, torch.pi / 2)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)

        theta = torch.atan2(y, x)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        ray_grid = [
            sin_r * cos_theta,
            sin_r * sin_theta,
            cos_r,
        ]

        return torch.stack(ray_grid, dim=-1)

    def rotate_retina(self, rmat):
        return torch.einsum("N C D , H W D -> N H W C", rmat, self.grid)

    # Take pupil center to return rotated grid of retina rays
    def rays(self, pupil_center):
        angles = self.mlp(pupil_center)
        angles = torch.clip(angles, -self.max_angle, self.max_angle)

        pad_zeros = torch.zeros((angles.shape[0], 1), device=angles.device)
        angles = torch.concat([angles, pad_zeros], axis=1)

        rmat = angles_to_rmat3d(angles)
        rays = self.rotate_retina(rmat)

        return rays


class Monitor(nn.Module):
    """
    Models a monitor in 3D space with optimizable position and orientation.

    Provides projection of retinal rays onto the monitor plane and sampling of images.
    """
    def __init__(
        self,
        init_center_x=0,
        init_center_y=0,
        init_center_z=0.5,
        init_center_std=0.05,
        init_angle_x=0,
        init_angle_y=0,
        init_angle_z=0,
        init_angle_std=0.05,
        eps=1e-5,
    ):
        super().__init__()

        center = [
            init_center_x,
            init_center_y,
            init_center_z,
        ]
        self.center = nn.Parameter(torch.tensor(center, dtype=torch.float32))

        angle = [
            init_angle_x,
            init_angle_y,
            init_angle_z,
        ]
        self.angle = nn.Parameter(torch.tensor(angle, dtype=torch.float32))

        self.center_std = nn.Parameter(torch.tensor(init_center_std, dtype=torch.float32))
        self.angle_std = nn.Parameter(torch.tensor(init_angle_std, dtype=torch.float32))
        self.eps = float(eps)

    # Optimize position of monitor
    def position(self, batch_size):
        center = self.center.repeat(batch_size, 1)
        angle = self.angle.repeat(batch_size, 1)

        if self.training:
            center = center + torch.randn(batch_size, 3, device=center.device) * self.center_std
            angle = angle + torch.randn(batch_size, 3, device=angle.device) * self.angle_std

        x, y, z = angles_to_rmat3d(angle).unbind(2)

        return center, x, y, z

    # Project rays onto monitor coordinates
    def project(self, rays):
        center, x, y, z = self.position(len(rays))

        a = torch.einsum("N D , N D -> N", z, center)[:, None, None]
        b = torch.einsum("N D , N H W D -> N H W", z, rays).clip(self.eps)

        c = torch.einsum("N H W , N H W D -> N H W D", a / b, rays)
        d = c - center[:, None, None, :]

        proj = [
            torch.einsum("N H W D , N D -> N H W", d, x),
            torch.einsum("N H W D , N D -> N H W", d, y),
        ]
        return torch.stack(proj, dim=3)

    # Samples values in img at positions given by grid
    def sample_screen(self, img, grid):
        _, _, H_in, W_in = img.shape
        grid_x, grid_y = grid.unbind(dim=3)

        grid_y = grid_y * W_in / H_in
        scale = W_in / (W_in - 1)

        _, H_out, W_out, _ = grid.shape
        grid = [
            grid_x * scale * (W_out - 1) / W_out,
            grid_y * scale * (H_out - 1) / H_out,
        ]

        return nn.functional.grid_sample(
            input=img,
            grid=torch.stack(grid, dim=3),
            mode="bilinear",
            align_corners=False,
        )


# Combines Retina and Monitor


class SinglePerspective(nn.Module):
    """
    Combines Retina, Monitor, and PixelTransform to generate a single visual perspective.
    """
    def __init__(self, retina, monitor, pixel_transform, static_power=1.7):
        super().__init__()

        self.retina = retina
        self.monitor = monitor
        self.pixel_transform = pixel_transform
        self.static_power = static_power

    def forward(self, img, pupil_center):
        rays = self.retina.rays(pupil_center)
        grid = self.monitor.project(rays)

        pixels = img

        img = (pixels[:, None, 0, :, :] / 255.0).pow(self.static_power)
        behaviour = pixels[:, 1:, :, :]
        pixels = torch.concat([img, behaviour], axis=1)

        pixels = self.monitor.sample_screen(pixels, grid)

        img = self.pixel_transform(pixels[:, None, 0, :, :])
        behaviour = pixels[:, 1:, :, :]
        pixels = torch.concat([img, behaviour], axis=1)

        return pixels


class Perspective(nn.ModuleDict):
    """
    Container for multiple SinglePerspective modules keyed by dataset identifiers.
    """
    def __init__(self, data_keys, retina_degree=75, mlp_features=16, mlp_layers=3):
        super().__init__()

        for k in data_keys:
            retina = Retina(degree=retina_degree, mlp_features=mlp_features, mlp_layers=mlp_layers)
            monitor = Monitor()
            self.add_module(k, SinglePerspective(retina, monitor, PixelTransform()))
