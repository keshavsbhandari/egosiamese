# Utils
import torch
import torchvision.transforms as trans
from PIL import Image
import pathlib
from tqdm import tqdm
torch.pi = torch.acos(torch.zeros(1)).item() * 2
from matplotlib import cm

# "Credits : https://github.com/sunset1995/py360convert"

#@profile
def getuvCenter(u_size=4, v_size=4):
    v = torch.linspace(90, -90, v_size)
    u = torch.linspace(-180, 180, u_size)
    vdiff = torch.abs(v[1] - v[0]).long()
    udiff = torch.abs(u[1] - u[0]).long()
    uvCenter = torch.stack(torch.meshgrid([v, u]), -1).reshape(-1, 2)
    return uvCenter, udiff, vdiff

#@profile
def Te2p(e_img, h_fov, v_fov, u_deg, v_deg, out_hw, in_rot_deg=torch.tensor([0.]), mode='bilinear'):
    '''
    e_img:   ndarray in shape of [H, W, *]
    h_fov,v_fov: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''
    b, c, h, w = e_img.shape

    h_fov, v_fov = h_fov * torch.pi / 180., v_fov * torch.pi / 180.

    in_rot = in_rot_deg * torch.pi / 180.

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    u = -u_deg * torch.pi / 180.
    v = v_deg * torch.pi / 180.
    xyz = Txyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = Txyz2uv(xyz)
    coor_xy = Tuv2coor(uv, torch.tensor([h], dtype=float), torch.tensor([w], dtype=float))
    mid = torch.tensor([w / 2., h / 2.]).reshape(1, 1, 2)
    cords = (coor_xy - mid) / mid
    pers_img = torch.nn.functional.grid_sample(input=e_img, grid=torch.cat([cords.unsqueeze(0).float()]*e_img.size(0)), align_corners=True,
                                               mode=mode)
    return pers_img, coor_xy

#@profile
def TgetCors(h_fov, v_fov, u_deg, v_deg, out_hw, in_rot_deg=torch.tensor([0.]), mode='bilinear'):
    '''
    e_img_shape:   [b,c,h,w]
    h_fov,v_fov: scalar or (scalar, scalar) field of view in degree
    u_deg:   horizon viewing angle in range [-180, 180]
    v_deg:   vertical viewing angle in range [-90, 90]
    '''

    h_fov, v_fov = h_fov * torch.pi / 180., v_fov * torch.pi / 180.

    in_rot = in_rot_deg * torch.pi / 180.

    if mode == 'bilinear':
        order = 1
    elif mode == 'nearest':
        order = 0
    else:
        raise NotImplementedError('unknown mode')

    u = -u_deg * torch.pi / 180.
    v = v_deg * torch.pi / 180.
    xyz = Txyzpers(h_fov, v_fov, u, v, out_hw, in_rot)
    uv = Txyz2uv(xyz)
    coor_xy = Tuv2coor(uv, torch.tensor([h], dtype=float), torch.tensor([w], dtype=float))
    return coor_xy.long()

#@profile
def Tuv2coor(uv, h, w):
    '''
    uv: ndarray in shape of [..., 2]
    h: int, height of the equirectangular image
    w: int, width of the equirectangular image
    '''
    u, v = torch.split(uv, 1, -1)
    coor_x = (u / (2 * torch.pi) + 0.5) * w - 0.5
    coor_y = (-v / torch.pi + 0.5) * h - 0.5
    return torch.cat([coor_x, coor_y], -1)

#@profile
def Tcoor2uv(coorxy, h, w):
    coor_x, coor_y = torch.split(coorxy, 1, -1)
    u = ((coor_x + 0.5) / w - 0.5) * 2 * torch.pi
    v = -((coor_y + 0.5) / h - 0.5) * torch.pi
    return torch.cat([u, v], -1)

#@profile
def Tuv2unitxyz(uv):
    u, v = torch.split(uv, 1, -1)
    y = torch.sin(v)
    c = torch.cos(v)
    x = c * np.sin(u)
    z = c * np.cos(u)

    return torch.cat([x, y, z], dim=-1)

#@profile
def Txyz2uv(xyz):
    '''
    xyz: ndarray in shape of [..., 3]
    '''
    x, y, z = torch.split(xyz, 1, -1)
    u = torch.atan2(x, z)
    c = torch.sqrt(x ** 2 + z ** 2)
    v = torch.atan2(y, c)

    return torch.cat([u, v], -1)

#@profile
def Trotation_matrix(rad, ax):
    """
    rad : torch.tensor, Eg. torch.tensor([2.0])
    ax  : torch.tensor, Eg. [1,0,0] or [0,1,0] or [0,0,1]
    """
    ax = ax / torch.pow(ax, 2).sum()
    R = torch.diag(torch.cat([torch.cos(rad)] * 3))
    R = R + torch.outer(ax, ax) * (1.0 - torch.cos(rad))
    ax = ax * torch.sin(rad)
    R = R + torch.tensor([[0, -ax[2], ax[1]],
                          [ax[2], 0, -ax[0]],
                          [-ax[1], ax[0], 0]], dtype=ax.dtype)
    return R

#@profile
def Txyzpers(h_fov, v_fov, u, v, out_hw, in_rot):
    out = torch.ones((*out_hw, 3), dtype=float)
    x_max = torch.tan(torch.tensor([h_fov / 2])).item()
    y_max = torch.tan(torch.tensor([v_fov / 2])).item()
    x_rng = torch.linspace(-x_max, x_max, out_hw[1], dtype=float)
    y_rng = torch.linspace(-y_max, y_max, out_hw[0], dtype=float)
    out[..., :2] = torch.stack(torch.meshgrid(x_rng, -y_rng), -1).permute(1, 0, 2)
    Rx = Trotation_matrix(v, torch.tensor([1, 0, 0], dtype=float))
    Ry = Trotation_matrix(u, torch.tensor([0, 1, 0], dtype=float))
    dots = (torch.tensor([[0, 0, 1]], dtype=float) @ Rx) @ Ry
    Ri = Trotation_matrix(in_rot, dots[0])
    return ((out @ Rx) @ Ry) @ Ri