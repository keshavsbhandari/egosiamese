import math
import torch.nn.functional as F
import torch
import numpy as np
from transformers import PerceiverForOpticalFlow
import torch
import itertools
from PIL import Image
from flow_vis import flow_to_color
from Utils.utils import maprange
from Extras.loadconfigs import DATA_ROOT
from pathlib import Path
from tqdm import tqdm

def compute_grid_indices(image_shape, patch_size, min_overlap=20):
  if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  ys = list(range(0, image_shape[0], patch_size[0] - min_overlap))
  xs = list(range(0, image_shape[1], patch_size[1] - min_overlap))
  # Make sure the final patch is flush with the image boundary
  ys[-1] = image_shape[0] - patch_size[0]
  xs[-1] = image_shape[1] - patch_size[1]
  return itertools.product(ys, xs)

def normalize(im):
  return im / 255.0 * 2 - 1

# source: https://discuss.pytorch.org/t/tf-extract-image-patches-in-pytorch/43837/9
def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
    
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    
    return patches.view(b,-1,patches.shape[-2], patches.shape[-1])

def compute_optical_flow(model, img1, img2, grid_indices, patch_size, FLOW_SCALE_FACTOR = 20):
  """Function to compute optical flow between two images.

  To compute the flow between images of arbitrary sizes, we divide the image
  into patches, compute the flow for each patch, and stitch the flows together.

  Args:
    model: PyTorch Perceiver model 
    img1: first image
    img2: second image
    grid_indices: indices of the upper left corner for each patch.
  """
  img1 = torch.tensor(np.moveaxis(img1, -1, 0))#HW3-> 3HW
  img2 = torch.tensor(np.moveaxis(img2, -1, 0))#HW3-> 3HW
  imgs = torch.stack([img1, img2], dim=0)[None]#23HW->123HW
  height = imgs.shape[-2]
  width = imgs.shape[-1]

  #print("Shape of imgs after stacking:", imgs.shape)

  #patch_size = model.config.train_size
  
  if height < patch_size[0]:
    raise ValueError(
        f"Height of image (shape: {imgs.shape}) must be at least {patch_size[0]}."
        "Please pad or resize your image to the minimum dimension."
    )
  if width < patch_size[1]:
    raise ValueError(
        f"Width of image (shape: {imgs.shape}) must be at least {patch_size[1]}."
        "Please pad or resize your image to the minimum dimension."
    )

  flows = 0
  flow_count = 0

  for y, x in grid_indices:    
    imgs = torch.stack([img1, img2], dim=0)[None]
    inp_piece = imgs[..., y : y + patch_size[0],
                     x : x + patch_size[1]]
    
    batch_size, _, C, H, W = inp_piece.shape
    patches = extract_image_patches(inp_piece.view(batch_size*2,C,H,W), kernel=3)
    _, C, H, W = patches.shape
    try:    
        patches = patches.view(batch_size, -1, C, H, W).float().to(model.device)
    except:
        patches = patches.view(batch_size, -1, C, H, W).float().to(model.module.device)
        
    # actual forward pass
    with torch.no_grad():
      output = model(inputs=patches).logits * FLOW_SCALE_FACTOR
    
    # the code below could also be implemented in PyTorch
    flow_piece = output.cpu().detach().numpy()
    
    weights_x, weights_y = np.meshgrid(
        torch.arange(patch_size[1]), torch.arange(patch_size[0]))

    weights_x = np.minimum(weights_x + 1, patch_size[1] - weights_x)
    weights_y = np.minimum(weights_y + 1, patch_size[0] - weights_y)
    weights = np.minimum(weights_x, weights_y)[np.newaxis, :, :,
                                                np.newaxis]
    padding = [(0, 0), (y, height - y - patch_size[0]),
               (x, width - x - patch_size[1]), (0, 0)]
    flows += np.pad(flow_piece * weights, padding)
    flow_count += np.pad(weights, padding)

    # delete activations to avoid OOM
    del output

  flows /= flow_count
  return flows

def computeFLow(allframes_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PerceiverForOpticalFlow.from_pretrained("deepmind/optical-flow-perceiver")
    model.to(device)
    TRAIN_SIZE = model.config.train_size
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    
    for frame_1 in tqdm(allframes_path):
        frame_2 = frame_1.parent / f"{str(int(frame_1.stem)+1).zfill(4)}{frame_1.suffix}"
        flow_root = Path(frame_1.parent.as_posix().replace("frames","perceiver"))
        
        if not flow_root.parent.exists():
            flow_root.parent.mkdir()
        
        if not flow_root.exists():
            flow_root.mkdir()
        
        flow_name = frame_1.stem
        flow_path = (flow_root/flow_name).as_posix()
        
        assert frame_1.exists(), f"{frame_1} doesn't exist"
        assert frame_2.exists(), f"{frame_2} doesn't exist"
    
        a1 = Image.open(frame_1)
        a2 = Image.open(frame_2)
        target_size = a1.size[::-1]
        a1 = a1.resize((1024,512))
        a2 = a2.resize((1024,512))
        im1 = np.array(a1)
        im2 = np.array(a2)
        grid_indices = compute_grid_indices(im1.shape, patch_size=TRAIN_SIZE)
        flow = compute_optical_flow(model, normalize(im1), normalize(im2), grid_indices, TRAIN_SIZE)
        flow = torch.from_numpy(flow).permute(0,3,1,2)
        rmap, flow = maprange(flow)
        flow = torch.nn.functional.interpolate(flow, target_size)
        _, flow = maprange(flow, **rmap)
        flow = flow.permute(0,2,3,1)
        scaleh, scalew = target_size[0]/flow.size(1), target_size[1]/flow.size(2)
        flow = flow * torch.tensor([scaleh,scalew])[None,None,None,:]
        flow = flow[0].numpy()
        np.save(flow_path, flow)

if __name__ == '__main__':
    allframes_path = []
    videos = sorted(Path(DATA_ROOT).glob('*/*/*/frames/*'))
    for video in tqdm(videos):
        allframes_path.extend(sorted(video.glob('*.jpg'))[:-1])
    
    computeFLow(allframes_path)