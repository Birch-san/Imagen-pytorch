import torch as th

from itertools import tee

from PIL import Image
from IPython.display import display

from imagen_pytorch.model_creation import create_model_and_diffusion as create_model_and_diffusion_imagen
from imagen_pytorch.model_creation import model_and_diffusion_defaults as model_and_diffusion_defaults_imagen
from imagen_pytorch.train_all import _fix_path
from imagen_pytorch.device import get_default_device_backend
from transformers import AutoTokenizer
import cv2

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

import numpy as np

def model_fn(x_t, ts, **kwargs):
    guidance_scale = 5
    half = x_t[: len(x_t) // 2]
    combined = th.cat([half, half], dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = th.cat([half_eps, half_eps], dim=0)
    return th.cat([eps, rest], dim=1)

def show_images(batch: th.Tensor):
    """ Display a batch of images inline."""
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))

def get_numpy_img(img):
    scaled = ((img + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([img.shape[2], -1, 3])
    return cv2.cvtColor(reshaped.numpy(), cv2.COLOR_BGR2RGB)

checkpoint_path = './ImagenT5-3B/model.pt'
checkpoint = _fix_path(checkpoint_path)

device_name = get_default_device_backend()

device = th.device(device_name)

options = model_and_diffusion_defaults_imagen()
options['use_fp16'] = False
options['diffusion_steps'] = 200
options['num_res_blocks'] = 3
options['t5_name'] = 't5-3b'
options['cache_text_emb'] = True
model, diffusion = create_model_and_diffusion_imagen(**options)
model.eval()

model.to(device)


model.load_state_dict(checkpoint, strict=False)
print('total base parameters', sum(x.numel() for x in model.parameters()))

num_params = sum(param.numel() for param in model.parameters())
print('num_params', num_params)

realesrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)

netscale = 4

upsampler = RealESRGANer(
    scale=netscale,
    model_path='./Real-ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth',
    model=realesrgan_model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False # on-CPU only supports False. if you're on GPU you should use True
)

face_enhancer = GFPGANer(
    # https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
    model_path='./GFPGANv1.3.pth',
    upscale=4,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler
)

tokenizer = AutoTokenizer.from_pretrained(options['t5_name'])

prompt = 'teddy bears lunch atop skyscraper'

text_encoding = tokenizer(
    prompt,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt"
)

uncond_text_encoding = tokenizer(
    '',
    max_length=128,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors="pt"
)

batch_size = 4
cond_tokens = th.from_numpy(np.array([text_encoding['input_ids'][0].numpy() for i in range(batch_size)]))
uncond_tokens = th.from_numpy(np.array([uncond_text_encoding['input_ids'][0].numpy() for i in range(batch_size)]))
cond_attention_mask = th.from_numpy(np.array([text_encoding['attention_mask'][0].numpy() for i in range(batch_size)]))
uncond_attention_mask = th.from_numpy(np.array([uncond_text_encoding['attention_mask'][0].numpy() for i in range(batch_size)]))
model_kwargs = {}
model_kwargs["tokens"] = th.cat((cond_tokens,
                                 uncond_tokens)).to(device)
model_kwargs["mask"] = th.cat((cond_attention_mask,
                               uncond_attention_mask)).to(device)

model.del_cache()
sample = diffusion.p_sample_loop(
    model_fn,
    (batch_size * 2, 3, 64, 64),
    clip_denoised=True,
    model_kwargs=model_kwargs,
    device=device_name,
    progress=True,
)[:batch_size]
model.del_cache()

images = map(lambda ix: get_numpy_img(sample[ix].unsqueeze(0)), range(batch_size))

images0, images1 = tee(images)

for ix,img in enumerate(images0):
    cv2.imwrite(f'./out/{prompt}{ix}.pre.jpg', img)
    th.save(img, f'./out/{prompt}{ix}.pre.pt')

for ix,img in enumerate(images1):
    _, _, enhanced = face_enhancer.enhance(img, has_aligned=False,
                                              only_center_face=False, paste_back=True)
    cv2.imwrite(f'./out/{prompt}{ix}.jpg', enhanced)
    th.save(img, f'./out/{prompt}{ix}.pt')
