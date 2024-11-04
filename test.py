from diffusers import StableDiffusionPipeline
import torch

# モデルのロード
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cpu")

# 画像生成
prompt = "a cozy bedroom with large windows and a comfortable bed"
image = pipe(prompt).images[0]

# 画像の保存
image.save("generated_bedroom.png")

from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# モデルのロード
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cpu")

# 入力画像の準備
init_image = Image.open("875.jpg").convert("RGB")
init_image = init_image.resize((512, 512))

# 画像生成
prompt = "a luxurious bedroom with a chandelier"
image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]

# 画像の保存
image.save("generated_bedroom_from_image.png")