from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
import torch
from PIL import Image
import numpy as np
import cv2
import random
import os
import glob
from tqdm import tqdm
import albumentations as A
import gc

GenImage_LIST = [
    'stable_diffusion_v_1_4/imagenet_ai_0419_sdv4', 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5',
    'Midjourney/imagenet_midjourney', 'ADM/imagenet_ai_0508_adm', 'wukong/imagenet_ai_0424_wukong',
    'glide/imagenet_glide', 'VQDM/imagenet_ai_0419_vqdm', 'BigGAN/imagenet_ai_0419_biggan'
]
DRCT_2M_LIST = [
    'ldm-text2im-large-256', 'stable-diffusion-v1-4', 'stable-diffusion-v1-5', 'stable-diffusion-2-1',
    'stable-diffusion-xl-base-1.0', 'stable-diffusion-xl-refiner-1.0', 'sd-turbo', 'sdxl-turbo',
    'lcm-lora-sdv1-5', 'lcm-lora-sdxl',  'sd-controlnet-canny',
    'sd21-controlnet-canny', 'controlnet-canny-sdxl-1.0', 'stable-diffusion-inpainting',
    'stable-diffusion-2-inpainting', 'stable-diffusion-xl-1.0-inpainting-0.1',
]

def create_crop_transforms(height=224, width=224):
    """
    创建一个用于填充和中心裁剪图像的增强变换组合。

    Args:
        height (int, optional): 目标图像的高度，默认为224。
        width (int, optional): 目标图像的宽度，默认为224。

    Returns:
        albumentations.core.composition.Compose: 包含填充和中心裁剪操作的增强变换组合。
    """
    aug_list = [
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.CenterCrop(height=height, width=width)
    ]
    return A.Compose(aug_list)


def set_seed(seed: int):
    """
    设置随机数种子，确保代码的可复现性。在深度学习任务中，随机数种子的设置能保证每次运行代码时，
    随机操作（如数据打乱、权重初始化等）生成的随机数序列相同。

    Args:
        seed (int): 要设置的随机数种子，是一个整数。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def find_nearest_multiple(a, multiple=8):
    """
    找到最接近a的multiple倍数，且该倍数大于或等于a

    Args:
        a (int): 输入的整数，需要找到其最接近的multiple倍数
        multiple (int, optional): 倍数，默认为8

    Returns:
        int: 最接近a且大于或等于a的multiple倍数
    """
    n = a // multiple
    remainder = a % multiple
    if remainder == 0:
        return a
    else:
        return (n + 1) * multiple


def pad_image_to_size(image, target_width=224, target_height=224, fill_value=255):
    """
    将图像填充为目标宽度和高度，使用指定的填充值（默认为255）

    Args:
        image (np.ndarray): 输入的图像数组，形状为 (height, width, channels)。
        target_width (int, optional): 目标图像的宽度，默认为224。
        target_height (int, optional): 目标图像的高度，默认为224。
        fill_value (int, optional): 填充时使用的值，默认为255。

    Returns:
        np.ndarray: 填充后的图像数组，形状为 (target_height, target_width, channels)。
    """
    height, width = image.shape[:2]

    if height < target_height:
        pad_height = target_height - height
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
    else:
        pad_top = pad_bottom = 0

    if width < target_width:
        pad_width = target_width - width
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
    else:
        pad_left = pad_right = 0

    padded_image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=fill_value
    )

    return padded_image


def center_crop(image, crop_width, crop_height):
    """
    对输入图像进行中心裁剪，如果裁剪后的图像尺寸小于目标尺寸，则进行填充。

    Args:
        image (np.ndarray): 输入的图像数组，形状为 (height, width, channels)。
        crop_width (int): 期望裁剪后的图像宽度。
        crop_height (int): 期望裁剪后的图像高度。

    Returns:
        np.ndarray: 中心裁剪后的图像数组，如果尺寸不足则进行填充。
    """
    height, width = image.shape[:2]

    if width > crop_width:
        start_x = (width - crop_width) // 2
        end_x = start_x + crop_width
    else:
        start_x, end_x = 0, width

    if height > crop_height:
        start_y = (height - crop_height) // 2
        end_y = start_y + crop_height
    else:
        start_y, end_y = 0, height

    cropped_image = image[start_y:end_y, start_x:end_x]

    if cropped_image.shape[0] < crop_height or cropped_image.shape[1] < crop_width:
        cropped_image = pad_image_to_size(cropped_image, target_width=crop_width, target_height=crop_height,
                                          fill_value=255)

    return cropped_image


def stable_diffusion_inpainting(pipe, image, mask_image, prompt, steps=50, height=512, width=512,
                                seed=2023, guidance_scale=7.5):
    """
    使用Stable Diffusion进行图像修复。

    Args:
        pipe: 预加载的Stable Diffusion图像修复模型管道。
        image (np.ndarray): 输入的待修复图像数组。
        mask_image (np.ndarray): 用于指定修复区域的掩码图像数组。
        prompt (str): 用于生成图像的文本提示。
        steps (int, optional): 推理步数，默认为50。
        height (int, optional): 生成图像的高度，默认为512。
        width (int, optional): 生成图像的宽度，默认为512。
        seed (int, optional): 随机数种子，用于保证结果的可复现性，默认为2023。
        guidance_scale (float, optional): 引导系数，控制生成结果与提示的匹配程度，默认为7.5。

    Returns:
        PIL.Image.Image: 修复后的图像。
    """
    set_seed(int(seed))
    image_pil = Image.fromarray(image)
    mask_image_pil = Image.fromarray(mask_image).convert("L")
    new_image = pipe(prompt=prompt, image=image_pil, mask_image=mask_image_pil,
                     height=height, width=width, num_inference_steps=steps,
                     guidance_scale=guidance_scale).images[0]

    return new_image


def read_image(image_path, max_size=512):
    """
    读取指定路径的图像，对图像进行裁剪、调整尺寸以满足8的倍数要求，并生成掩码图像。

    Args:
        image_path (str): 图像文件的路径。
        max_size (int, optional): 图像的最大高度和宽度，默认为512。

    Returns:
        tuple: 包含调整后的图像数组、掩码图像数组和原始图像形状的元组。
    """
    create_crop_transforms(height=224, width=224)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = image.shape[:2]
    height = height if height < max_size else max_size
    width = width if width < max_size else max_size
    transform = create_crop_transforms(height=height, width=width)
    image = transform(image=image)["image"]

    original_shape = image.shape
    new_height = find_nearest_multiple(original_shape[0], multiple=8)
    new_width = find_nearest_multiple(original_shape[1], multiple=8)
    new_image = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    new_image[:original_shape[0], :original_shape[1]] = image

    mask_image = np.zeros_like(image)

    del transform
    del image
    gc.collect()

    return new_image, mask_image, original_shape


def func(image_path, save_path, crop_save_path, step=50, max_size=1024):
    """
    读取指定路径的图像，使用Stable Diffusion进行图像修复，
    并将修复后的图像和裁剪后的原始图像保存到指定路径。

    Args:
        image_path (str): 输入图像的文件路径。
        save_path (str): 保存修复后图像的文件路径。
        crop_save_path (str): 保存裁剪后原始图像的文件路径。
        step (int, optional): 推理步数，默认为50。
        max_size (int, optional): 图像的最大高度和宽度，默认为1024。
    """
    image, mask_image, original_shape = read_image(image_path, max_size)
    new_image = stable_diffusion_inpainting(pipe, image, mask_image, prompt='', steps=step,
                                            height=image.shape[0],
                                            width=image.shape[1],
                                            seed=2023, guidance_scale=7.5)
    new_image = new_image.crop(box=(0, 0, original_shape[1], original_shape[0]))
    new_image.save(save_path)
    if crop_save_path and not os.path.exists(crop_save_path):
        image = Image.fromarray(image).crop(box=(0, 0, original_shape[1], original_shape[0]))
        image.save(crop_save_path)


if __name__ == '__main__':
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    root = '/home/data1/Datasets/Dataset_hjs/dataset_DRCT'

    sd_model_names = ["runwayml/stable-diffusion-inpainting",
                      "stabilityai/stable-diffusion-2-inpainting",
                      "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                      ]
    index = 0
    sd_model_name = sd_model_names[index]
    if 'xl' in sd_model_name:
        pipe = AutoPipelineForInpainting.from_pretrained(
            sd_model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(device)
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_name,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.enable_xformers_memory_efficient_attention()
        pipe = pipe.to(device)
    print(f"Load model successful:{sd_model_name}")

    step = 50
    phase = 'val'
    inpainting_dir = {0: 'full_inpainting', 1: 'full_inpainting2', 2: 'full_inpainting_xl'}[index]
    if step != 50:
        inpainting_dir = f'step{step}_{inpainting_dir}'

    for model_name in DRCT_2M_LIST:
        print(f"\n{'='*30}\n[INFO] Starting processing for model: {model_name}\n{'='*30}")

        image_root = f'{root}/images/{model_name}/{phase}2017'
        save_root = f'{root}/fake_rec_images_all_final/{model_name}/{inpainting_dir}/{phase}2017'
        
        crop_root = None

        if not os.path.isdir(image_root):
            print(f"[WARNING] Source directory not found, skipping: {image_root}")
            continue

        os.makedirs(save_root, exist_ok=True)
        if crop_root is not None:
            os.makedirs(crop_root, exist_ok=True)

        start_index, end_index = 0, 200000
        image_paths = sorted(glob.glob(f"{image_root}/*.*"))[start_index:end_index]

        if not image_paths:
            print(f"[WARNING] No images found in {image_root}, skipping.")
            continue

        print(f'[INFO] Found {len(image_paths)} images for model: {model_name}')

        failed_num = 0
        for image_path in tqdm(image_paths, desc=f"Processing {model_name}"):
            image_name = os.path.basename(image_path).split('.')[0]
            save_path = os.path.join(save_root, image_name + '.png')

            if crop_root is not None:
                crop_save_path = os.path.join(crop_root, image_name + '.png')
            else:
                crop_save_path = None

            if os.path.exists(save_path):
                if (crop_root is not None and os.path.exists(crop_save_path)) or crop_root is None:
                    continue
            
            try:
                func(image_path, save_path, crop_save_path, step=step, max_size=1024)
            except Exception as e:
                failed_num += 1
                print(f'[ERROR] Failed to generate image in {image_path}. Reason: {e}')

        print(f"[SUCCESS] Inference finished for model_id: {model_name}. Failed images: {failed_num}\n")

    print(f"\n{'='*30}\nAll 16 models have been processed.\n{'='*30}")
