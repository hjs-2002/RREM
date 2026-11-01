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

# GenImage_LIST = [
#     'stable_diffusion_v_1_4/imagenet_ai_0419_sdv4'
# ]
DRCT_2M_LIST = [
    'ldm-text2im-large-256', 'stable-diffusion-v1-4', 'stable-diffusion-v1-5', 'stable-diffusion-2-1',
    'stable-diffusion-xl-base-1.0', 'stable-diffusion-xl-refiner-1.0', 'sd-turbo', 'sdxl-turbo',
    'lcm-lora-sdv1-5', 'lcm-lora-sdxl', 'sd-controlnet-canny',
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
    # 定义增强操作列表，包含填充和中心裁剪操作
    aug_list = [
        # 如果图像高度或宽度小于目标值，则进行填充，使用常量值0填充
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0),
        # 对图像进行中心裁剪，裁剪为目标高度和宽度
        A.CenterCrop(height=height, width=width)
    ]
    # 将增强操作列表组合成一个变换对象
    return A.Compose(aug_list)


def set_seed(seed: int):
    """
    设置随机数种子，确保代码的可复现性。在深度学习任务中，随机数种子的设置能保证每次运行代码时，
    随机操作（如数据打乱、权重初始化等）生成的随机数序列相同。

    Args:
        seed (int): 要设置的随机数种子，是一个整数。
    """
    # 设置PyTorch的CPU随机数种子，保证在CPU上的随机操作可复现
    torch.manual_seed(seed)
    # 设置PyTorch的所有GPU随机数种子，保证在多GPU环境下的随机操作可复现
    torch.cuda.manual_seed_all(seed)
    # 将CuDNN的卷积算法设置为确定性模式，确保每次卷积操作的结果相同
    torch.backends.cudnn.deterministic = True
    # 关闭CuDNN的自动寻找最优卷积算法功能，避免因选择不同算法导致结果不一致
    torch.backends.cudnn.benchmark = False
    # 设置NumPy的随机数种子，保证NumPy中的随机操作可复现
    np.random.seed(seed)
    # 设置Python内置random模块的随机数种子，保证该模块的随机操作可复现
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
    # 计算a除以multiple的商
    n = a // multiple
    # 计算a除以multiple的余数
    remainder = a % multiple
    if remainder == 0:
        # 如果a能被multiple整除，那么a本身就是multiple的倍数，直接返回a
        return a
    else:
        # 若a不能被multiple整除，则需要找到比a大的下一个multiple的倍数
        # 通过将商加1后再乘以multiple得到
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
    # 获取输入图像的高度和宽度
    height, width = image.shape[:2]

    # 计算垂直方向上需要填充的像素数
    if height < target_height:
        # 计算需要填充的总像素数
        pad_height = target_height - height
        # 计算顶部需要填充的像素数
        pad_top = pad_height // 2
        # 计算底部需要填充的像素数，确保总填充像素数正确
        pad_bottom = pad_height - pad_top
    else:
        # 若图像高度已达到或超过目标高度，则不需要填充
        pad_top = pad_bottom = 0

    # 计算水平方向上需要填充的像素数
    if width < target_width:
        # 计算需要填充的总像素数
        pad_width = target_width - width
        # 计算左侧需要填充的像素数
        pad_left = pad_width // 2
        # 计算右侧需要填充的像素数，确保总填充像素数正确
        pad_right = pad_width - pad_left
    else:
        # 若图像宽度已达到或超过目标宽度，则不需要填充
        pad_left = pad_right = 0

    # 使用np.pad函数对图像进行填充
    padded_image = np.pad(
        image,
        # 定义每个轴上的填充量，格式为 ((top, bottom), (left, right), (front, back))
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        # 使用常量填充模式
        mode="constant",
        # 指定填充的值
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
    # 获取输入图像的高度和宽度
    height, width = image.shape[:2]

    # 计算裁剪区域在水平方向的起始点和终点
    if width > crop_width:
        # 若图像宽度大于期望裁剪宽度，计算起始点
        start_x = (width - crop_width) // 2
        # 计算终点
        end_x = start_x + crop_width
    else:
        # 若图像宽度小于等于期望裁剪宽度，从图像边缘开始裁剪
        start_x, end_x = 0, width

    # 计算裁剪区域在垂直方向的起始点和终点
    if height > crop_height:
        # 若图像高度大于期望裁剪高度，计算起始点
        start_y = (height - crop_height) // 2
        # 计算终点
        end_y = start_y + crop_height
    else:
        # 若图像高度小于等于期望裁剪高度，从图像边缘开始裁剪
        start_y, end_y = 0, height

    # 使用数组切片操作对图像进行中心裁剪
    cropped_image = image[start_y:end_y, start_x:end_x]

    # 检查裁剪后的图像尺寸是否小于目标尺寸
    if cropped_image.shape[0] < crop_height or cropped_image.shape[1] < crop_width:
        # 若尺寸不足，调用 pad_image_to_size 函数进行填充
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
    # 设置随机数种子，确保每次运行的结果一致
    set_seed(int(seed))
    # 将输入的图像数组转换为PIL图像对象
    image_pil = Image.fromarray(image)
    # 将掩码图像数组转换为PIL图像对象，并转换为单通道灰度图像
    mask_image_pil = Image.fromarray(mask_image).convert("L")
    # image and mask_image should be PIL images.
    # 掩码图像中白色区域表示需要修复的部分，黑色区域表示保持原样
    # 调用模型管道进行图像修复，传入提示、图像、掩码等参数
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
    # 调用函数创建一个用于填充和中心裁剪图像的增强变换组合，此处参数固定为224x224，但实际未使用
    create_crop_transforms(height=224, width=224)
    # 使用OpenCV读取指定路径的图像
    image = cv2.imread(image_path)
    # 将图像从BGR颜色空间转换为RGB颜色空间
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 裁剪图像
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 确保图像高度不超过最大尺寸
    height = height if height < max_size else max_size
    # 确保图像宽度不超过最大尺寸
    width = width if width < max_size else max_size
    # 根据调整后的高度和宽度创建新的裁剪变换组合
    transform = create_crop_transforms(height=height, width=width)
    # 应用裁剪变换到图像上
    image = transform(image=image)["image"]

    # 处理图像尺寸为8的倍数
    # 记录原始图像的形状
    original_shape = image.shape
    # 找到最接近且大于等于原始图像高度的8的倍数
    new_height = find_nearest_multiple(original_shape[0], multiple=8)
    # 找到最接近且大于等于原始图像宽度的8的倍数
    new_width = find_nearest_multiple(original_shape[1], multiple=8)
    # 创建一个新的全零数组，尺寸为调整后的高度和宽度
    new_image = np.zeros(shape=(new_height, new_width, 3), dtype=image.dtype)
    # 将原始图像复制到新数组的对应位置
    new_image[:original_shape[0], :original_shape[1]] = image

    # 创建与原始图像形状相同的全零掩码图像
    mask_image = np.zeros_like(image)

    # 删除不再使用的变换对象和原始图像对象
    del transform
    del image
    # 手动触发垃圾回收，释放内存
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
    # 调用 read_image 函数读取图像，对图像进行裁剪、调整尺寸以满足8的倍数要求，并生成掩码图像
    # 返回调整后的图像数组、掩码图像数组和原始图像形状
    image, mask_image, original_shape = read_image(image_path, max_size)
    # 打印调试信息，可查看图像形状、掩码图像形状和掩码图像的唯一值，当前为注释状态
    # print(image.shape, mask_image.shape, np.unique(mask_image))
    # 调用 stable_diffusion_inpainting 函数进行图像修复
    # 传入预加载的模型管道、调整后的图像、掩码图像、空提示文本、推理步数等参数
    new_image = stable_diffusion_inpainting(pipe, image, mask_image, prompt='', steps=step,
                                          height=image.shape[0],
                                          width=image.shape[1],
                                          seed=2023, guidance_scale=7.5)
    # 恢复修复后图像到原始尺寸，使用crop方法裁剪图像
    new_image = new_image.crop(box=(0, 0, original_shape[1], original_shape[0]))
    # 将修复后的图像保存到指定路径
    new_image.save(save_path)
    # 检查裁剪后原始图像的保存路径是否存在
    if crop_save_path and not os.path.exists(crop_save_path):
        # 将调整后的图像数组转换为PIL图像对象，并裁剪到原始尺寸
        image = Image.fromarray(image).crop(box=(0, 0, original_shape[1], original_shape[0]))
        # 将裁剪后的原始图像保存到指定路径
        image.save(crop_save_path)


if __name__ == '__main__':
    # 加载稳定扩散模型
    # 检查是否有可用的CUDA设备，若有则使用GPU，否则使用CPU
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    # 数据集根目录
    root = '/home/law/data'

    # 定义要使用的稳定扩散模型名称列表
    sd_model_names = ["runwayml/stable-diffusion-inpainting",
                      "stabilityai/stable-diffusion-2-inpainting",
                      "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
                      ]
    # 选择要使用的模型索引 (例如，我们固定使用第一个inpainting模型来重建所有图像)
    index = 0
    # 根据索引获取对应的模型名称
    sd_model_name = sd_model_names[index]
    # 判断是否为XL版本的模型
    if 'xl' in sd_model_name:
        # 加载XL版本的图像修复模型
        pipe = AutoPipelineForInpainting.from_pretrained(
            sd_model_name,
            torch_dtype=torch.float16,  # 使用半精度浮点数以减少内存占用
            variant="fp16",
            safety_checker=None,  # 移除安全检查器
            requires_safety_checker=False,  # 关闭安全审查机制
        )
        # 启用xformers以提高内存使用效率
        # pipe.enable_xformers_memory_efficient_attention()
        # 启用模型CPU卸载以节省GPU内存
        # pipe.enable_model_cpu_offload()
        # 将模型移动到指定设备
        pipe = pipe.to(device)
    else:
        # 加载非XL版本的图像修复模型
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model_name,
            # revision="fp16",
            torch_dtype=torch.float16,  # 使用半精度浮点数以减少内存占用
            safety_checker=None,  # 移除安全检查器
            requires_safety_checker=False,  # 关闭安全审查机制
        )
        # 启用xformers以提高内存使用效率
        pipe.enable_xformers_memory_efficient_attention()
        # 启用模型CPU卸载以节省GPU内存
        # pipe.enable_model_cpu_offload()
        # 将模型移动到指定设备
        pipe = pipe.to(device)
    # 打印模型加载成功信息
    print(f"Load model successful:{sd_model_name}")

    # ==================== 主要修改区域：循环处理 GenImage_LIST 中的模型 ====================
    # 设置通用参数
    # step = 50
    # step = 1
    step =10
    phase = 'val'
    label = 'nature'
    # 根据选择的inpainting模型索引，确定输出目录名
    inpainting_dir = {0: 'inpainting', 1: 'inpainting2', 2: 'inpainting_xl'}[index]
    if step != 50:
        inpainting_dir = f'step{step}_{inpainting_dir}'

    # 遍历 GenImage_LIST 中的每一个模型
    for model_name in GenImage_LIST:
        print(f"\n{'='*30}\n[INFO] Starting processing for model: {model_name}\n{'='*30}")
        
        # 为当前模型配置路径
        image_root = f'{root}/GenImage/{model_name}/{phase}/{label}'
        save_root = f'{root}/DR/GenImage/{model_name}/{phase}/{label}/{inpainting_dir}'
        crop_root = f'{root}/DR/GenImage/{model_name}/{phase}/{label}/crop'

        # 检查源文件夹是否存在，如果不存在则跳过
        if not os.path.isdir(image_root):
            print(f"[WARNING] Source directory not found, skipping: {image_root}")
            continue

        # 创建保存重建图像和裁剪图像的目录
        os.makedirs(save_root, exist_ok=True)
        if crop_root is not None:
            os.makedirs(crop_root, exist_ok=True)

        # 获取图像路径
        start_index, end_index = 0, 200000
        image_paths = sorted(glob.glob(f"{image_root}/*.*"))[start_index:end_index]

        # 如果找不到图像，打印警告并跳过
        if not image_paths:
            print(f"[WARNING] No images found in {image_root}, skipping.")
            continue

        print(f'[INFO] Found {len(image_paths)} images for model: {model_name}')

        failed_num = 0
        # 使用tqdm显示处理进度
        for image_path in tqdm(image_paths, desc=f"Processing {model_name}"):
            image_name = os.path.basename(image_path).split('.')[0]
            save_path = os.path.join(save_root, image_name + '.png')

            if crop_root is not None:
                crop_save_path = os.path.join(crop_root, image_name + '.png')
            else:
                crop_save_path = None

            # 若重建图像和裁剪图像都已存在，则跳过
            if os.path.exists(save_path):
                if (crop_root is not None and os.path.exists(crop_save_path)) or crop_root is None:
                    continue
            
            try:
                # 调用func函数生成重建图像
                func(image_path, save_path, crop_save_path, step=step, max_size=1024)
            except Exception as e:
                # 若生成失败，失败数量加1并打印失败信息和错误详情
                failed_num += 1
                print(f'[ERROR] Failed to generate image in {image_path}. Reason: {e}')

        print(f"[SUCCESS] Inference finished for model_id: {model_name}. Failed images: {failed_num}\n")

    print(f"\n{'='*30}\nAll GenImage models have been processed.\n{'='*30}")

