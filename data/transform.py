import os
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from torchvision import transforms
import torch
import albumentations as A
import math
from PIL import Image
current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录

class SpectrumNormalize(ImageOnlyTransform):
    """Spectrum Normalization
    """
    def __init__(self, always_apply=False, p=1.0):
        super(SpectrumNormalize, self).__init__(always_apply, p)

    def apply(self, image, **params):
        normalized_spectrum = self.extract_spectrum(image)

        return normalized_spectrum

    @staticmethod
    def extract_spectrum(image):
        image_float32 = np.float32(image)
        x = transforms.ToTensor()(image_float32)
        x_freq = torch.fft.fft2(x)
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
        out = np.transpose(x_freq.abs().numpy(), (1, 2, 0))  # 幅度谱
        out = cv2.normalize(out, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return out


class DoNothing(ImageOnlyTransform):
    """Do nothing"""
    def __init__(self, always_apply=False, p=1.0):
        super(DoNothing, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return image



def create_train_transforms(size=300, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                            is_crop=False,):
    """
    创建用于训练阶段的图像增强变换组合。

    :param size: 图像的目标尺寸，默认为 300。
    :param mean: 归一化使用的均值，默认为 ImageNet 的均值。
    :param std: 归一化使用的标准差，默认为 ImageNet 的标准差。
    :param is_crop: 是否进行裁剪操作，默认为 False。
    :return: 包含一系列图像增强变换的 albumentations.Compose 对象。
    """
    # 根据 is_crop 参数选择不同的尺寸调整方法
    # 若 is_crop 为 True，使用随机裁剪；否则，将图像的最长边调整为指定尺寸
    resize_fuc = A.RandomCrop(height=size, width=size) if is_crop else A.LongestMaxSize(max_size=size)
    # 定义一系列训练时使用的图像增强操作
    aug_hard = [
        # 图像压缩，随机调整图像质量，质量范围在 30 到 100 之间，有 0.5 的概率应用
        A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
        # 随机缩放图像，缩放比例在 -0.5 到 0.5 之间，有 0.2 的概率应用
        A.RandomScale(scale_limit=(-0.5, 0.5), p=0.2),  # 23/11/04 add
        # 水平翻转图像，有 0.5 的概率应用
        A.HorizontalFlip(),
        # 添加高斯噪声，有 0.1 的概率应用
        A.GaussNoise(p=0.1),
        # 高斯模糊，有 0.1 的概率应用
        A.GaussianBlur(p=0.1),
        # 随机旋转 90 度的倍数，有 0.5 的概率应用
        A.RandomRotate90(),
        # 若 is_crop 为 True，将图像填充到指定尺寸；否则，不进行操作
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0) if is_crop else DoNothing(),
        # 尺寸调整操作
        resize_fuc,
        # 将图像填充到指定尺寸
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        # 从随机亮度对比度调整、FancyPCA、色调饱和度值调整中随机选择一个操作，有 0.5 的概率应用
        A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.5),
        # 从粗粒度丢弃和网格丢弃中随机选择一个操作，有 0.5 的概率应用
        A.OneOf([A.CoarseDropout(), A.GridDropout()], p=0.5),
        # 将图像转换为灰度图，有 0.2 的概率应用
        A.ToGray(p=0.2),
        # 平移、缩放和旋转操作，有 0.5 的概率应用
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        # 使用指定的均值和标准差对图像进行归一化
        A.Normalize(mean=mean, std=std),
        # 将图像转换为 PyTorch 张量
        ToTensorV2()
    ]
    
    # 将上述增强操作组合起来，并为额外目标 'rec_image' 应用相同的变换
    return A.Compose(aug_hard, additional_targets={'rec_image': 'image'})


def create_val_transforms(size=300, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_crop=False):
    # resize_fuc = A.CenterCrop(height=size, width=size) if is_crop else A.Resize(height=size, width=size)
    resize_fuc = A.CenterCrop(height=size, width=size) if is_crop else A.LongestMaxSize(max_size=size)
    return A.Compose([
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0) if is_crop else DoNothing(),
        resize_fuc,
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], additional_targets={'rec_image': 'image'})

def translate_duplicate(img, cropSize):
    """
    将图像进行平移复制，以确保图像的最小维度不小于cropSize。
    
    如果图像的最小维度（宽度或高度）小于cropSize，则在图像的右侧和下方重复粘贴原图像，
    直到新图像的宽度和高度都不小于cropSize。如果图像的最小维度不小于cropSize，则直接返回原图像。
    
    参数:
    img (PIL.Image.Image or np.ndarray): 输入的图像对象。
    cropSize (int): 期望的最小裁剪尺寸。
    
    返回:
    PIL.Image.Image or np.ndarray: 平移复制后的图像对象。
    """
    # 检查输入是PIL图像还是numpy数组
    if isinstance(img, np.ndarray):
        # 处理numpy数组格式的图像 (height, width, channels)
        height, width = img.shape[:2]
        # 检查图像的最小维度是否小于cropSize
        if min(width, height) < cropSize:
            # 计算新图像的宽度和高度，确保它们不小于cropSize
            new_width = width * math.ceil(cropSize/width)
            new_height = height * math.ceil(cropSize/height)
            
            # 创建一个新的图像数组，尺寸为计算出的新宽度和新高度
            if len(img.shape) == 3:
                new_img = np.zeros((new_height, new_width, img.shape[2]), dtype=img.dtype)
            else:
                new_img = np.zeros((new_height, new_width), dtype=img.dtype)
            
            # 在新图像上重复粘贴原图像
            for i in range(0, new_width, width):
                for j in range(0, new_height, height):
                    w_end = min(i + width, new_width)
                    h_end = min(j + height, new_height)
                    new_img[j:h_end, i:w_end] = img[0:h_end-j, 0:w_end-i]
            
            # 返回平移复制后的新图像
            # print(f"Image resized from ({width}, {height}) to ({new_width}, {new_height}) for cropping.")
            return new_img
        else:
            # 如果原图像的最小维度不小于cropSize，直接返回原图像
            return img
    else:
        # 处理PIL图像格式
        # 检查图像的最小维度是否小于cropSize
        if min(img.size) < cropSize:
            # 获取原图像的宽度和高度
            width, height = img.size
            
            # 计算新图像的宽度和高度，确保它们不小于cropSize
            new_width = width * math.ceil(cropSize/width)
            new_height = height * math.ceil(cropSize/height)
            
            # 创建一个新的图像对象，尺寸为计算出的新宽度和新高度
            new_img = Image.new('RGB', (new_width, new_height))
            
            # 在新图像上重复粘贴原图像
            for i in range(0, new_width, width):
                for j in range(0, new_height, height):
                    new_img.paste(img, (i, j))
            
            # 返回平移复制后的新图像
            # print(f"Image resized from {img.size} to {new_img.size} for cropping.")
            return new_img
        else:
            # 如果原图像的最小维度不小于cropSize，直接返回原图像
            return img
        
    #这个create_sdie_transforms效果最好
def create_sdie_transforms(size=224, phase='train', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if phase == 'train':
        aug_list = [
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
           
            A.RandomCrop(height=size, width=size),


            # 水平翻转变换 （概率为0.2）垂直翻转变换 （概率为0.2）随机旋转90度变换 （概率为0.2）
        #     A.HorizontalFlip(p=0.2),
        #     A.VerticalFlip(p=0.2),
        #     A.RandomRotate90(p=0.2),

        #     A.ImageCompression(quality_lower=30, quality_upper=100, p=0.5),
        #     A.GaussNoise(p=0.1),
        # # 高斯模糊，有 0.1 的概率应用
        #     A.GaussianBlur(p=0.1),

            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    else:
        aug_list = [
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
            
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    
    return A.Compose(aug_list, additional_targets={'rec_image': 'image'})



def create_sdie_transforms_dup(size=224, phase='train', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    if phase == 'train':
        aug_list = [
            # A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Lambda(lambda x, **kwargs: translate_duplicate(x, 256), mask=None),
            A.RandomCrop(height=size, width=size),
            # A.HorizontalFlip(p=0.2),
            # A.VerticalFlip(p=0.2),
            # A.RandomRotate90(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    else:
        aug_list = [
            # A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Lambda(lambda x, **kwargs: translate_duplicate(x, 256), mask=None),
            A.CenterCrop(height=size, width=size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ]
    
    return A.Compose(aug_list, additional_targets={'rec_image': 'image'})


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = cv2.imread('samples/01.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (128, 128))
    print(image.shape)
    # transform = create_sdie_transforms(size=224, phase='train')
    transform = create_train_transforms(size=512, is_crop=False)
    # transform = create_val_transforms(size=300, is_crop=True)

    data = transform(image=image)
    out = data["image"]






