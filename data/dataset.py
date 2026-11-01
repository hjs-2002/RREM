import warnings

warnings.filterwarnings("ignore")
from torchvision import transforms
import cv2
import numpy as np
import os
import torch
import glob
import json
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage.filters import gaussian_filter
try:
    from transform import create_train_transforms, create_val_transforms, create_sdie_transforms
except:
    from .transform import create_train_transforms, create_val_transforms, create_sdie_transforms

CLASS2LABEL_MAPPING = {
    'real': 0,
    'ldm-text2im-large-256': 1,
    'stable-diffusion-v1-4': 2,
    'stable-diffusion-v1-5': 3,
    'stable-diffusion-2-1': 4,
    'stable-diffusion-xl-base-1.0': 5,
    'stable-diffusion-xl-refiner-1.0': 6,
    'sd-turbo': 7,
    'sdxl-turbo': 8,
    'lcm-lora-sdv1-5': 9,
    'lcm-lora-sdxl': 10,
    'sd-controlnet-canny': 11,
    'sd21-controlnet-canny': 12,
    'controlnet-canny-sdxl-1.0': 13,
    'stable-diffusion-inpainting': 14,
    'stable-diffusion-2-inpainting': 15,
    'stable-diffusion-xl-1.0-inpainting-0.1': 16,
}
LABEL2CLASS_MAPPING = {CLASS2LABEL_MAPPING.get(key): key for key in CLASS2LABEL_MAPPING.keys()}
GenImage_LIST = ['stable_diffusion_v_1_4/imagenet_ai_0419_sdv4', 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5',
                 'Midjourney/imagenet_midjourney', 'ADM/imagenet_ai_0508_adm', 'wukong/imagenet_ai_0424_wukong',
                 'glide/imagenet_glide', 'VQDM/imagenet_ai_0419_vqdm', 'BigGAN/imagenet_ai_0419_biggan']

FF_LIST = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
DiffusionForensics_list = ['adm','ddpm','iddpm','pndm','sdv2','ldm','vqdiffusion']
Chameleon = ['Chameleon']

DiffusionForensics_LIST_sd14 = [
    
    'adm',
    'dalle2',
    'ddpm',
    'diff-projectedgan',
    'diff-stylegan',
    'iddpm',
    'if',
    'ldm',
    'midjourney',
    'pndm',
    'projectedgan',
    'sdv2',
    'stylegan_official',
    'vqdiffusion'
]

DiffusionForensics_LIST_fire_imagenet = [
    
    'adm',
    'sdv1'
]
def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def apply_post_aug(img, post_aug_mode=None):
    """
    img: numpy.ndarray, RGB
    post_aug_mode:
        None
        "jpeg_60", "jpeg_70", ..., "jpeg_100"
        "scale_0.5", "scale_0.75", ..., "scale_1.5"
    """
    if post_aug_mode is None or str(post_aug_mode).lower() == "none":
        return img

    if post_aug_mode.startswith("jpeg_"):
        qf = int(post_aug_mode.split("_")[1])
        img = cv2_jpg(img, qf)
        return img

    if post_aug_mode.startswith("scale_"):
        scale = float(post_aug_mode.split("_")[1])
        img = cv2_scale(img, scale)
        return img

    raise ValueError(f"Unsupported post_aug_mode: {post_aug_mode}")
def cv2_scale(img, scale):
    h, w = img.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))

    return resized_img


def resize_long_size(img, long_size=512):
    scale_percent = long_size / max(img.shape[0], img.shape[1])

    new_width = int(img.shape[1] * scale_percent)
    new_height = int(img.shape[0] * scale_percent)

    img_resized = cv2.resize(img, (new_width, new_height))

    return img_resized


def read_image(image_path, resize_size=None):
    try:
        image = cv2.imread(image_path)
        if resize_size is not None:
            image = resize_long_size(image, long_size=resize_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        
    except:
        print(f'{image_path} read error!!!')
        return np.zeros(shape=(512, 512, 3), dtype=np.uint8)


def align_pair_image_shape(image, rec_image, image_path=None, rec_image_path=None):
    image_h, image_w = image.shape[:2]
    rec_h, rec_w = rec_image.shape[:2]
    if (image_h, image_w) == (rec_h, rec_w):
        return rec_image

    rec_image = cv2.resize(rec_image, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    if image_path is not None and rec_image_path is not None:
        print(
            f"[dataset] resized rec_image from {(rec_h, rec_w)} to {(image_h, image_w)} "
            f"for pair: {os.path.basename(image_path)} <-> {os.path.basename(rec_image_path)}"
        )
    return rec_image


def shuffle_two_array(a, b, seed=None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(b)
    return a, b


def one_hot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def split_data(image_paths, labels, val_split=0.1, test_split=0.0, phase='train', seed=2022):
    """
    根据指定的划分比例和阶段，将图像路径列表和对应的标签列表划分为训练集、验证集或测试集。

    :param image_paths: 包含所有图像文件路径的列表。
    :param labels: 与图像路径列表对应的标签列表。
    :param val_split: 验证集在总数据中的比例，默认为 0.1。
    :param test_split: 测试集在总数据中的比例，默认为 0.0。
    :param phase: 当前划分阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
    :param seed: 随机数种子，用于确保数据打乱的可重复性，默认为 2022。
    :return: 划分后的图像路径列表和对应的标签列表。
    """
    image_paths, labels = shuffle_two_array(image_paths, labels, seed=seed)
    total_len = len(image_paths)
    if test_split > 0:
        if phase == 'train':
            start_index, end_index = 0, int(total_len * (1 - val_split - test_split))
        elif phase == 'val':
            start_index, end_index = int(total_len * (1 - val_split - test_split)), int(total_len * (1 - test_split))
        else:
            start_index, end_index = int(total_len * (1 - test_split)), total_len
    else:
        if phase == 'train':
            start_index, end_index = 0, int(total_len * (1 - val_split))
        else:
            start_index, end_index = int(total_len * (1 - val_split)), total_len
    image_paths, labels = image_paths[start_index:end_index], labels[start_index:end_index]

    return image_paths, labels


def split_dir(image_dirs, val_split=0.1, phase='train', seed=2022):
    if phase == 'all':
        return image_dirs
    image_dirs, _ = shuffle_two_array(image_dirs, image_dirs, seed=seed)
    total_len = len(image_dirs)
    if phase == 'train':
        start_index, end_index = 0, int(total_len * (1 - val_split * 2))
    elif phase == 'val':
        start_index, end_index = int(total_len * (1 - val_split * 2)), int(total_len * (1 - val_split))
    else:
        start_index, end_index = int(total_len * (1 - val_split)), total_len
    image_dirs = image_dirs[start_index:end_index]

    return image_dirs


def find_images(dir_path, extensions=['.jpg', '.png', '.jpeg', '.bmp']):
    image_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if os.path.basename(file).startswith("._"):
                continue
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))

    return image_files


def calculate_dire(img, sdir_img, is_success=True, input_size=224, phase='train'):
    if not is_success:
        return torch.zeros(size=(3, input_size, input_size), dtype=torch.float32)
    sdie_transforms = create_sdie_transforms(size=input_size, phase=phase)

    data = sdie_transforms(image=img, rec_image=sdir_img)
    img, sdir_img = data['image'], data['rec_image']

    img = img / 127.5 - 1
    sdir_img = sdir_img / 127.5 - 1
    sdie = np.abs(img - sdir_img)
    sdie_tensor = torch.from_numpy(np.transpose(np.array(sdie, dtype=np.float32), [2, 0, 1]))

    return sdie_tensor


def get_class_name_by_path(image_path):
    if 'GenImage' in image_path:
        class_names = GenImage_LIST
        class_name = class_names[0]
        for name in class_names[1:]:
            if f'/{name}/' in image_path:
                class_name = name
                break
    else:
        class_name = 'real'
        class_names = list(CLASS2LABEL_MAPPING.keys())
        for name in class_names:
            if f'/{name}/' in image_path:
                class_name = name
                break

    return class_name


def load_DRCT_2M(real_root_path='dataset/MSCOCO',
                 fake_root_path='dataset',
                 fake_indexes='1,2,3,4,5,6', phase='train', val_split=0.1,
                 seed=2022):
    """
    加载 DRCT-2M 数据集，根据不同阶段（训练、验证、测试）划分真实和伪造图像数据。

    :param real_root_path: 真实图像的根目录，默认为 'dataset/MSCOCO'。
    :param fake_root_path: 伪造图像的根目录，默认为 'dataset'。
    :param fake_indexes: 伪造图像类别的索引，多个索引用逗号分隔，默认为 '1,2,3,4,5,6'。
    :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
    :param val_split: 验证集的划分比例，默认为 0.1。
    :param seed: 随机数种子，用于数据划分，默认为 2022。
    :return: 包含所有图像路径的列表和对应的标签列表。
    """
    fake_indexes = [int(index) for index in fake_indexes.split(',')]
    if phase != 'test':  
        real_paths = sorted(glob.glob(f"{real_root_path}/train2017/*.*"))
        real_labels = [0 for _ in range(len(real_paths))]
        real_paths, real_labels = split_data(real_paths, real_labels, val_split=val_split, phase=phase, seed=seed)
        fake_paths = []
        fake_labels = []
        for i, index in enumerate(fake_indexes):
            fake_paths_t = sorted(glob.glob(f"{fake_root_path}/{LABEL2CLASS_MAPPING[index]}/train2017/*.*"))
            fake_labels_t = [i + 1 for _ in range(len(fake_paths_t))]
            fake_paths_t, fake_labels_t = split_data(fake_paths_t, fake_labels_t, val_split=val_split, phase=phase,
                                                     seed=seed)
            fake_paths += fake_paths_t
            fake_labels += fake_labels_t
    else:  
        real_paths = sorted(glob.glob(f"{real_root_path}/val2017/*.*"))
        real_labels = [0 for _ in range(len(real_paths))]
        fake_paths = []
        fake_labels = []
        for i, index in enumerate(fake_indexes):
            fake_paths_t = sorted(glob.glob(f"{fake_root_path}/{LABEL2CLASS_MAPPING[index]}/val2017/*.*"))
            fake_labels_t = [i + 1 for _ in range(len(fake_paths_t))]
            fake_paths += fake_paths_t
            fake_labels += fake_labels_t
    image_paths = real_paths + fake_paths
    labels = real_labels + fake_labels

    class_count_mapping = {cls: 0 for cls in range(len(fake_indexes) + 1)}
    for label in labels:
        class_count_mapping[label] += 1
    class_name_mapping = {0: 'real'}
    for i, fake_index in enumerate(fake_indexes):
        class_name_mapping[i + 1] = LABEL2CLASS_MAPPING[fake_index]
    print(f"{phase}:{class_count_mapping}, total:{len(image_paths)}, class_name_mapping:{class_name_mapping}")

    return image_paths, labels


def load_normal_data(root_path, val_split, seed, phase='train', regex='*.*', test_all=False):
    """
    加载指定目录下的图像数据及其描述。

    :param root_path: 图像数据的根目录。
    :param val_split: 验证集的划分比例。
    :param seed: 随机数种子，用于数据划分。
    :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
    :param regex: 用于过滤图像文件的正则表达式，默认为 '*.*'，表示匹配所有文件。
    :param test_all: 是否在测试阶段加载所有数据，默认为 False。
    :return: 包含图像文件路径的列表和对应的描述列表。
    """
    images_t = sorted(glob.glob(f'{root_path}/{regex}'))
    if not test_all:
        images_t, _ = split_data(images_t, images_t, val_split=val_split, phase=phase, seed=seed)

    captions_t = [' ' for _ in images_t]
    print(f'{root_path}: {len(images_t)}')
    return images_t, captions_t


def load_GenImage(root_path='/disk1/chenby/dataset/AIGC_data/GenImage', phase='train', seed=2023,
                  indexes='1,2,3,4,5,6,7,8', val_split=0.1):
    """
    加载 GenImage 数据集，根据指定的阶段和索引选择相应的数据。

    :param root_path: GenImage 数据集的根目录，默认为 '/disk1/chenby/dataset/AIGC_data/GenImage'。
    :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
    :param seed: 随机数种子，用于数据划分，默认为 2023。
    :param indexes: 要加载的数据集目录索引，多个索引用逗号分隔，默认为 '1,2,3,4,5,6,7,8'。
    :param val_split: 验证集的划分比例，默认为 0.1。
    :return: 包含所有图像文件路径的列表和对应的标签列表。
    """
    indexes = [int(i) - 1 for i in indexes.split(',')]
    dir_list = GenImage_LIST
    selected_dir_list = [dir_list[i] for i in indexes]
    real_images, real_labels, fake_images, fake_labels = [], [], [], []
    dir_phase = 'train' if phase != 'test' else 'val'
    for i, selected_dir in enumerate(selected_dir_list):
        real_root = os.path.join(root_path, selected_dir, dir_phase, 'nature')
        fake_root = os.path.join(root_path, selected_dir, dir_phase, 'ai')
        real_images_t = sorted(glob.glob(f'{real_root}/*.*'))
        fake_images_t = sorted(glob.glob(f'{fake_root}/*.*'))
        if phase != 'test':
            real_images_t, _ = split_data(real_images_t, real_images_t, val_split, phase=phase, seed=seed)
            fake_images_t, _ = split_data(fake_images_t, fake_images_t, val_split, phase=phase, seed=seed)
        real_images += real_images_t
        real_labels += [0 for _ in real_images_t]
        fake_images += fake_images_t
        fake_labels += [i + 1 for _ in fake_images_t]
        print(f'phase:{phase}, real:{len(real_images_t)}, fake-{i+1}:{len(fake_images_t)}, selected_dir:{selected_dir}')
    total_images = real_images + fake_images
    labels = real_labels + fake_labels
    print(f'phase:{phase}, real:{len(real_images)}, fake:{len(fake_images)}')

    return total_images, labels


def load_data(real_root_path, fake_root_path,
              phase='train', val_split=0.1, seed=2022, ):
    """
    加载真实和伪造图像数据及其标签。

    :param real_root_path: 真实图像的根目录，多个目录用逗号分隔。
    :param fake_root_path: 伪造图像的根目录，多个目录用逗号分隔。
    :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
    :param val_split: 验证集的划分比例，默认为 0.1。
    :param seed: 随机数种子，用于数据划分，默认为 2022。
    :return: 包含所有图像路径的列表和对应的标签列表。
    """
    total_real_images, total_real_captions = [], []
    for real_root in real_root_path.split(','):
        real_images_t, real_captions_t = load_normal_data(real_root, val_split, seed, phase)
        total_real_images += list(real_images_t)
        total_real_captions += list(real_captions_t)
    total_fake_images, total_fake_captions = [], []
    for fake_root in fake_root_path.split(','):
        fake_images_t, fake_captions_t = load_normal_data(fake_root, val_split, seed, phase)
        total_fake_images += list(fake_images_t)
        total_fake_captions += list(fake_captions_t)
    image_paths = total_real_images + total_fake_images
    labels = [0 for _ in total_real_images] + [1 for _ in total_fake_images]
    print(f'{phase}-total:{len(image_paths)}, real:{len(total_real_images)},fake:{len(total_fake_images)}')

    return image_paths, labels


def load_pair_data(root_path, fake_root_path=None, phase='train', seed=2023, fake_indexes='1',
                   inpainting_dir='full_inpainting'):
    if 'MSCOCO' in root_path and phase in ['train', 'val']:
        phase_mapping = {'train': 'train2017', 'val': 'train2017', 'test': 'val2017'}
        real_root, real_rec_root = root_path.split(',')[:2]
        

        print(f'real_root:{real_root}, real_rec_root:{real_rec_root}')
        fake_root, fake_rec_root = fake_root_path.split(',')[:2]
        
        print(f'fake_root:{fake_root}, fake_rec_root:{fake_rec_root}')


    if 'MSCOCO' in root_path and phase == 'test':
        phase_mapping = {'train': 'train2017', 'val': 'train2017', 'test': 'val2017'}
        real_root, real_rec_root = root_path.split(',')[:2]
        

        print(f'real_root:{real_root}, real_rec_root:{real_rec_root}')
        fake_root, fake_rec_root = fake_root_path.split(',')[:2]
        fake_root = f'{fake_root}/{LABEL2CLASS_MAPPING[int(fake_indexes)]}/{phase_mapping[phase]}'
        fake_rec_root = f'{fake_rec_root}/{LABEL2CLASS_MAPPING[int(fake_indexes)]}/{inpainting_dir}/{phase_mapping[phase]}'
    if 'DR/GenImage' in root_path:
        phase_mapping = {'train': 'train', 'val': 'train', 'test': 'val'}
        fake_indexes = int(fake_indexes)
        fake_name = GenImage_LIST[fake_indexes-1]
        real_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/nature/crop'
        real_rec_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/nature/{inpainting_dir}'
        fake_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/ai/crop'
        fake_rec_root = f'{root_path}/{fake_name}/{phase_mapping[phase]}/ai/{inpainting_dir}'
        print(f'fake_name:{fake_name}')
    if 'DiffusionForensics' in root_path and phase == 'test':
        phase_mapping = {'train': 'train2017', 'val': 'train2017', 'test': 'val2017'}
        real_root, real_rec_root = root_path.split(',')[:2]
        

        print(f'real_root:{real_root}, real_rec_root:{real_rec_root}')
        fake_indexes = int(fake_indexes)
        fake_root, fake_rec_root = fake_root_path.split(',')[:2]
        fake_root = f'{fake_root}/{DiffusionForensics_LIST_sd14[fake_indexes-1]}'
        fake_rec_root = f'{fake_rec_root}/{DiffusionForensics_LIST_sd14[fake_indexes-1]}/{inpainting_dir}'
    if 'FIRE' in root_path and (phase == 'train' or phase == 'val'):
        phase_mapping = {'train': 'train', 'val': 'train', 'test': 'test'}
        real_root, real_rec_root = root_path.split(',')[:2]

        real_root = f'{real_root}/{phase_mapping[phase]}/imagenet/real/'
        real_rec_root = f'{real_rec_root}/{phase_mapping[phase]}/imagenet/real'
        print(f'real_root:{real_root}, real_rec_root:{real_rec_root}')
        fake_root, fake_rec_root = fake_root_path.split(',')[:2]
        fake_root = f'{fake_root}/{phase_mapping[phase]}/imagenet/adm/'
        fake_rec_root = f'{fake_rec_root}/{phase_mapping[phase]}/imagenet/adm'
            
        print(f'fake_root:{fake_root}, fake_rec_root:{fake_rec_root}')
    if 'FIRE' in root_path and phase == 'test':
        phase_mapping = {'train': 'train', 'val': 'train', 'test': 'test'}
        real_root, real_rec_root = root_path.split(',')[:2]
        real_root = f'{real_root}/{phase_mapping[phase]}/lsun_bedroom/real/'
        real_rec_root = f'{real_rec_root}/{phase_mapping[phase]}/lsun_bedroom/real'

        print(f'real_root:{real_root}, real_rec_root:{real_rec_root}')
        fake_indexes = int(fake_indexes)
        fake_root, fake_rec_root = fake_root_path.split(',')[:2]
        fake_root = f'{fake_root}/{phase_mapping[phase]}/lsun_bedroom/{DiffusionForensics_LIST_sd14[fake_indexes-1]}'
        fake_rec_root = f'{fake_rec_root}/{phase_mapping[phase]}/lsun_bedroom/{DiffusionForensics_LIST_sd14[fake_indexes-1]}'
        print(f'fake_root:{fake_root}, fake_rec_root:{fake_rec_root}')
        
    if 'Chameleon'  in root_path and (phase == 'test'):
        phase_mapping = {'train': 'train', 'val': 'train', 'test': 'test'}
        real_root, real_rec_root = root_path.split(',')[:2]
            

        print(f'real_root:{real_root}, real_rec_root:{real_rec_root}')
        fake_root, fake_rec_root = fake_root_path.split(',')[:2]
            
        print(f'fake_root:{fake_root}, fake_rec_root:{fake_rec_root}')
    image_paths, labels = [], []
    percent = 1
   
    if 'FIRE' in root_path and ('imagenet' in real_root ): 
        real_image_paths = sorted(glob.glob(os.path.join(real_root, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))   
        real_image_rec_paths = sorted(glob.glob(os.path.join(real_rec_root, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
       
    else:
        real_image_paths = sorted(glob.glob(f"{real_root}/*.*"))
        real_image_rec_paths = sorted(glob.glob(f"{real_rec_root}/*.*"))

    real_image_paths = real_image_paths[:int(len(real_image_paths) * percent)]
    real_image_rec_paths = real_image_rec_paths[:int(len(real_image_rec_paths) * percent)]
    print(f'Real data-{phase}:{len(real_image_paths)}.')
    print(f'Real rec data-{phase}:{len(real_image_rec_paths)}.')
    
    assert len(real_image_paths) == len(real_image_rec_paths) and len(real_image_paths) > 0
    total_real = len(real_image_paths)
   
    if phase != 'test' and 'DiffusionForensics' not in root_path:
        real_image_paths, real_image_rec_paths = split_data(real_image_paths, real_image_rec_paths, phase=phase, seed=seed)
    for real_image_path, real_image_rec_path in zip(real_image_paths, real_image_rec_paths):
        image_paths.append((real_image_path, real_image_rec_path))
    if 'FIRE' in root_path and ('imagenet' in fake_root ):    
        fake_image_paths = sorted(glob.glob(os.path.join(fake_root, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))   
        fake_image_rec_paths = sorted(glob.glob(os.path.join(fake_rec_root, "*", "*.[jJp][pPEn]*[gG]"), recursive=True))
        
    else:
        fake_image_paths = sorted(glob.glob(f"{fake_root}/*.*"))
        fake_image_rec_paths = sorted(glob.glob(f"{fake_rec_root}/*.*"))
    
    fake_image_paths = fake_image_paths[:int(len(fake_image_paths) * percent)]
    fake_image_rec_paths = fake_image_rec_paths[:int(len(fake_image_rec_paths) * percent)]
    print(f'Fake data-{phase}:{len(fake_image_paths)}.')
    print(f'Fake rec data-{phase}:{len(fake_image_rec_paths)}.')
   
    assert len(fake_image_paths) == len(fake_image_rec_paths) and len(fake_image_paths) > 0
    total_fake = len(fake_image_paths)
    if phase != 'test' and 'DiffusionForensics' not in root_path:
        fake_image_paths, fake_image_rec_paths  = split_data(fake_image_paths, fake_image_rec_paths, phase=phase, seed=seed)
    for fake_image_path, fake_image_rec_path in zip(fake_image_paths, fake_image_rec_paths):
        image_paths.append((fake_image_path, fake_image_rec_path))

    required_models = [
        'stable-diffusion-inpainting',
        'stable-diffusion-2-inpainting', 
        'stable-diffusion-xl-1.0-inpainting-0.1'
    ]

    labels = [0 for _ in range(len(real_image_paths))] + [1 for _ in range(len(fake_image_paths))]
    print(f'Phase:{phase}, real:{len(real_image_paths)}, rec real:{len(real_image_rec_paths)}, fake:{len(fake_image_paths)},'
          f' rec fake:{len(fake_image_rec_paths)}')
    print(f'前五个样本:{image_paths[:5]}，前五个标签:{labels[:5]}')
    print(f"最后5个样本：{image_paths[-5:]}, 最后5个标签:{labels[-5:]}")
    return image_paths, labels

def load_ff_images(root_path='/home/law/HDD/serein/Dataset/FF++', phase='train', seed=2023,
                  indexes='1,2,3,4', val_split=0.1):
    """
    加载 test 目录下 Deepfakes、Face2Face、FaceSwap、NeuralTextures 目录的所有伪造图片（标签为1），
    以及 original 目录下的所有真实图片（标签为0）。

    :param root_path: test 数据集的根目录
    :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'
    :param seed: 随机数种子，默认为 2023
    :param indexes: 索引字符串，默认为 '1,2,3,4,5,6,7,8'
    :param val_split: 验证集划分比例，默认为 0.1
    :return: 包含所有图像文件路径的列表和对应的标签列表
    """
    indexes = [int(i) - 1 for i in indexes.split(',')]
    fake_dirs = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    real_dir = 'original'
    selected_dir_list = [fake_dirs[i] for i in indexes]
    exts = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']

    fake_images, fake_labels = [], []
    for fake_dir in selected_dir_list:
        fake_dir_path = os.path.join(root_path, phase, fake_dir)
        print(f'fake_dir_path:{fake_dir_path}')
        count = 0
        
        for ext in exts:
            fake_images_t = sorted(glob.glob(os.path.join(fake_dir_path, '**', f'*.{ext}'), recursive=True))
            print(f'fake_images_t:{len(fake_images_t)}')
            if phase != 'test':
                fake_images_t, _ = split_data(fake_images_t, fake_images_t, val_split, phase=phase, seed=seed)
              
            fake_images += fake_images_t*4
            fake_labels += [1] * len(fake_images)
            count += len(fake_images_t)
        print("fake_images:", fake_images)
        print("fake_labels:", fake_labels)
        print(f'fake_dir:{fake_dir}, fake_count:{count}')
       
    real_images, real_labels = [], []
    real_dir_path = os.path.join(root_path, phase, real_dir)
    count = 0
    for ext in exts:
        real_images_t = sorted(glob.glob(os.path.join(real_dir_path, '**', f'*.{ext}'), recursive=True))
        if phase != 'test':
            real_images_t, _ = split_data(real_images_t, real_images_t, val_split, phase=phase, seed=seed)
        real_images += real_images_t
        real_labels += [0] * len(real_images_t)
        count += len(real_images_t)
    print(f'real_dir:{real_dir}, real_count:{count}')

    total_images = real_images + fake_images
    labels = real_labels + fake_labels
    print(f'total real:{len(real_images)}, total fake:{len(fake_images)}, total:{len(total_images)}')

    return total_images, labels
def _convert_to_rgb(image):
    return image.convert('RGB')

class AIGCDetectionDataset(Dataset):
    def __init__(self, root_path='dataset/MSCOCO', fake_root_path='/disk4/chenby/dataset/DRCT-2M',
                 fake_indexes='1,2,3,4,5,6', phase='train', is_one_hot=False, seed=2021,
                 transform=None, use_label=True, num_classes=None, regex='*.*',
                 is_dire=False, inpainting_dir='full_inpainting', post_aug_mode=None):
        """
        初始化 AIGCDetectionDataset 类的实例。

        :param root_path: real 图像的根目录，默认为 'dataset/MSCOCO'。
        :param fake_root_path: fake 图像的根目录，默认为 '/disk4/chenby/dataset/DRCT-2M'。
        :param fake_indexes: 假图像类别的索引，多个类别用逗号分隔，默认为 '1,2,3,4,5,6'。
        :param phase: 当前阶段，可选值为 'train', 'val', 'test'，默认为 'train'。
        :param is_one_hot: 标签是否使用 one-hot 编码，默认为 False。
        :param seed: 随机数种子，用于数据划分，默认为 2021。
        :param transform: 图像数据预处理的转换函数，默认为 None。
        :param use_label: 是否使用标签，默认为 True。
        :param num_classes: 分类的类别数量，若为 None 则根据 fake_indexes 计算，默认为 None。
        :param regex: 数据过滤的正则表达式，默认为 '*.*'。
        :param is_dire: 是否使用 DIRE（Differential Representation）训练，默认为 False。
        :param inpainting_dir: 图像修复目录，默认为 'full_inpainting'。
        :param post_aug_mode: 抗后处理测试模式，例如 'blur_1', 'jpeg_30' 等，默认为 None。
        """
        self.root_path = root_path  
        self.phase = phase
        self.is_one_hot = is_one_hot
        self.num_classes = len(fake_indexes.split(',')) + 1 if num_classes is None else num_classes
        self.transform = transform
      
        self.use_label = use_label
        self.regex = regex  
        self.is_dire = is_dire  
        self.post_aug_mode = post_aug_mode
        self.seed = seed

        if use_label:
            if self.is_dire:
                self.image_paths, self.labels = load_pair_data(root_path, fake_root_path, phase,
                                                               fake_indexes=fake_indexes,
                                                               inpainting_dir=inpainting_dir)
            elif 'MSCOCO' in root_path and len(fake_root_path.split(',')) == 1:
                self.image_paths, self.labels = load_DRCT_2M(real_root_path=root_path,
                                                             fake_root_path=fake_root_path,
                                                             fake_indexes=fake_indexes, phase=phase, seed=seed)
            elif 'FF++' in root_path:
                print(f'FF++ root_path:{root_path}, phase:{phase}, seed:{seed}, fake_indexes:{fake_indexes}')
                self.image_paths,self.labels = load_ff_images(root_path=root_path, phase=phase, seed=seed,
                                                              indexes=fake_indexes)
                
             
            else:
                self.image_paths, self.labels = load_pair_data(root_path, fake_root_path, phase,
                                                               fake_indexes=fake_indexes,
                                                               inpainting_dir=inpainting_dir)   

            self.labels = [int(label > 0) for label in self.labels] if self.num_classes == 2 else self.labels
        else:
            if len(root_path.split(',')) == 2 and 'DR' in root_path:
                self.is_dire = True
                self.image_paths = load_pair_data(root_path, phase=phase, fake_indexes=fake_indexes,
                                                  inpainting_dir=inpainting_dir)
            else:
                if self.regex == 'all':
                    self.image_paths = sorted(find_images(dir_path=root_path, extensions=['.jpg', '.png', '.jpeg', '.bmp']))
                else:
                    self.image_paths = sorted(glob.glob(f'{root_path}/{self.regex}'))[:]
            print(f'Total predict images:{len(self.image_paths)}, regex:{self.regex}')
        if self.phase == 'test' and self.post_aug_mode is not None:
            print(f"post_aug_mode:{self.post_aug_mode}, {self.post_aug_mode.split('_')[1]}")

    def __len__(self):
        return len(self.image_paths)

    def get_labels(self):
        return list(self.labels)
    def _load_image(self, path: str) -> Image.Image:
        if 'npy' in path[-3:]:
            img = np.load(path)
            return img
        return Image.open(path)
    def __getitem__(self, index):
        if not self.is_dire:
            
            image_path, rec_image_path = self.image_paths[index]
           
            image = read_image(image_path)
            rec_image = read_image(rec_image_path)
            rec_image = align_pair_image_shape(image, rec_image, image_path, rec_image_path)
          
            if self.phase == 'test' and self.post_aug_mode is not None:
                if 'jpeg' in self.post_aug_mode:
                    compress_val = int(self.post_aug_mode.split('_')[1])
                    image = cv2_jpg(image, compress_val)
                    rec_image = cv2_jpg(rec_image, compress_val)

                elif 'scale' in self.post_aug_mode:
                    scale = float(self.post_aug_mode.split('_')[1])
                    image = cv2_scale(image, scale)
                    rec_image = cv2_scale(rec_image, scale)

          
            transformed = self.transform(image=image, rec_image=rec_image)
            image = transformed['image']
            rec_image = transformed['rec_image']
        else:
            image_path, rec_image_path = self.image_paths[index]
            image, is_success = read_image(image_path)
            rec_image, rec_is_success = read_image(rec_image_path)
            is_success = is_success and rec_is_success
            image = calculate_dire(image, rec_image, phase=self.phase)

        label = 0
        if self.use_label:
            label = self.labels[index] 
            if label == 0:
                type = 'real'
            else :
                type = 'fake'

      
        image_list = [image, rec_image]
        if not self.use_label:
            return image, image_path.replace(f"{self.root_path}", '')

        if self.is_one_hot:
            label = one_hot(self.num_classes, label)

        return image_list, type, label
