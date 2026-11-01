import numpy as np
import tensorboardX
from sklearn.metrics import log_loss, accuracy_score, precision_score, average_precision_score, roc_auc_score, \
    recall_score, confusion_matrix
import torch
import torch.nn.functional as F


class Test_time_agumentation(object):
    """
    测试时增强（Test Time Augmentation, TTA）工具类，用于在测试阶段对输入图像进行多种变换，
    包括旋转和翻转，以提高模型预测的准确性。同时提供逆变换方法，可将变换后的图像恢复原状。
    """

    def __init__(self, is_rotation=True):
        """
        初始化 Test_time_agumentation 类。

        :param is_rotation: 是否启用旋转增强，默认为 True。
        """
        self.is_rotation = is_rotation

    def __rotation(self, img):
        """
        对输入图像进行顺时针 90 度、180 度和 270 度旋转。

        :param img: 输入的图像张量。
        :return: 包含旋转后图像的列表，顺序为 [90 度旋转, 180 度旋转, 270 度旋转]。
        """
        img90 = img.rot90(-1, [2, 3])  # 1 表示逆时针旋转； -1 表示顺时针旋转
        img180 = img.rot90(-1, [2, 3]).rot90(-1, [2, 3])
        img270 = img.rot90(1, [2, 3])
        return [img90, img180, img270]

    def __inverse_rotation(self, img90, img180, img270):
        """
        对旋转后的图像进行逆时针 90 度、180 度和 270 度旋转，恢复原始图像方向。

        :param img90: 顺时针旋转 90 度后的图像张量。
        :param img180: 顺时针旋转 180 度后的图像张量。
        :param img270: 顺时针旋转 270 度后的图像张量。
        :return: 恢复后的图像张量元组，顺序为 (90 度逆旋转, 180 度逆旋转, 270 度逆旋转)。
        """
        img90 = img90.rot90(1, [2, 3])  # 1 表示逆时针旋转； -1 表示顺时针旋转
        img180 = img180.rot90(1, [2, 3]).rot90(1, [2, 3])
        img270 = img270.rot90(-1, [2, 3])
        return img90, img180, img270

    def __flip(self, img):
        """
        对输入图像进行垂直和水平翻转。

        :param img: 输入的图像张量。
        :return: 包含翻转后图像的列表，顺序为 [垂直翻转, 水平翻转]。
        """
        return [img.flip(2), img.flip(3)]

    def __inverse_flip(self, img_v, img_h):
        """
        对翻转后的图像进行垂直和水平翻转，恢复原始图像状态。

        :param img_v: 垂直翻转后的图像张量。
        :param img_h: 水平翻转后的图像张量。
        :return: 恢复后的图像张量元组，顺序为 (垂直逆翻转, 水平逆翻转)。
        """
        return img_v.flip(2), img_h.flip(3)

    def tensor_rotation(self, img):
        """
        对输入图像进行旋转增强。

        :param img: 输入的图像张量，尺寸应为 [H, W]。
        :return: 包含旋转后图像的列表，旋转角度为 [90 度, 180 度, 270 度]。
        """
        # assert img.shape == (1024, 1024)
        return self.__rotation(img)

    def tensor_inverse_rotation(self, img_list):
        """
        对旋转后的图像进行逆旋转操作，恢复原始图像方向。

        :param img_list: 包含旋转后图像的列表，顺序为 [90 度旋转, 180 度旋转, 270 度旋转]。
        :return: 包含逆旋转后图像的列表，顺序为 [90 度逆旋转, 180 度逆旋转, 270 度逆旋转]。
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_rotation(img_list[0], img_list[1], img_list[2])

    def tensor_flip(self, img):
        """
        对输入图像进行翻转增强。

        :param img: 输入的图像张量，尺寸应为 [H, W]。
        :return: 包含翻转后图像的列表，顺序为 [垂直翻转, 水平翻转]。
        """
        # assert img.shape == (1024, 1024)
        return self.__flip(img)

    def tensor_inverse_flip(self, img_list):
        """
        对翻转后的图像进行逆翻转操作，恢复原始图像状态。

        :param img_list: 包含翻转后图像的列表，顺序为 [垂直翻转, 水平翻转]。
        :return: 包含逆翻转后图像的列表，顺序为 [垂直逆翻转, 水平逆翻转]。
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_flip(img_list[0], img_list[1])


class AverageMeter(object):
    """
    计算并存储当前值和平均值的工具类。
    常用于在训练或测试过程中跟踪诸如损失值、准确率等指标的平均值。
    """

    def __init__(self):
        # 初始化类的属性，调用 reset 方法将所有属性重置为初始值
        self.reset()

    def reset(self):
        """
        将所有存储指标的属性重置为初始值。
        通常在每个训练周期（epoch）开始时调用。
        """
        self.val = 0  # 当前值，记录最新一次更新的数据值
        self.avg = 0  # 平均值，记录从开始到当前所有数据的平均值
        self.sum = 0  # 总和，记录从开始到当前所有数据的总和
        self.count = 0  # 计数，记录从开始到当前更新数据的次数

    def update(self, val, n=1):
        """
        使用新的数据值更新当前值、总和、计数和平均值。

        :param val: 新的数据值，用于更新指标
        :param n: 新数据值的数量，默认为 1。用于在批量数据更新时正确计算总和
        """
        self.val = val  # 更新当前值为最新的数据值
        self.sum += val * n  # 将新数据值乘以数量 n 后累加到总和中
        self.count += n  # 更新计数，增加 n 次
        self.avg = self.sum / self.count  # 根据更新后的总和和计数重新计算平均值


class Logger(object):
    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter(model_name)

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']

        for col in self.header[1:]:
            self.writer.add_scalar(phase + "/" + col, float(values[col]), int(epoch))


def calculate_metrics(outputs, targets, metric_name='acc'):
    if len(targets.data.numpy().shape) > 1:
        _, targets = torch.max(targets.detach(), dim=1)
    if len(outputs.data.numpy().shape) > 1 and outputs.data.numpy().shape[1] == 1:  # 尾部是sigmoid
        outputs = torch.cat([1 - outputs, outputs], dim=1)

    # print(outputs.shape, targets.shape, pred_labels.size())
    if metric_name == 'acc':
        pred_labels = torch.max(outputs, 1)[1]
        # print(targets, pred_labels)
        return accuracy_score(targets.data.numpy(), pred_labels.detach().numpy())
    elif metric_name == 'auc':
        pred_labels = outputs[:, 1]  # 为假的概率
        return roc_auc_score(targets.data.numpy(), pred_labels.detach().numpy())


def calculate_fnr(y_true, y_pred):
    """
    计算False Negative Rate

    参数:
    y_true -- 真实的类别标签
    y_pred -- 预测的类别标签

    返回:
    fnr -- False Negative Rate
    """
    # 生成混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算FNR
    fnr = fn / float(fn + tp)

    return fnr


def compute_confusion_matric(outputs, targets):
    """
         ｜   预测   ｜
    -----------------
    真｜正｜ TP ｜ FN ｜
       --------------
    值｜负｜ FP ｜ TN ｜
    -----------------
    :param outputs:
    :param targets:
    :return: tp, fp, tn ,fn
    """
    part = outputs ^ targets
    pcount = np.bincount(part)
    tp_list = list(outputs & targets)
    fp_list = list(outputs & ~targets)
    tp = tp_list.count(1)
    fp = fp_list.count(1)
    tn = pcount[0] - tp
    fn = pcount[1] - fp

    return tp, fp, tn, fn


if __name__ == '__main__':
    pass
