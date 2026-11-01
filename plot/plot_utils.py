import matplotlib
# 【修复】在导入 pyplot 之前，强制使用非交互式后端 'Agg'
# 这会阻止 matplotlib 尝试连接 X11 服务器和弹出窗口
matplotlib.use('Agg')

from scipy.interpolate import interp1d
from easydict import EasyDict
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt # 导入 matplotlib
import os


def plot_score_distribution(pre_dict, save_path,fake_index,epoch):
    """
    绘制真实/伪造样本的预测概率分布图并保存。

    Args:
        y_trues (list or np.array): 真实标签 (0 for real, 1 for fake)。
        y_preds (list or np.array): 模型预测为伪造样本的概率。
        dataset_name (str): 数据集名称，用于图表标题。
        save_path (str): 图像保存的完整路径。
    """
    y_trues = pre_dict['label']
    y_preds = pre_dict['pred']
    dataset_name = pre_dict['dataset']
    save_path = os.path.join(save_path, dataset_name+'fake_index'+str(fake_index)+'_epoch_'+str(epoch)+'_score_distribution.png')

    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)

    real_scores = y_preds[y_trues == 0]
    fake_scores = y_preds[y_trues == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(real_scores, bins=100, alpha=0.5, color='green', label='Real', density=True)
    plt.hist(fake_scores, bins=100, alpha=0.5, color='red', label='Fake', density=True)
    
    plt.title(f'Probability Distribution on {dataset_name}')
    plt.xlabel('Prob')
    plt.ylabel('Density')
    plt.legend(loc='upper center')
    plt.grid(True)
    
    # 保存图像
    plt.savefig(save_path, bbox_inches='tight')
    plt.close() # 关
    print(f'Score distribution plot saved to {save_path}')



# [文件名: DR-inctro/plot/plot_utils.py]

# ... (保持原有的 import 不变)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
# 【新增】导入 TSNE
from sklearn.manifold import TSNE

# ... (保持 plot_score_distribution 不变) ...

def plot_feature_tsne(features, labels, save_path, dataset_name, epoch, max_points=2000):
    """
    使用 t-SNE 绘制真假样本的特征空间分布图。
    
    Args:
        features (np.array): 模型提取的特征向量 [N, Feature_Dim]
        labels (np.array): 真实标签 [N] (0=Real, 1=Fake)
        save_path (str): 保存路径
        dataset_name (str): 数据集名称
        epoch (int): 当前轮数
        max_points (int): 为了速度，限制绘图的最大点数（随机采样）
    """
    # 1. 随机采样 (如果数据太多，t-SNE 会跑得很慢)
    if len(labels) > max_points:
        # 生成随机索引
        indices = np.random.choice(len(labels), max_points, replace=False)
        features_sampled = features[indices]
        labels_sampled = labels[indices]
    else:
        features_sampled = features
        labels_sampled = labels

    print(f"正在进行 t-SNE 降维 (样本数: {len(labels_sampled)})...")
    
    # 2. 运行 t-SNE (将高维特征降到 2D)
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features_sampled)

    # 3. 绘图
    plt.figure(figsize=(10, 8))
    
    # 绘制 Real (Label 0) - 绿色
    plt.scatter(features_2d[labels_sampled==0, 0], features_2d[labels_sampled==0, 1], 
                c='green', label='Real', alpha=0.6, s=20, edgecolors='w', linewidth=0.5)
    
    # 绘制 Fake (Label 1) - 红色
    plt.scatter(features_2d[labels_sampled==1, 0], features_2d[labels_sampled==1, 1], 
                c='red', label='Fake', alpha=0.6, s=20, edgecolors='w', linewidth=0.5)

    plt.title(f'Feature Space Visualization (t-SNE) - {dataset_name} Epoch {epoch}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 4. 保存
    file_name = f'{dataset_name}_epoch_{epoch}_tsne.png'
    full_path = os.path.join(save_path, file_name)
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'特征分布图已保存至: {full_path}')



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

# def plot_feature_tsne_multi(features, labels, save_path, dataset_name="GenImage", epoch=0, max_points=4000):
#     """
#     绘制多类别 t-SNE 特征空间分布图 (论文发表级质量)
    
#     Args:
#         features: numpy array [N, D] 高维特征 (例如 1024维)
#         labels: numpy array [N] 标签 (0为真图，1-8为不同生成器的假图)
#     """
#     print(f"\n[t-SNE] 正在计算降维... 特征维度: {features.shape}，请耐心等待1-2分钟...")
    
#     # 1. 限制最大点数以加快 t-SNE 计算速度并防止图像过于拥挤
#     if len(features) > max_points:
#         indices = np.random.choice(len(features), max_points, replace=False)
#         features = features[indices]
#         labels = labels[indices]

#     # 2. 执行 t-SNE 降维 (1024维 -> 2维)
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
#     # tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto', metric='euclidean')
    
#     # # 确保使用 metric='cosine'
#     # tsne = TSNE(
#     # n_components=2, 
#     # random_state=42, 
#     # perplexity=40, 
#     # metric='cosine',          # 必须是 cosine，配合前面的 L2 归一化
#     # early_exaggeration=24.0,  # 默认是12，调大一点可以强制把不同的簇推得更远！
#     # init='pca', 
#     # learning_rate='auto')

#     features_2d = tsne.fit_transform(features)

#     # 3. 科研级配色与标签映射 (仿照顶会论文配色)
#     # 0代表Real(深紫色)，1-8代表不同的AIGC生成器
#     model_names = {
#         0: 'Real', 
#         1: 'SDv1.4', 2: 'SDv1.5', 3: 'Midjourney', 4: 'Wukong', 
#         5: 'VQDM', 6: 'Glide', 7: 'BigGAN', 8: 'ADM'
#     }
    
#     # 配色方案：Real用深紫，其他Fake使用过渡色盘 (viridis/plasma的抽样)
#     colors = {
#         0: '#440154', # Real - 深紫
#         1: '#482878', # SDv1.4 
#         2: '#3e4989', # SDv1.5 
#         3: '#31688e', # Midjourney
#         4: '#26828e', # Wukong
#         5: '#1f9e89', # VQDM
#         6: '#35b779', # Glide
#         7: '#6ece58', # BigGAN
#         8: '#b5de2b'  # ADM (亮黄绿)
#     }

#     # 4. 开始绘图
#     plt.figure(figsize=(8, 6), dpi=300) # 高清分辨率
    
#     # 为了保证图例顺序，按 0-8 遍历绘制
#     unique_labels = np.unique(labels)
#     for lbl in sorted(unique_labels):
#         if lbl in model_names:
#             idx = (labels == lbl)
#             plt.scatter(
#                 features_2d[idx, 0], 
#                 features_2d[idx, 1], 
#                 c=colors[lbl], 
#                 label=model_names[lbl],
#                 alpha=0.8,    # 稍微透明一点，防止重叠点看不见
#                 edgecolors='none',
#                 s=30          # 点的大小
#             )

#     # 5. 美化图表
#     plt.legend(loc='lower right', fontsize=10, markerscale=1.5, framealpha=0.9)
#     # 去除顶部和右侧的边框线，更具学术感
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
#     # 6. 保存图片
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
    
#     img_save_path = os.path.join(save_path, f"tsne_{dataset_name}_epoch{epoch}.png")
#     plt.tight_layout()
#     plt.savefig(img_save_path, bbox_inches='tight', format='png')
#     plt.close()
#     print(f"[t-SNE] 绘制完成！已保存至: {img_save_path}")



#润色的图表
def plot_feature_tsne_multi(features, labels, save_path, dataset_name="GenImage", epoch=0, max_points=4000):
    """
    绘制多类别 t-SNE 特征空间分布图 (全边框 + 图例右置)
    """
    print(f"\n[t-SNE] 正在计算降维... 特征维度: {features.shape}，请耐心等待1-2分钟...")
    
    # 1. 限制最大点数以加快 t-SNE 计算速度并防止图像过于拥挤
    if len(features) > max_points:
        indices = np.random.choice(len(features), max_points, replace=False)
        features = features[indices]
        labels = labels[indices]

    # 2. 执行 t-SNE 降维 (1024维 -> 2维)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features)

    # 3. 科研级配色与标签映射
    model_names = {
        0: 'Real', 
        1: 'SDv1.4', 2: 'SDv1.5', 3: 'Midjourney', 4: 'Wukong', 
        5: 'VQDM', 6: 'Glide', 7: 'BigGAN', 8: 'ADM'
    }
    
    colors = {
        0: '#440154', # Real - 深紫
        1: '#482878', # SDv1.4 
        2: '#3e4989', # SDv1.5 
        3: '#31688e', # Midjourney
        4: '#26828e', # Wukong
        5: '#1f9e89', # VQDM
        6: '#35b779', # Glide
        7: '#6ece58', # BigGAN
        8: '#b5de2b'  # ADM (亮黄绿)
    }

    # 4. 开始绘图
    plt.figure(figsize=(8, 6), dpi=300) 
    
    unique_labels = np.unique(labels)
    for lbl in sorted(unique_labels):
        if lbl in model_names:
            idx = (labels == lbl)
            plt.scatter(
                features_2d[idx, 0], 
                features_2d[idx, 1], 
                c=colors[lbl], 
                label=model_names[lbl],
                alpha=0.8,    
                edgecolors='none',
                s=30          
            )

    # ================= 核心修改区域 =================
    
    # 5. 美化图表
    # 【修改1】：使用 bbox_to_anchor 将图例移到画板外侧的右边，loc='upper left' 意味着图例的左上角对齐边界
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, markerscale=1.5, framealpha=0.9)
    
    # 【修改2】：删除了隐藏 top 和 right 边框线的代码，默认保留完整的四面方框
    # plt.gca().spines['top'].set_visible(False)   <-- 已删除
    # plt.gca().spines['right'].set_visible(False) <-- 已删除

    # ===============================================
    
    # 6. 保存图片
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    img_save_path = os.path.join(save_path, f"tsne_{dataset_name}_epoch{epoch}.png")
    
    # bbox_inches='tight' 在这里非常重要，它保证了画在框外的图例不会被裁剪掉
    plt.savefig(img_save_path, bbox_inches='tight', format='png')
    plt.close()
    print(f"[t-SNE] 绘制完成！已保存至: {img_save_path}")