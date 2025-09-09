import argparse
import logging
import os
import os.path as osp
import sys
import yaml
import random
import math
import pprint

import numpy as np
from tqdm import tqdm

import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.transform import (
    Compose, Resize, CenterCrop, RandomCrop, RandomRotation, RandomVerticalFlip, 
    RandomHorizontalFlip, ToTensor, ImageNormalize, RandomResizedCrop, RandomAffine, 
    ColorJitter
)

from utils.val_utils import evaluate_val_set, evaluate_val_set_multi
from utils.ema import EMA
from utils.dataset import load_train_val_samples, split_kfold_dataset, ImageFolder2
from utils.samplerv5 import SemiSupervisedBalanceDataset, ManualBalancedSampler, EfficientBalancedSampler
from utils.ldamlossV2 import RWLDAMDRWLoss

from jimm import swin_base_patch4_window12_384, vit_base_patch16_224

# 导入自定义变换和构建函数
from utils.jittor_transform import (
    GridDistortion, ElasticTransform, CoarseDropout, 
    JittorTransformWrapper, RandomChoice, RandomApply,
    build_transform  # 直接导入build_transform函数
)

jt.flags.use_cuda = 1


import jittor as jt
import jittor.nn as nn

def compute_self_confidence(teacher_probs, student_probs):
    """
    计算自信度：衡量教师模型与学生模型预测的一致性
    使用KL散度的倒数作为置信度指标
    
    Args:
        teacher_probs: [B, C] 教师模型的softmax概率分布
        student_probs: [B, C] 学生模型的softmax概率分布
    
    Returns:
        self_conf: [B] 每个样本的自信度，值域[0,1]，越大表示越自信
    """
    # 添加数值稳定性处理
    eps = 1e-8
    teacher_probs = jt.clamp(teacher_probs, eps, 1 - eps)
    student_probs = jt.clamp(student_probs, eps, 1 - eps)
    
    # 使用 Jittor 的 KLDivLoss 计算KL散度：D_KL(teacher || student)
    # KLDivLoss 的输入：input 是 log_prob，target 是 prob
    kl_loss = jt.nn.KLDivLoss(reduction='none', log_target=False)
    kl_div = kl_loss(jt.log(student_probs), teacher_probs).sum(dim=-1)  # [B]
    
    # 归一化KL散度：使用log(num_classes)作为最大可能的KL散度
    num_classes = teacher_probs.shape[-1]
    max_kl = jt.log(jt.array([float(num_classes)]))
    
    # 自信度 = 1 - normalized_kl，值域[0,1]
    self_conf = 1.0 - jt.clamp(kl_div / max_kl, 0.0, 1.0)
    
    return self_conf

def compute_mutual_confidence(probs_f, probs_c):
    """
    计算互信度：衡量两个教师模型预测的相似性
    使用Jensen-Shannon散度的倒数作为相似性指标
    
    Args:
        probs_f: [B, C] Foundation模型(ViT)的softmax概率分布
        probs_c: [B, C] Conventional模型(Swin)的softmax概率分布
    
    Returns:
        mutual_conf: [B] 每个样本的互信度，值域[0,1]，越大表示两模型越相似
    """
    # 添加数值稳定性处理
    eps = 1e-8
    probs_f = jt.clamp(probs_f, eps, 1 - eps)
    probs_c = jt.clamp(probs_c, eps, 1 - eps)
    
    # 计算JS散度的中间分布M
    m = 0.5 * (probs_f + probs_c)
    m = jt.clamp(m, eps, 1 - eps)
    
    # JS散度 = 0.5 * [D_KL(P||M) + D_KL(Q||M)]
    # 使用 Jittor 的 KLDivLoss
    kl_loss = jt.nn.KLDivLoss(reduction='none', log_target=False)
    
    kl_f_m = kl_loss(jt.log(m), probs_f).sum(dim=-1)
    kl_c_m = kl_loss(jt.log(m), probs_c).sum(dim=-1)
    
    js_div = 0.5 * (kl_f_m + kl_c_m)
    
    # JS散度的最大值是log(2)，用于归一化
    max_js = jt.log(jt.array([2.0]))
    
    # 互信度 = 1 - normalized_js，值域[0,1]
    mutual_conf = 1.0 - jt.clamp(js_div / max_js, 0.0, 1.0)
    
    return mutual_conf

def cdcr_loss(probs_f, probs_c, lambda_div=1.0):
    """
    CDCR损失 for 分类任务 (Consensus-Divergence Collaborative Regulation)
    
    Args:
        probs_f: [B, C] ViT基础模型的softmax概率
        probs_c: [B, C] Swin常规模型的softmax概率
        lambda_div: 分歧损失的权重系数
    
    Returns:
        tuple: (total_loss, consensus_loss, divergence_loss, consensus_ratio)
    """
    # 数值稳定性处理
    eps = 1e-8
    probs_f = jt.clamp(probs_f, eps, 1 - eps)
    probs_c = jt.clamp(probs_c, eps, 1 - eps)
    
    # 共识掩码: [B]，1 if 两个模型预测类相同
    pred_f = jt.argmax(probs_f, dim=-1)[0]  # Jittor的argmax返回tuple
    pred_c = jt.argmax(probs_c, dim=-1)[0]
    consensus_mask = (pred_f == pred_c).float()  # [B]
    
    # 共识损失: 鼓励共识样本的高自信 (平均熵最小化)
    entropy_f = -(probs_f * jt.log(probs_f)).sum(dim=-1)  # [B]
    entropy_c = -(probs_c * jt.log(probs_c)).sum(dim=-1)  # [B]
    avg_entropy = (entropy_f + entropy_c) / 2  # [B]
    
    consensus_count = consensus_mask.sum() + eps
    l_cons = (avg_entropy * consensus_mask).sum() / consensus_count
    
    # 分歧损失: 对分歧样本对齐概率 (MSE)
    divergence_mask = 1 - consensus_mask  # [B]
    # 手动计算MSE，避免Jittor的mse_loss参数问题
    mse = ((probs_f - probs_c) ** 2).mean(dim=-1)  # [B]
    
    divergence_count = divergence_mask.sum() + eps
    l_div = (mse * divergence_mask).sum() / divergence_count
    
    # 统计信息
    consensus_ratio = consensus_count.item() / len(consensus_mask)
    
    total_loss = l_cons + lambda_div * l_div
    
    return total_loss, l_cons, l_div, consensus_ratio

def smc_integration(probs_f_teacher, probs_f_student, probs_c_teacher, probs_c_student):
    """
    Self-Mutual Confidence Integration：自互信度整合
    
    基于自信度和互信度动态融合两个模型的预测结果
    
    Args:
        probs_f_teacher: [B, C] ViT教师模型的softmax概率
        probs_f_student: [B, C] ViT学生模型的softmax概率  
        probs_c_teacher: [B, C] Swin教师模型的softmax概率
        probs_c_student: [B, C] Swin学生模型的softmax概率
    
    Returns:
        pseudo_label: [B] 融合后的硬标签
        pseudo_probs: [B, C] 融合后的软概率分布
        confidence_info: dict 包含各种置信度信息用于分析
    """
    # 步骤1：计算各自的自信度
    self_conf_f = compute_self_confidence(probs_f_teacher, probs_f_student)  # ViT自信度
    self_conf_c = compute_self_confidence(probs_c_teacher, probs_c_student)  # Swin自信度
    
    # 步骤2：计算互信度（两个教师模型的相似性）
    mutual_conf = compute_mutual_confidence(probs_f_teacher, probs_c_teacher)
    
    # 步骤3：计算动态融合权重
    # 加权策略：每个模型的权重 = 自信度 × 互信度
    weight_f = self_conf_f * mutual_conf  # ViT权重
    weight_c = self_conf_c * mutual_conf  # Swin权重
    
    # 归一化权重，确保和为1
    total_weight = weight_f + weight_c + 1e-8  # 避免除零
    alpha_c = weight_c / total_weight  # Swin的权重比例
    alpha_f = weight_f / total_weight  # ViT的权重比例
    
    # 验证权重和为1（调试用）
    # weight_sum = alpha_c + alpha_f  # 应该接近1
    
    # 步骤4：融合预测概率
    # 加权平均：alpha_c * Swin + alpha_f * ViT
    pseudo_probs = (alpha_c.unsqueeze(-1) * probs_c_teacher + 
                   alpha_f.unsqueeze(-1) * probs_f_teacher)
    
    # 生成硬标签
    pseudo_label = jt.argmax(pseudo_probs, dim=-1)[0]  # Jittor的argmax返回tuple
    
    # 收集置信度信息用于分析和调试
    confidence_info = {
        'self_conf_vit': self_conf_f,      # ViT自信度
        'self_conf_swin': self_conf_c,     # Swin自信度  
        'mutual_conf': mutual_conf,        # 互信度
        'weight_vit': alpha_f,             # ViT最终权重
        'weight_swin': alpha_c,            # Swin最终权重
        'avg_self_conf_vit': self_conf_f.mean(),    # 平均ViT自信度
        'avg_self_conf_swin': self_conf_c.mean(),   # 平均Swin自信度
        'avg_mutual_conf': mutual_conf.mean(),      # 平均互信度
        'avg_weight_vit': alpha_f.mean(),           # 平均ViT权重
        'avg_weight_swin': alpha_c.mean()           # 平均Swin权重
    }
    
    return pseudo_label, pseudo_probs, confidence_info


def compute_class_distribution(samples, num_classes):
    """
    统计训练样本的类别分布
    
    Args:
        samples: 样本列表，每个元素为 (image_path, label)
        num_classes: 类别总数
    
    Returns:
        list: 每个类别的样本数量，按类别标签 0,1,2,... 的顺序排列
    """
    class_counts = [0] * num_classes
    
    for _, label in samples:
        if 0 <= label < num_classes:
            class_counts[label] += 1
        else:
            logging.warning(f"发现超出范围的标签: {label}, 期望范围: 0-{num_classes-1}")
    
    return class_counts


def training(model: nn.Module, vit_model: nn.Module, criterion, optimizer: nn.Optimizer, vit_optimizer: nn.Optimizer,
             train_loader: Dataset, now_epoch: int, num_epochs: int,
             global_step: int, warmup_steps: int, base_lr: float, 
             total_steps: int, cosine: bool, final_lr: float, 
             model_ema=None, vit_ema=None, consistency_weight=0.1):
    model.train()
    vit_model.train()
    if model_ema is not None:
        model_ema.get_model().eval()
    if vit_ema is not None:
        vit_ema.get_model().eval()
    
    losses = []
    vit_losses = []
    ema_losses = []
    ema_vit_losses = []
    
    # 计算正确的batch数量
    total_batches = (len(train_loader) + train_loader.batch_size - 1) // train_loader.batch_size
    pbar = tqdm(train_loader, total=total_batches,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    def get_lr(global_step):
        if global_step < warmup_steps:
            return base_lr * global_step / warmup_steps
        if cosine:
            progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
            lr = final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * progress))
            return lr
        else:
            return base_lr
    
    for step, data in enumerate(pbar, 1):
        global_step += 1
        
        # 半监督模式：解包5个元素
        strong_vit, weak_vit, strong_orig, weak_orig, labels = data
        
        # 根据数据集配置直接计算有标记和无标记数据的索引
        labeled_count = train_loader.labeled_per_batch
        unlabeled_count = train_loader.unlabeled_per_batch
        
        # 分离有标记和无标记数据
        if labeled_count > 0:  # 有标记数据存在
            labeled_strong_orig = strong_orig[:labeled_count]
            labeled_strong_vit = strong_vit[:labeled_count]
            labeled_weak_orig = weak_orig[:labeled_count]
            labeled_weak_vit = weak_vit[:labeled_count]
            labeled_labels = labels[:labeled_count]
        else:
            labeled_strong_orig = labeled_strong_vit = None
            labeled_weak_orig = labeled_weak_vit = None
            labeled_labels = None
            
        if unlabeled_count > 0:  # 无标记数据存在
            unlabeled_strong_orig = strong_orig[labeled_count:]
            unlabeled_strong_vit = strong_vit[labeled_count:]
            unlabeled_weak_orig = weak_orig[labeled_count:]
            unlabeled_weak_vit = weak_vit[labeled_count:]
        else:
            unlabeled_strong_orig = unlabeled_strong_vit = None
            unlabeled_weak_orig = unlabeled_weak_vit = None

        lr = get_lr(global_step)
        for group in optimizer.param_groups:
            group['lr'] = lr
        for group in vit_optimizer.param_groups:
            group['lr'] = lr

        total_loss = 0
        total_vit_loss = 0
        loss_count = 0
        
        # 有标记数据的损失计算
        if labeled_labels is not None and len(labeled_labels) > 0:
            # 强增强数据输入主模型训练
            pred_strong = model(labeled_strong_orig)
            vit_pred_strong = vit_model(labeled_strong_vit)
            
            # 计算有标记数据的监督损失
            supervised_loss = criterion(pred_strong, labeled_labels, now_epoch)
            supervised_vit_loss = criterion(vit_pred_strong, labeled_labels, now_epoch)
            
            total_loss += supervised_loss
            total_vit_loss += supervised_vit_loss
            loss_count += 1
            
            # 注意：移除重复的一致性损失，因为SMC Integration已经包含了teacher-student一致性
        
        # 无标记数据的半监督损失（SMC Integration + CDCR Loss）
        if unlabeled_strong_orig is not None and len(unlabeled_strong_orig) > 0:
            if model_ema is not None and vit_ema is not None:
                with jt.no_grad():
                    # 弱增强输入EMA教师生成伪标签
                    ema_pred_weak_c = model_ema.get_model()(unlabeled_weak_orig)  # Swin (conventional)
                    ema_pred_weak_f = vit_ema.get_model()(unlabeled_weak_vit)    # ViT (foundation)
                
                # 强增强输入学生（需要梯度，用于训练）
                pred_strong_c = model(unlabeled_strong_orig)                 # Swin学生
                pred_strong_f = vit_model(unlabeled_strong_vit)              # ViT学生
                
                # Softmax转换为概率
                probs_c_teacher = jt.nn.softmax(ema_pred_weak_c, dim=1)
                probs_f_teacher = jt.nn.softmax(ema_pred_weak_f, dim=1)
                probs_c_student = jt.nn.softmax(pred_strong_c, dim=1)
                probs_f_student = jt.nn.softmax(pred_strong_f, dim=1)
                
                # === 策略1: SMC Integration 伪标签生成 ===
                pseudo_label, pseudo_probs, conf_info = smc_integration(
                    probs_f_teacher, probs_f_student,  # ViT (foundation)
                    probs_c_teacher, probs_c_student   # Swin (conventional)
                )
                
                # === 策略2: CDCR Loss 教师模型协作损失 ===
                cdcr_total, cdcr_cons, cdcr_div, consensus_ratio = cdcr_loss(
                    probs_f_teacher, probs_c_teacher, 
                    lambda_div=args.get('cdcr_lambda_div', 1.0)
                )
                
                # === 改进1: 置信度阈值过滤 ===
                confidence_threshold = args.get('confidence_threshold', 0.7)
                high_conf_mask = (conf_info['mutual_conf'] > confidence_threshold) & \
                               (conf_info['self_conf_vit'] > confidence_threshold) & \
                               (conf_info['self_conf_swin'] > confidence_threshold)
                
                # 分离模型的损失计算和记录
                swin_semi_losses = []
                vit_semi_losses = []
                
                if high_conf_mask.sum() > 0:  # 有高置信度样本时才计算SMC损失
                    # 过滤高置信度样本
                    filtered_probs_c_student = probs_c_student[high_conf_mask]
                    filtered_probs_f_student = probs_f_student[high_conf_mask]
                    filtered_pseudo_probs = pseudo_probs[high_conf_mask]
                    
                    # === 改进2: 基于epoch的动态策略 ===
                    epoch_progress = now_epoch / num_epochs
                    # 早期：ViT主导（Foundation模型通常更稳定）
                    if now_epoch < 30:
                        vit_boost = 1.8
                        swin_boost = 0.2
                        dynamic_factor = "early_vit_dominant"
                    # 中期：平衡融合
                    elif now_epoch < 70:
                        vit_boost = 1.0
                        swin_boost = 1.0
                        dynamic_factor = "balanced"
                    # 后期：根据各自表现动态调整
                    else:
                        avg_vit_conf = conf_info['avg_self_conf_vit'].item()
                        avg_swin_conf = conf_info['avg_self_conf_swin'].item()
                        vit_boost = avg_vit_conf + 0.5
                        swin_boost = avg_swin_conf + 0.5
                        dynamic_factor = "adaptive"
                    
                    # 重新计算调整后的伪标签
                    if dynamic_factor != "balanced":
                        adjusted_weights_vit = conf_info['weight_vit'][high_conf_mask] * vit_boost
                        adjusted_weights_swin = conf_info['weight_swin'][high_conf_mask] * swin_boost
                        total_adjusted_weights = adjusted_weights_vit + adjusted_weights_swin + 1e-8
                        
                        alpha_vit_adj = adjusted_weights_vit / total_adjusted_weights
                        alpha_swin_adj = adjusted_weights_swin / total_adjusted_weights
                        
                        filtered_pseudo_probs = (alpha_swin_adj.unsqueeze(-1) * probs_c_teacher[high_conf_mask] + 
                                               alpha_vit_adj.unsqueeze(-1) * probs_f_teacher[high_conf_mask])
                    
                    # === 简化版损失：只使用MSE损失（移除重复的KL损失） ===
                    semi_weight = args.get('semi_weight', 1.0)
                    
                    # 软标签一致性损失（手动MSE计算）- 学生强增强输出 vs. 整合伪标签
                    mse_loss_c = ((filtered_probs_c_student - filtered_pseudo_probs) ** 2).mean()
                    mse_loss_f = ((filtered_probs_f_student - filtered_pseudo_probs) ** 2).mean()
                    
                    # === 改进3: 动态调整半监督损失权重 ===
                    avg_confidence = (conf_info['avg_mutual_conf'].item() + 
                                    conf_info['avg_self_conf_vit'].item() + 
                                    conf_info['avg_self_conf_swin'].item()) / 3
                    
                    dynamic_semi_weight = semi_weight * avg_confidence * min(1.0, epoch_progress * 1.5)
                    
                    # 记录各模型的半监督损失
                    swin_semi_losses.append(('SMC_MSE', dynamic_semi_weight * mse_loss_c))
                    vit_semi_losses.append(('SMC_MSE', dynamic_semi_weight * mse_loss_f))
                    
                    # 添加到各自的总损失
                    total_loss += dynamic_semi_weight * mse_loss_c
                    total_vit_loss += dynamic_semi_weight * mse_loss_f
                    
                else:
                    # 没有高置信度样本时记录0损失
                    swin_semi_losses.append(('SMC_MSE', jt.array(0.0)))
                    vit_semi_losses.append(('SMC_MSE', jt.array(0.0)))
                
                # === 策略3: 添加CDCR协作损失到两个模型 ===
                cdcr_weight = args.get('cdcr_weight', 0.5)
                cdcr_weighted = cdcr_weight * cdcr_total
                
                # CDCR损失同时影响两个模型（促进协作）
                total_loss += cdcr_weighted
                total_vit_loss += cdcr_weighted
                
                # 记录CDCR损失
                swin_semi_losses.append(('CDCR', cdcr_weighted))
                vit_semi_losses.append(('CDCR', cdcr_weighted))
                
                loss_count += 1
                
                # === 改进4: 详细的分模型监控信息 ===
                if now_epoch % 5 == 0 or (now_epoch < 10):
                    high_conf_ratio = high_conf_mask.sum().item() / len(high_conf_mask) if len(high_conf_mask) > 0 else 0.0
                    
                    # 计算各模型的总半监督损失
                    swin_total_semi = sum([loss.item() for _, loss in swin_semi_losses])
                    vit_total_semi = sum([loss.item() for _, loss in vit_semi_losses])
                    logging.info(f"Semi-Supervised Stats - Epoch {now_epoch}:")
                    logging.info(f"  Strategy={dynamic_factor}, HighConfRatio={high_conf_ratio:.3f}")
                    logging.info(f"  Confidence - ViT={conf_info['avg_self_conf_vit']:.3f}, "
                               f"Swin={conf_info['avg_self_conf_swin']:.3f}, "
                               f"Mutual={conf_info['avg_mutual_conf']:.3f}")
                    logging.info(f"  Weights - ViT={conf_info['avg_weight_vit']:.3f}, "
                               f"Swin={conf_info['avg_weight_swin']:.3f}")
                    logging.info(f"  CDCR - Total={cdcr_total.item():.4f}, "
                               f"Consensus={cdcr_cons.item():.4f}, "
                               f"Divergence={cdcr_div.item():.4f}, "
                               f"ConsensusRatio={consensus_ratio:.3f}")
                    logging.info(f"  Semi Loss - Swin={swin_total_semi:.4f}, ViT={vit_total_semi:.4f}")
                    
                    # 分解损失记录
                    for loss_name, loss_val in swin_semi_losses:
                        logging.info(f"    Swin_{loss_name}={loss_val.item():.4f}")
                    for loss_name, loss_val in vit_semi_losses:
                        logging.info(f"    ViT_{loss_name}={loss_val.item():.4f}")
                        
                else:
                    # 简化记录
                    if now_epoch % 20 == 0 and high_conf_mask.sum() == 0:
                        logging.warning(f"Epoch {now_epoch}: No high-confidence unlabeled samples "
                                      f"(threshold={confidence_threshold}). "
                                      f"Consider lowering confidence_threshold.")
            else:
                logging.warning("EMA未初始化，无法生成伪标签")
            
        if loss_count > 0:
            # 平均损失
            final_loss = total_loss / loss_count
            final_vit_loss = total_vit_loss / loss_count
            
            final_loss.sync()
            final_vit_loss.sync()
            
            optimizer.step(final_loss)
            vit_optimizer.step(final_vit_loss)
            
            losses.append(final_loss.item())
            vit_losses.append(final_vit_loss.item())
        else:
            # 没有有标记数据的情况（理论上不应该发生）
            losses.append(0.0)
            vit_losses.append(0.0)
        
        # 更新EMA
        if model_ema is not None:
            model_ema.update(model)
        if vit_ema is not None:
            vit_ema.update(vit_model)

        pbar.set_description(f'Epoch {now_epoch} [TRAIN] loss={losses[-1]:.4f} vit_loss={vit_losses[-1]:.4f} lr={lr:.6f}')

    avg_ema_loss = np.mean(ema_losses) if ema_losses else 0.0
    avg_ema_vit_loss = np.mean(ema_vit_losses) if ema_vit_losses else 0.0
    
    logging.info(f'Epoch {now_epoch} / {num_epochs} [TRAIN] mean loss = {np.mean(losses):.4f} vit_loss = {np.mean(vit_losses):.4f} '
                f'ema_loss = {avg_ema_loss:.4f} ema_vit_loss = {avg_ema_vit_loss:.4f} lr={lr:.6f}')
    return global_step

def run(args, snapshot_path):
    model = swin_base_patch4_window12_384(pretrained=True, num_classes=6)

    vit_model = vit_base_patch16_224(pretrained=False, num_classes=6)
    vit_model.load('./biomdeclip_vit_only_jittor.pth')
    
    # 从外部yml读取transform配置
    with open(args['transform_cfg'], 'r') as f:
        transform_yaml = yaml.load(f, Loader=yaml.FullLoader)
    transform_val = build_transform(transform_yaml['transform_val'])
    transform_val_vit = build_transform(transform_yaml['transform_val_vit'])

    train_samples, val_samples = split_kfold_dataset(
        osp.join(args['root_path'], 'labels/trainval.txt'),
        fold=args['fold'], num_folds=args['total_folds'],
        shuffle=True, seed=args.get('seed', 42)
    )
    logging.info(f"Fold {args['fold']} / {args['total_folds']} - 使用样本: {len(train_samples)}, 验证样本: {len(val_samples)}")

    # ===== 动态统计训练集标签分布 =====
    actual_cls_num_list = compute_class_distribution(train_samples, args.get('num_classes', 6))
    logging.info(f"实际训练集类别分布: {actual_cls_num_list}")

    # 根据实际的类别分布创建RW-LDAM-DRW损失函数
    criterion = RWLDAMDRWLoss(
        cls_num_list=actual_cls_num_list,
        max_m=args.get('max_m', 0.5),
        s=args.get('s', 30),
        reweight_epoch=args.get('reweight_epoch', 80),
        total_epochs=args['epochs'],
        reweight_type=args.get('reweight_type', 'inverse')
    )
    
    logging.info(f"使用RW-LDAM-DRW损失函数")
    logging.info(f"实际类别分布: {actual_cls_num_list}")
    logging.info(f"重加权开始epoch: {args.get('reweight_epoch', 80)}")
    logging.info(f"LDAM边距参数 max_m: {args.get('max_m', 0.5)}")
    logging.info(f"LDAM缩放因子 s: {args.get('s', 30)}")
    logging.info(f"重加权类型: {args.get('reweight_type', 'inverse')}")

    # 构建增强策略
    # 强增强：用于主模型训练
    strong_transform = build_transform(transform_yaml['transform_strong'])
    # 弱增强：用于EMA模型
    weak_transform = build_transform( transform_yaml['transform_weak'])
    # ViT专用增强
    vit_transform = build_transform(transform_yaml['transform_vit'])

    # 使用半监督数据集
    class_oversample_ratios = args.get('class_oversample_ratios', {
        0: 1.0, 1: 1.0, 2: 1.5, 3: 8.0, 4: 12.0, 5: 1.0
    })
    
    train_loader = SemiSupervisedBalanceDataset(
        root=os.path.join(args['root_path'], 'images/train'),
        samples=train_samples,
        unlabeled_dir=args.get('unlabeled_dir', '/app/Jittor/DATASET/OTHERS/Unlabeled2'),
        unlabeled_ratio=args.get('unlabeled_ratio', 0.25),
        strong_aug=strong_transform,
        weak_aug=weak_transform,
        vit_aug=vit_transform,
        use_balanced_sampling=True,
        class_oversample_ratios=class_oversample_ratios,
        batch_size=args['batch_size'],
        num_workers=8,
        shuffle=True
    )
    logging.info(f"使用半监督平衡数据集，无标记数据比例: {args.get('unlabeled_ratio', 0.25)}")
    logging.info(f"手动平衡采样系数: {class_oversample_ratios}")

    val_loader = ImageFolder2(
        root=os.path.join(args['root_path'], 'images/train'),
        samples=val_samples,
        transform=transform_val,
        batch_size=args['batch_size'],
        num_workers=8,
        shuffle=False
    )
    
    # 为ViT创建专门的验证数据集（224x224输入）
    val_loader_vit = ImageFolder2(
        root=os.path.join(args['root_path'], 'images/train'),
        samples=val_samples,
        transform=transform_val_vit,
        batch_size=args['batch_size'],
        num_workers=8,
        shuffle=False
    )

    optimizer = jt.optim.AdamW(model.parameters(), lr=args['base_lr'])
    vit_optimizer = jt.optim.AdamW(vit_model.parameters(), lr=args['base_lr'])
    
    # 初始化EMA（为两个模型都创建EMA）
    ema_decay = args.get('ema_decay', 0.9999)
    model_ema = EMA(model, decay=ema_decay)
    vit_ema = EMA(vit_model, decay=ema_decay)
    logging.info(f"EMA initialized for both models with decay={ema_decay}")
    
    # 初始化最佳准确率跟踪
    best_model_acc = 0
    best_vit_acc = 0
    best_model_ema_acc = 0
    best_vit_ema_acc = 0
    
    global_step = 0
    num_epochs = args['epochs']
    warmup_epochs = args.get('warmup_epochs', 5)
    cosine = args.get('cosine', False)
    final_lr = args.get('final_lr', 1e-6)
    steps_per_epoch = (len(train_loader) + train_loader.batch_size - 1) // train_loader.batch_size
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    base_lr = args['base_lr']

    # 定义类别名称（根据您的数据集调整）
    class_names = ['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5']  # 根据实际类别调整

    for epoch in range(num_epochs):
        global_step = training(
            model, vit_model, criterion, optimizer, vit_optimizer,
            train_loader, epoch, num_epochs,
            global_step, warmup_steps, base_lr,
            total_steps, cosine, final_lr, 
            model_ema, vit_ema,
            consistency_weight=args.get('consistency_weight', 0.1)
        )
        
        # 准备验证模型字典 - 包含4个模型，分别用不同的数据集
        # Swin模型和EMA使用384x384输入
        swin_models_to_eval = {
            "swin_model": model,
            "swin_ema": model_ema.get_model()
        }
        
        # ViT模型和EMA使用224x224输入
        vit_models_to_eval = {
            "vit_model": vit_model,
            "vit_ema": vit_ema.get_model()
        }
        
        # 分别验证Swin模型（384x384）
        swin_results_dict = evaluate_val_set_multi(
            swin_models_to_eval, val_loader, 
            num_classes=6, 
            class_names=class_names, 
            save_path=snapshot_path, 
            epoch=epoch,
        )
        
        # 分别验证ViT模型（224x224）
        vit_results_dict = evaluate_val_set_multi(
            vit_models_to_eval, val_loader_vit, 
            num_classes=6, 
            class_names=class_names, 
            save_path=snapshot_path, 
            epoch=epoch,
        )
        
        # 提取各模型结果
        swin_acc, swin_macro_acc, swin_report, swin_cm = swin_results_dict["swin_model"]
        swin_ema_acc, swin_ema_macro_acc, swin_ema_report, swin_ema_cm = swin_results_dict["swin_ema"]
        vit_acc, vit_macro_acc, vit_report, vit_cm = vit_results_dict["vit_model"]
        vit_ema_acc, vit_ema_macro_acc, vit_ema_report, vit_ema_cm = vit_results_dict["vit_ema"]
        
        # 记录详细的验证指标
        logging.info(f"Epoch {epoch} [VAL] Model Results:")
        logging.info(f"  Swin Model: Acc={swin_acc:.4f}, Macro={swin_macro_acc:.4f}")
        logging.info(f"  ViT Model: Acc={vit_acc:.4f}, Macro={vit_macro_acc:.4f}")
        logging.info(f"  Swin EMA: Acc={swin_ema_acc:.4f}, Macro={swin_ema_macro_acc:.4f}")
        logging.info(f"  ViT EMA: Acc={vit_ema_acc:.4f}, Macro={vit_ema_macro_acc:.4f}")
        
        # 记录每个类别的准确率（以Swin模型为例）
        for i, class_name in enumerate(class_names):
            if class_name in swin_report:
                class_acc = swin_report[class_name]['recall']
                class_precision = swin_report[class_name]['precision']
                class_f1 = swin_report[class_name]['f1-score']
                class_support = swin_report[class_name]['support']
                logging.info(f"  {class_name}: Acc={class_acc:.4f}, P={class_precision:.4f}, F1={class_f1:.4f}, N={class_support}")
            else:
                logging.info(f"  {class_name}: No samples")

        # 保存最佳模型（分别为Swin和ViT模型保存）
        if swin_acc > best_model_acc:
            best_model_acc = swin_acc
            model.save(os.path.join(snapshot_path, 'best_swin_model.pkl'))
        
        if vit_acc > best_vit_acc:
            best_vit_acc = vit_acc
            vit_model.save(os.path.join(snapshot_path, 'best_vit_model.pkl'))
        
        if swin_ema_acc > best_model_ema_acc:
            best_model_ema_acc = swin_ema_acc
            model_ema.get_model().save(os.path.join(snapshot_path, 'best_swin_ema.pkl'))
        
        if vit_ema_acc > best_vit_ema_acc:
            best_vit_ema_acc = vit_ema_acc
            vit_ema.get_model().save(os.path.join(snapshot_path, 'best_vit_ema.pkl'))
        
        # 保存最后一个epoch的模型（更新后缀）
        if epoch == num_epochs - 1:
            model.save(os.path.join(snapshot_path, 'last_swin_model.pkl'))
            vit_model.save(os.path.join(snapshot_path, 'last_vit_model.pkl'))
            model_ema.get_model().save(os.path.join(snapshot_path, 'last_swin_ema.pkl'))
            vit_ema.get_model().save(os.path.join(snapshot_path, 'last_vit_ema.pkl'))
        
        # 定期保存checkpoint（更新后缀）
        if (epoch + 1) % 50 == 0:
            model.save(os.path.join(snapshot_path, f'epoch_{epoch + 1}_swin_model.pkl'))
            vit_model.save(os.path.join(snapshot_path, f'epoch_{epoch + 1}_vit_model.pkl'))
            model_ema.get_model().save(os.path.join(snapshot_path, f'epoch_{epoch + 1}_swin_ema.pkl'))
            vit_ema.get_model().save(os.path.join(snapshot_path, f'epoch_{epoch + 1}_vit_ema.pkl'))

        # 日志记录（修正变量名）
        logging.info(f'Epoch {epoch} / {num_epochs} [VAL] '
                    f'Swin: best={best_model_acc:.2f}, current={swin_acc:.2f} | '
                    f'ViT: best={best_vit_acc:.2f}, current={vit_acc:.2f} | '
                    f'Swin_EMA: best={best_model_ema_acc:.2f}, current={swin_ema_acc:.2f} | '
                    f'ViT_EMA: best={best_vit_ema_acc:.2f}, current={vit_ema_acc:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform_cfg', type=str, default='/app/Jittor/Mycode2/cfgs/transform1_224.yml', help='transform配置文件')
    parser.add_argument('--root_path', type=str, default='/app/Jittor/DATASET/TrainSet')
    parser.add_argument('--res_path', type=str, default='/app/Jittor/Mycode2/resultsv3/')
    parser.add_argument('--exp', type=str, default='')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=80)

    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--final_lr', type=float, default=1e-6, help='cosine退火的最小学习率')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup轮数')
    parser.add_argument('--cosine', action='store_true', help='使用cosine退火调度')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_interval_ep', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    
    # K折交叉验证参数
    parser.add_argument('--fold', type=int, default=0, help='当前fold编号 (0-based)')
    parser.add_argument('--total_folds', type=int, default=4, help='总fold数量')
    
    # 数据采样策略参数
    parser.add_argument('--class_oversample_ratios', type=str, default='0:1.0,1:1.0,2:1.5,3:7.5,4:12.0,5:1.0',
                       help='手动平衡采样的各类别过采样系数，格式: 0:1.0,1:1.0,2:1.5...')
    
    # RW-LDAM-DRW损失函数参数


    parser.add_argument('--max_m', type=float, default=0.5, help='LDAM损失的最大边距值')
    parser.add_argument('--s', type=float, default=30, help='RW-LDAM-DRW的缩放因子')
    parser.add_argument('--reweight_epoch', type=int, default=80, help='开始重加权的epoch')
    parser.add_argument('--reweight_type', type=str, default='inverse', choices=['inverse', 'sqrt_inverse'], 
                       help='重加权类型')
    
    # EMA相关参数
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    
    # 半监督学习相关参数
    parser.add_argument('--unlabeled_dir', type=str, default='/app/Jittor/DATASET/OTHERS/Unlabeled2', 
                       help='无标记数据目录')
    parser.add_argument('--unlabeled_ratio', type=float, default=0.25, 
                       help='每个batch中无标记数据的比例')
    parser.add_argument('--consistency_weight', type=float, default=0.1, 
                       help='一致性损失的权重')
    parser.add_argument('--semi_weight', type=float, default=1.0, 
                       help='SMC Integration半监督损失的权重')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                       help='SMC Integration的置信度阈值，只对高置信度样本应用半监督损失')
    parser.add_argument('--cdcr_weight', type=float, default=0.5,
                       help='CDCR协作损失权重')
    parser.add_argument('--cdcr_lambda_div', type=float, default=1.0,
                       help='CDCR分歧损失的权重系数lambda_div')

    args = parser.parse_args()
    args = vars(args)
    
    
    # 解析class_oversample_ratios字符串为字典
    if 'class_oversample_ratios' in args and isinstance(args['class_oversample_ratios'], str):
        ratio_dict = {}
        for pair in args['class_oversample_ratios'].split(','):
            cls, ratio = pair.split(':')
            ratio_dict[int(cls)] = float(ratio)
        args['class_oversample_ratios'] = ratio_dict
    
    snapshot_path = osp.join(args["res_path"], args["exp"], f'fold_{args["fold"]}')
    os.makedirs(snapshot_path, exist_ok=True)
    
    # 创建验证结果保存的目录结构
    images_dir = osp.join(snapshot_path, 'images')
    val_txt_dir = osp.join(snapshot_path, 'val_txt')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(val_txt_dir, exist_ok=True)
    
    if args.get('deterministic', False):
        seed = args.get('seed', 2023)
        jt.set_global_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        logging.info(f"设置随机种子: {seed}")

    # os.environ['http_proxy'] = 'http://127.0.0.1:20171'
    # os.environ['https_proxy'] = 'https://127.0.0.1:20171'


    # 设置日志配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 移除所有旧的 handler
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    
    # 文件日志
    file_handler = logging.FileHandler(osp.join(snapshot_path, "log.txt"), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)
    
    # 终端日志
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(stream_handler)

    import pprint
    logging.info(pprint.pformat(args))

    run(args, snapshot_path)