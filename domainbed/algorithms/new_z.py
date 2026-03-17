import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

#  import higher

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches




class CustomModels(nn.Module):  # 四个网络类型
    def __init__(self, in_channels, num_features, k):
        super(CustomModels, self).__init__()
        
        self.k = k  # 划分的份数
        self.attention_layers = nn.ModuleList([ECAAttentionLayer3(in_channels) for _ in range(k + 1)])  # 生成k+1个网络
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)
        
    def split_tensor(self, x, k):
        # 获取输入 tensor 的 shape
        batch_size, *other_dims = x.shape
        
        # 确保 batch_size 可以被 k 整除
        assert batch_size % k == 0, "Batch size must be divisible by k"
        
        # 切分 tensor
        split_tensors = torch.chunk(x, k, dim=0)
        
        return split_tensors
    
    def forward(self, x):
        split_x = self.split_tensor(x, self.k)
        
        if self.training:
            output_splits = []
            # 第一个网络处理所有数据
            mul_weight, add_weight = self.attention_layers[0](x)
            first_re = x * mul_weight + add_weight

            # # 后续网络处理对应的分块数据
            # for i in range(1, self.k + 1):
            #     mul_weight, add_weight = self.attention_layers[i](split_x[i - 1])  # 只处理对应的分块
            #     output_splits.append(split_x[i] * mul_weight + add_weight)
            # output = torch.cat(output_splits, dim=0)
            # output = first_re + output # 这个是加法
            split_x = self.split_tensor(first_re, self.k)
            for i in range(1, self.k + 1):
                mul_weight, add_weight = self.attention_layers[i](split_x[i - 1])  # 只处理对应的分块
                output_splits.append(split_x[i] * mul_weight + add_weight)
            output = torch.cat(output_splits, dim=0)
            # output = first_re + output # 这个是加法

            # output = torch.cat(output_splits, dim=0)  # 输出为所有层的结果

        else:
            # 测试模式，仅处理第一个网络的输出
            mul_weight, add_weight = self.attention_layers[0](x)
            output = x * mul_weight + add_weight
            
            # 归一化
           #  output = self.layer_norm(output) # 这个有必要吗？
        
        return output





class CustomModelslect(nn.Module):
    def __init__(self, in_channels, num_features, k):
        super(CustomModelslect, self).__init__()
        self.k = k
        self.attention_layers = nn.ModuleList([ECAAttentionLayer3(in_channels) for _ in range(k)])
        self.layer_norm = nn.LayerNorm(num_features, elementwise_affine=False)

        # 初始化掩码为与参数大小相同的全1张量
        self.masks = [torch.zeros_like(layer.parameters().__next__().data) for layer in self.attention_layers]

    def forward(self, x):
        split_x = self.split_tensor(x, self.k)
        output_splits = []

        for i in range(self.k):
            output = self.attention_layers[i](split_x[i])
            output_splits.append(output)

        return torch.cat(output_splits, dim=0)

    def update_masks(self, percentage):
        """根据自有参数的梯度更新掩码"""
        for i in range(self.k):
            for param in self.attention_layers[i].parameters():
                if param.grad is not None:
                    grad = param.grad.data.abs()
                    topk_values, topk_indices = torch.topk(grad.view(-1), k=int(percentage * grad.numel()))
                    mask = torch.zeros_like(grad.view(-1))
                    mask[topk_indices] = 1
                    self.masks[i] = mask.view_as(grad).to(param.device)

    def average_parameters_with_masks(self):
        """对掩码为0的参数进行平均化"""
        with torch.no_grad():  # 确保在不跟踪梯度的情况下进行平均化
            # 将掩码倒转
            inverted_masks = [1 - mask for mask in self.masks]
            
            # 将倒转后的掩码转换为张量并逐元素相乘
            masks_tensor = torch.stack(inverted_masks)  # 将倒转的掩码转换为张量
            combined_mask = torch.prod(masks_tensor, dim=0)  # 逐元素相乘得到新的掩码

            for name, param in self.attention_layers[0].named_parameters():
                # 对于每个参数，访问所有k个网络中的同名参数
                params_to_average = []
                for j in range(self.k):
                    if name in self.attention_layers[j].named_parameters():
                        params_to_average.append(self.attention_layers[j].named_parameters()[name].data)

                # 计算均值并更新所有相关参数
                if params_to_average:
                    average_param = sum(params_to_average) / self.k
                    for j in range(self.k):
                        if name in self.attention_layers[j].named_parameters():
                            self.attention_layers[j].named_parameters()[name].data.copy_(average_param)

            return combined_mask  # 返回新的掩码

    def split_tensor(self, x, k):
        batch_size, *other_dims = x.shape
        assert batch_size % k == 0, "Batch size must be divisible by k"
        return torch.chunk(x, k, dim=0)




class SPU_min:
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        """
        初始化 SPU2 类。

        Args:
            args (dict): 参数字典，包含 'k', 'selection_rate', 'device', 'score_batch_percentage' 等。
            Tnetwork (nn.Module): 主网络模型。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            classifier (nn.Module): 分类器模型。
        """
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
    
    def compute_score(self, data, labels, selected_layers):
        """
        计算每个参数的梯度重要性。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。
            selected_layers (str): 需要计算重要性的层名标识。

        Returns:
            Dict[str, Tensor]: 每个参数名称对应的梯度重要性。
        """
        importance = {name: torch.zeros_like(param, device='cpu') 
                     for name, param in self.network.named_parameters() 
                     if selected_layers in name}
        self.network.train()
        self.network.zero_grad()

        # 前向传播
        logits_per_image = self.network(data)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss.backward()

        # 收集梯度
        for name, param in self.network.named_parameters():
            if selected_layers in name and param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().cpu().clone()

        self.network.zero_grad()  # 清除梯度，避免累加

        return importance
    
    def compute_importance(self, data_list, labels_list, selected_layers):
        """
        计算参数重要性，并生成最终掩码以选择可训练的参数。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
            task (str): 任务名称（未使用）。
            selected_layers (str): 需要计算重要性的层名标识。
        """
        print('Compute importance for the current task...')
        
        # 生成每个域的掩码
        domain_masks = []  # List of dicts, each dict is {name: mask_tensor}
        
        for domain_idx in range(self.args['k']):
            # 从 data_list 和 labels_list 中取出当前域的数据和标签
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
    
            # 计算当前域的重要性
            importance = self.compute_score(current_data, current_labels, selected_layers)
    
            # 创建当前域的掩码
            mask = {}
            for name, param in self.network.named_parameters():
                if selected_layers in name and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
    
                    # 计算前 k 个显著性参数的索引
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), largest = False ,k=k)
    
                    # 创建掩码，确保掩码的形状与参数相同
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
    
                    mask[name] = mask_tensor
            domain_masks.append(mask)  # 保存当前域的掩码
        
        # 计算最终掩码
        final_mask = {}
    
        # 对每一层，逐域相乘掩码
        for name, param in self.network.named_parameters():
            if selected_layers in name:
                # 初始化最终掩码为第一个域的掩码
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1

                # 逐域相乘每个域的掩码
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1
    
                final_mask[name] = layer_mask
    
        # 更新参数的 requires_grad 并收集需要训练的参数
        selected_params = []
        # s = 0
        for name, param in self.network.named_parameters():
            if name in final_mask:
                # 判断该层的掩码中是否有任何一个1
                if final_mask[name].sum() > 0:
                    param.requires_grad = True
                    selected_params.append(param)
                    # s += 1
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False  # 其他层不训练
    
        # 初始化优化器，仅包含选定的参数张量
        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # print(f'Number of trainable parameters: {s}')
    
        self.mask = final_mask  # 保存最终掩码
        # print(f'Final mask computed for each layer.')
        total_count = 0
        for key, mask_tensor in self.mask.items():
            count = mask_tensor.sum().item()  # 统计每个张量中1的数量
            total_count += count
           #  print(f'Mask for {key} has {count} ones.')

        print(f'Total number of ones in the mask: {total_count}')
    
    def train_step(self, data, labels):
        """
        执行一次训练步骤。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to(param.grad.device)
                param.grad = param.grad * mask
        
        # 更新网络参数
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        
        return loss.item()
    
    def predict(self, data):
        """
        进行预测。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。

        Returns:
            Tensor: 输出 logits，形状为 [B, num_classes]。
        """
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y







class SPUr:
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        """
        初始化 SPU2 类。

        Args:
            args (dict): 参数字典，包含 'k', 'selection_rate', 'device', 'score_batch_percentage' 等。
            Tnetwork (nn.Module): 主网络模型。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            classifier (nn.Module): 分类器模型。
        """
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
    
    def compute_score(self, data, labels, selected_layers):
        """
        计算每个参数的梯度重要性。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。
            selected_layers (str): 需要计算重要性的层名标识。

        Returns:
            Dict[str, Tensor]: 每个参数名称对应的梯度重要性。
        """
        importance = {name: torch.zeros_like(param, device='cpu') 
                     for name, param in self.network.named_parameters() 
                     if selected_layers in name}
        self.network.train()
        self.network.zero_grad()

        # 前向传播
        logits_per_image = self.network(data)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss.backward()

        # 收集梯度
        for name, param in self.network.named_parameters():
            if selected_layers in name and param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().cpu().clone()

        self.network.zero_grad()  # 清除梯度，避免累加

        return importance
    
    def compute_importance(self, data_list, labels_list, selected_layers):
        """
        计算参数重要性，并生成最终掩码以选择可训练的参数。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
            task (str): 任务名称（未使用）。
            selected_layers (str): 需要计算重要性的层名标识。
        """
        print('Compute importance for the current task...')
        
        # 生成每个域的掩码
        domain_masks = []  # List of dicts, each dict is {name: mask_tensor}
        
        for domain_idx in range(self.args['k']):
            # 从 data_list 和 labels_list 中取出当前域的数据和标签
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
    
            # 计算当前域的重要性
            importance = self.compute_score(current_data, current_labels, selected_layers)
    
            # 创建当前域的掩码
            mask = {}
            for name, param in self.network.named_parameters():
                if selected_layers in name and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
    
                    # 计算前 k 个显著性参数的索引
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
    
                    # 创建掩码，确保掩码的形状与参数相同
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
    
                    mask[name] = mask_tensor
            domain_masks.append(mask)  # 保存当前域的掩码
        
        # 计算最终掩码
        final_mask = {}
    
        # 对每一层，逐域相乘掩码
        for name, param in self.network.named_parameters():
            if selected_layers in name:
                # 初始化最终掩码为第一个域的掩码
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1

                # 逐域相乘每个域的掩码
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1
    
                final_mask[name] = layer_mask
    
        # 更新参数的 requires_grad 并收集需要训练的参数
        selected_params = []
        # s = 0
        for name, param in self.network.named_parameters():
            if name in final_mask:
                # 判断该层的掩码中是否有任何一个1
                if final_mask[name].sum() > 0:
                    param.requires_grad = True
                    selected_params.append(param)
                    # s += 1
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False  # 其他层不训练
    
        # 初始化优化器，仅包含选定的参数张量
        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # print(f'Number of trainable parameters: {s}')
    
        self.mask = final_mask  # 保存最终掩码
        # print(f'Final mask computed for each layer.')
        total_count = 0
        for key, mask_tensor in self.mask.items():
            count = mask_tensor.sum().item()  # 统计每个张量中1的数量
            total_count += count
           #  print(f'Mask for {key} has {count} ones.')

        print(f'Total number of ones in the mask: {total_count}')
    
    def refine_final_mask(self, data_list, labels_list, selected_layers, r_rate=0.3):
        """
        根据每个参数的梯度重要性，重新筛选 final_mask。

        Args:
            data_list (List[Tensor]): 输入数据列表，每个元素形状为 [B, C, H, W]。
            labels_list (List[Tensor]): 标签列表，每个元素形状为 [B]。
            selected_layers (str): 需要计算梯度重要性的层名标识。
            r_rate (float): 筛选比例，选择最终掩码中最不重要的部分（例如 r_rate = 0.1 表示选取前 10% 的最不重要部分）。
        """
        # 1. 计算每个参数的梯度重要性
        # 初始化一个字典存储所有批次的累积梯度重要性
        total_importance = None

        # 遍历每个批次的数据
        for data, labels in zip(data_list, labels_list):
            # 计算每个批次的梯度重要性
            importance = self.compute_score(data, labels, selected_layers=selected_layers)
            
            # 如果是第一次迭代，初始化 total_importance
            if total_importance is None:
                total_importance = {name: imp.clone() for name, imp in importance.items()}
            else:
                # 将当前批次的重要性累加到 total_importance 中
                for name in total_importance:
                    total_importance[name] += importance[name]

        # 2. 根据累计的梯度重要性对 final_mask 进行进一步筛选
        for name, mask in self.mask.items():
            if selected_layers in name and mask.sum() > 0:  # 确保掩码中有至少一个值为 1 的部分进行筛选
                # 获取掩码中为 1 的部分
                mask_indices = mask.view(-1).nonzero(as_tuple=False).squeeze(1)

                if mask_indices.numel() == 0:
                    continue

                # 确保 mask_indices 和 total_importance 在相同设备上
                mask_indices = mask_indices.to(total_importance[name].device)

                # 获取对应的累积梯度重要性值
                param_grad_values = total_importance[name].view(-1)[mask_indices]

                # 计算需要保留的数量
                k = max(1, int(mask_indices.numel() * r_rate))

                # 选择梯度重要性最小的 k 个参数的索引
                _, smallest_k_indices = torch.topk(param_grad_values.abs(), k=k, largest=False)

                # 创建新的掩码，仅保留这些最小的梯度部分
                new_mask = torch.zeros_like(mask.view(-1), device=mask.device)
                new_mask[mask_indices[smallest_k_indices]] = 1
                self.mask[name] = new_mask.view_as(mask)  # 更新 final_mask
        
    def train_step(self, data, labels):
        """
        执行一次训练步骤。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to(param.grad.device)
                param.grad = param.grad * mask
        
        # 更新网络参数
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        
        return loss.item()
    
    def predict(self, data):
        """
        进行预测。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。

        Returns:
            Tensor: 输出 logits，形状为 [B, num_classes]。
        """
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y





class SPU5:  # 加上滑动平均
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        """
        初始化 SPU4 类，并集成滑动平均。
        
        Args:
            args (dict): 参数字典，包含 'k', 'selection_rate', 'device', 'score_batch_percentage', 'c' 等。
            Tnetwork (nn.Module): 主网络模型。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            classifier (nn.Module): 分类器模型。
        """
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.c = args['c']  # 新增超参数 c

        # 滑动平均相关
        self.swa_start = args['swa_start']
        self.swa_freq = args['swa_freq']
        self.swa_model = None  # 用于保存滑动平均的模型权重
        self.swa_n = 0  # 用于计算滑动平均的次数
        self.step = 0
    
    def compute_score(self, data, labels, selected_layers):
        """
        计算每个参数的梯度重要性。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。
            selected_layers (str): 需要计算重要性的层名标识。

        Returns:
            Dict[str, Tensor]: 每个参数名称对应的梯度重要性。
        """
        importance = {name: torch.zeros_like(param, device='cpu') 
                     for name, param in self.network.named_parameters() 
                     if selected_layers in name}
        self.network.train()
        self.network.zero_grad()

        # 前向传播
        logits_per_image = self.network(data)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss.backward()

        # 收集梯度
        for name, param in self.network.named_parameters():
            if selected_layers in name and param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().cpu().clone()

        self.network.zero_grad()  # 清除梯度，避免累加

        return importance

        return importance
    def compute_importance(self, data_list, labels_list, selected_layers):
        """
        计算参数重要性，并生成最终掩码以选择可训练的参数。
        """
        print('Compute importance for the current task...')
        
        # 统计符合名字的层数
        matching_layer_count = 0
        for name, param in self.network.named_parameters():
            if selected_layers in name:
                matching_layer_count += 1
        
        # 限制优化的层数为 N * c
        max_optimizable_layers = int(matching_layer_count * self.c)
        
        # 生成每个域的掩码
        domain_masks = []  # List of dicts, each dict is {name: mask_tensor}
        
        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
    
            importance = self.compute_score(current_data, current_labels, selected_layers)
    
            mask = {}
            for name, param in self.network.named_parameters():
                if selected_layers in name and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
    
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
    
                    mask[name] = mask_tensor
            domain_masks.append(mask)
        
        final_mask = {}
        layer_counter = 0  # 计数器，用于确定当前层是否应优化
    
        for name, param in self.network.named_parameters():
            if selected_layers in name:
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])

                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])
    
                # 如果该层超过了 N * c 限制，则不优化此层
                if layer_counter < max_optimizable_layers:
                    final_mask[name] = layer_mask
                else:
                    final_mask[name] = torch.zeros_like(param, device=self.args['device'])
                
                layer_counter += 1
    
        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                if final_mask[name].sum() > 0:
                    param.requires_grad = True
                    selected_params.append(param)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
    
        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    
        self.mask = final_mask
        total_count = 0
        for key, mask_tensor in self.mask.items():
            count = mask_tensor.sum().item()
            total_count += count

        print(f'Total number of ones in the mask: {total_count}')
    
    def train_step(self, data, labels):
        """
        执行一次训练步骤。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name]
                '''mask = self.mask[name]
                # print(name)
                if not hasattr(mask, 'to'):  # 检查mask是否有 `to` 方法，来确认它不是NoneType
                    print(f"Warning: Mask for {name} is not a valid tensor, skipping mask application.")
                    continue
                mask = mask.to(param.grad.device)'''
                if param.grad is not None:
                    # 将掩码移动到param.grad同样的设备
                    mask = mask.to(param.grad.device)
                    # 应用掩码到梯度上
                    param.grad = param.grad * mask
    
        # 更新网络参数
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        
        return loss.item()
    
    def predict(self, data):
        """
        进行预测。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。

        Returns:
            Tensor: 输出 logits，形状为 [B, num_classes]。
        """
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y



class SPU_x2:
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        """
        初始化 SPU2 类。

        Args:
            args (dict): 参数字典，包含 'k', 'selection_rate', 'device', 'score_batch_percentage' 等。
            Tnetwork (nn.Module): 主网络模型。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            classifier (nn.Module): 分类器模型。
        """
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
    
    def compute_score(self, data, labels, selected_layers):
        """
        计算每个参数的梯度重要性。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。
            selected_layers (str): 需要计算重要性的层名标识。

        Returns:
            Dict[str, Tensor]: 每个参数名称对应的梯度重要性。
        """
        importance = {name: torch.zeros_like(param, device='cpu') 
                     for name, param in self.network.named_parameters() 
                     if selected_layers in name}
        self.network.train()
        self.network.zero_grad()

        # 前向传播
        logits_per_image = self.network(data)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss.backward()

        # 收集梯度
        for name, param in self.network.named_parameters():
            if selected_layers in name and param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().cpu().clone()

        self.network.zero_grad()  # 清除梯度，避免累加

        return importance
    
    def compute_importance(self, data_list, labels_list, selected_layers):
        """
        计算参数重要性，并生成最终掩码以选择可训练的参数。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
            task (str): 任务名称（未使用）。
            selected_layers (str): 需要计算重要性的层名标识。
        """
        print('Compute importance for the current task...')
        
        # 生成每个域的掩码
        domain_masks = []  # List of dicts, each dict is {name: mask_tensor}
        
        for domain_idx in range(self.args['k']):
            # 从 data_list 和 labels_list 中取出当前域的数据和标签
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
    
            # 计算当前域的重要性
            importance = self.compute_score(current_data, current_labels, selected_layers)
    
            # 创建当前域的掩码
            mask = {}
            for name, param in self.network.named_parameters():
                if selected_layers in name and name in importance:
                    magnitudes = importance[name]**2
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
    
                    # 计算前 k 个显著性参数的索引
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
    
                    # 创建掩码，确保掩码的形状与参数相同
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
    
                    mask[name] = mask_tensor
            domain_masks.append(mask)  # 保存当前域的掩码
        
        # 计算最终掩码
        final_mask = {}
    
        # 对每一层，逐域相乘掩码
        for name, param in self.network.named_parameters():
            if selected_layers in name:
                # 初始化最终掩码为第一个域的掩码
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1

                # 逐域相乘每个域的掩码
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1
    
                final_mask[name] = layer_mask
    
        # 更新参数的 requires_grad 并收集需要训练的参数
        selected_params = []
        # s = 0
        for name, param in self.network.named_parameters():
            if name in final_mask:
                # 判断该层的掩码中是否有任何一个1
                if final_mask[name].sum() > 0:
                    param.requires_grad = True
                    selected_params.append(param)
                    # s += 1
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False  # 其他层不训练
    
        # 初始化优化器，仅包含选定的参数张量
        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        # print(f'Number of trainable parameters: {s}')
    
        self.mask = final_mask  # 保存最终掩码
        # print(f'Final mask computed for each layer.')
        total_count = 0
        for key, mask_tensor in self.mask.items():
            count = mask_tensor.sum().item()  # 统计每个张量中1的数量
            total_count += count
           #  print(f'Mask for {key} has {count} ones.')

        print(f'Total number of ones in the mask: {total_count}')
    
    def train_step(self, data, labels):
        """
        执行一次训练步骤。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to(param.grad.device)
                param.grad = param.grad * mask
        
        # 更新网络参数
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        
        return loss.item()
    
    def predict(self, data):
        """
        进行预测。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。

        Returns:
            Tensor: 输出 logits，形状为 [B, num_classes]。
        """
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y



class SPU6:
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        """
        初始化 SPU5 类，增加基于层的重要性选择优化层的功能。
        """
        self.args = args  # args 是字典
        self.network = Tnetwork  # 主网络
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.c = args['c']  # 超参数 c，代表前 c% 的重要层被优化

        # 滑动平均相关参数（如果需要，可以进一步实现）
        self.swa_start = args['swa_start']
        self.swa_freq = args['swa_freq']
        self.swa_model = None
        self.swa_n = 0
        self.step = 0

    def compute_score(self, data, labels, selected_layers):
        """
        计算每个参数的梯度重要性。
        """
        importance = {name: torch.zeros_like(param, device='cpu')
                      for name, param in self.network.named_parameters()
                      if selected_layers in name}
        self.network.train()
        self.network.zero_grad()

        # 前向传播
        logits_per_image = self.network(data)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss.backward()

        # 收集梯度
        for name, param in self.network.named_parameters():
            if selected_layers in name and param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().cpu().clone()

        self.network.zero_grad()  # 清除梯度，避免累加
        return importance

    def compute_importance(self, data_list, labels_list, selected_layers):
        """
        计算参数重要性，并根据层的重要性选择哪些层需要优化。
        """
        print('Compute importance for the current task...')

        # 生成每个域的掩码
        domain_masks = []  # 每个域对应一个掩码的列表
        
        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]

            importance = self.compute_score(current_data, current_labels, selected_layers)
            mask = {}
            for name, param in self.network.named_parameters():
                if selected_layers in name and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)

                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1

                    mask[name] = mask_tensor
            domain_masks.append(mask)

        # 计算每层的重要性
        layer_importance = {}
        for name, param in self.network.named_parameters():
            if selected_layers in name:
                total_importance = sum(domain_masks[domain_idx][name].sum().item() 
                                       for domain_idx in range(self.args['k']) 
                                       if name in domain_masks[domain_idx])
                layer_importance[name] = total_importance

        # 按照重要性排序，选出前 c% 的层
        sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
        num_layers_to_optimize = int(len(sorted_layers) * self.c)

        # 根据重要性选取需要优化的层
        final_mask = {}
        for idx, (name, importance) in enumerate(sorted_layers):
            if idx < num_layers_to_optimize:
                # 保留选中的层的掩码
                layer_mask = domain_masks[0][name].clone()
                for domain_idx in range(1, self.args['k']):
                    layer_mask = layer_mask * domain_masks[domain_idx].get(name, torch.ones_like(layer_mask))
                final_mask[name] = layer_mask
            else:
                # 没有被选中的层，设置为全 0 掩码
                final_mask[name] = torch.zeros_like(param, device=self.args['device'])

        # 选择需要优化的参数
        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask and final_mask[name].sum() > 0:
                param.requires_grad = True
                selected_params.append(param)
            else:
                param.requires_grad = False

        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.mask = final_mask
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

    def train_step(self, data, labels):
        """
        执行一次训练步骤。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()

        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()

        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()

        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask and param.grad is not None:
                mask = self.mask[name].to(param.grad.device)
                param.grad *= mask

        # 更新网络参数
        self.optimizer.step()

        # 更新分类器参数
        self.optimizer2.step()

        return loss.item()

    def predict(self, data):
        """
        进行预测。
        """
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y

'''from sklearn.cluster import KMeans
class SPU_kmeans:
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        """
        初始化 SPU2 类。

        Args:
            args (dict): 参数字典，包含 'k', 'selection_rate', 'device', 'score_batch_percentage' 等。
            Tnetwork (nn.Module): 主网络模型。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            classifier (nn.Module): 分类器模型。
        """
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
    
    def compute_score(self, data, labels, selected_layers):
        
        importance = {name: torch.zeros_like(param, device='cpu') 
                     for name, param in self.network.named_parameters() 
                     if selected_layers in name}
        self.network.train()
        self.network.zero_grad()

        # 前向传播
        logits_per_image = self.network(data)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss.backward()

        # 收集梯度
        for name, param in self.network.named_parameters():
            if selected_layers in name and param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().cpu().clone()

        self.network.zero_grad()  # 清除梯度，避免累加

        return importance
    
    def compute_importance(self, data_list, labels_list, selected_layers):
        """
        计算参数重要性，并生成最终掩码以选择可训练的参数。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
            task (str): 任务名称（未使用）。
            selected_layers (str): 需要计算重要性的层名标识。
        """
        print('Compute importance for the current task...')

        # 生成每个域的掩码
        domain_masks = []  # List of dicts, each dict is {name: mask_tensor}
        
        for domain_idx in range(self.args['k']):
            # 从 data_list 和 labels_list 中取出当前域的数据和标签
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]

            # 计算当前域的重要性
            importance = self.compute_score(current_data, current_labels, selected_layers)

            # 创建当前域的掩码
            mask = {}
            for name, param in self.network.named_parameters():
                if selected_layers in name and name in importance:
                    magnitudes = importance[name].abs()

                    # 将显著性值展平成1D向量
                    magnitudes_flat = magnitudes.view(-1).cpu().numpy().reshape(-1, 1)

                    # 使用 KMeans 聚类，将显著性值聚成两类
                    kmeans = KMeans(n_clusters=2,n_init=10)
                    clusters = kmeans.fit_predict(magnitudes_flat)

                    # 找出两个聚类的中心
                    cluster_centers = kmeans.cluster_centers_

                    # 获取值较大的聚类标签
                    high_importance_label = cluster_centers.argmax()

                    # 创建掩码，选择属于高重要性类别的参数
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    high_importance_indices = (clusters == high_importance_label).nonzero()[0]
                    
                    # 将高重要性参数的掩码设置为1
                    mask_tensor.view(-1)[high_importance_indices] = 1

                    mask[name] = mask_tensor

            domain_masks.append(mask)  # 保存当前域的掩码

        # 计算最终掩码
        final_mask = {}

        # 对每一层，逐域相乘掩码
        for name, param in self.network.named_parameters():
            if selected_layers in name:
                # 初始化最终掩码为第一个域的掩码
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1

                # 逐域相乘每个域的掩码
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1

                final_mask[name] = layer_mask

        # 更新参数的 requires_grad 并收集需要训练的参数
        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                # 判断该层的掩码中是否有任何一个1
                if final_mask[name].sum() > 0:
                    param.requires_grad = True
                    selected_params.append(param)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False  # 其他层不训练

        # 初始化优化器，仅包含选定的参数张量
        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.mask = final_mask  # 保存最终掩码
        total_count = 0
        for key, mask_tensor in self.mask.items():
            count = mask_tensor.sum().item()  # 统计每个张量中1的数量
            total_count += count

        print(f'Total number of ones in the mask: {total_count}')
    
    def train_step(self, data, labels):
        """
        执行一次训练步骤。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to(param.grad.device)
                param.grad = param.grad * mask
        
        # 更新网络参数
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        
        return loss.item()
    
    def predict(self, data):
        """
        进行预测。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。

        Returns:
            Tensor: 输出 logits，形状为 [B, num_classes]。
        """
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y

'''

class SPU_list: # 匹配的是一个列表不再是一个值
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        """
        初始化 SPU3 类。

        Args:
            args (dict): 参数字典，包含 'k', 'selection_rate', 'device', 'score_batch_percentage' 等。
            Tnetwork (nn.Module): 主网络模型。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            classifier (nn.Module): 分类器模型。
        """
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.selected_layers = args['selected_layers']  # 选择的层名称列表
        self.selected_layers = self.selected_layers.split()
        # print(self.selected_layers)

    def compute_score(self, data, labels,name = None):
        """
        计算每个参数的梯度重要性。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            Dict[str, Tensor]: 每个参数名称对应的梯度重要性。
        """
        importance = {name: torch.zeros_like(param, device='cpu') 
                     for name, param in self.network.named_parameters() 
                     if any(layer_name in name for layer_name in self.selected_layers)}  # 检查是否包含任意一个子字符串
        self.network.train()
        self.network.zero_grad()

        # 前向传播
        logits_per_image = self.network(data)
        loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss.backward()

        # 收集梯度
        for name, param in self.network.named_parameters():
            # print(name)
            if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
                importance[name] += param.grad.detach().cpu().clone()

        self.network.zero_grad()  # 清除梯度，避免累加

        return importance

    def train_step(self, data, labels):
        """
        执行一次训练步骤。 

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to('cuda') # param.grad.device
                if param.grad is None:
                    print(f"Parameter {name} has no gradient. requires_grad={param.requires_grad}")
                else:
                    param.grad = param.grad * mask
        
        # 更新网络参数
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        
      
        return loss.item()
    def compute_importance(self, data_list, labels_list, name=None):
        """
        计算参数重要性，并生成最终掩码以选择可训练的参数。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        print('Compute importance for the current task...')
        
        # 生成每个域的掩码
        domain_masks = []  # List of dicts, each dict is {name: mask_tensor}
        
        for domain_idx in range(self.args['k']):
            # 从 data_list 和 labels_list 中取出当前域的数据和标签
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]

            # 计算当前域的重要性
            importance = self.compute_score(current_data, current_labels)

            # 创建当前域的掩码
            mask = {}
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))

                    # 计算前 k 个显著性参数的索引
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)

                    # 创建掩码，确保掩码的形状与参数相同
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1

                    mask[name] = mask_tensor
            domain_masks.append(mask)  # 保存当前域的掩码
        
        # 计算最终掩码
        final_mask = {}

        # 对每一层，逐域相乘掩码
        for name, param in self.network.named_parameters():
            if any(layer_name in name for layer_name in self.selected_layers):  # 使用any检查
                # 初始化最终掩码为第一个域的掩码
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1

                # 逐域相乘每个域的掩码
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])  # 如果没有该层的掩码，默认为1

                final_mask[name] = layer_mask

        # 更新参数的 requires_grad 并收集需要训练的参数
        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                # 如果该层的掩码全部为0，删除该层的参数
                if final_mask[name].sum() == 0:
                    print(f"Layer {name} is excluded from training (mask is all zeros).")
                    # 直接从 final_mask 中删除该层
                    del final_mask[name]  # 删除该层的掩码

                else:
                    param.requires_grad = True
                    selected_params.append(param)
            else:
                param.requires_grad = False  # 其他层不训练

        # 初始化优化器，仅包含选定的参数张量
        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.mask = final_mask  # 保存最终掩码
        total_count = 0
        for key, mask_tensor in self.mask.items():
            count = mask_tensor.sum().item()  # 统计每个张量中1的数量
            total_count += count

        print(f'Total number of ones in the mask: {total_count}')
    

        
    
    def predict(self, data):
        """
        进行预测。

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。

        Returns:
            Tensor: 输出 logits，形状为 [B, num_classes]。
        """
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y

import pickle

# 这是标准模型 接下来所有的都从这里出发
class SPU_step:  # 匹配的是一个列表不再是一个值
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.selected_layers = args['selected_layers'].split()
        self.way = args['way']

    def compute_score(self, data_list, labels_list, name=None):
        # 初始化 importance 字典
        importance = {name: torch.zeros_like(param, device='cpu') 
                    for name, param in self.network.named_parameters() 
                    if any(layer_name in name for layer_name in self.selected_layers)}

        # 设置网络为训练模式
        self.network.train()
        self.network.zero_grad()

        # 遍历每个数据批次
        for data, labels in zip(data_list, labels_list):
            # 前向传播
            logits_per_image = self.network(data)  # 获取当前批次的输出
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度

            # 累加梯度
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.detach().cpu().clone()  # 累加梯度

            # 清除梯度，避免累加
            self.network.zero_grad()

        return importance

    def compute_importance(self, data_list, labels_list, name=None, save = False ):
        print('Compute importance for the current task...')
        save_impor = False
        domain_masks = []
        if save_impor:
            domain_importance = []
        # print(len(data_list))
        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance = self.compute_score(current_data, current_labels)
            if save_impor:
                domain_importance.append(importance)

            mask = {}
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
                    mask[name] = mask_tensor
            domain_masks.append(mask)
        if save_impor:
            save_dir = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_r_pacs_vit.pth'
            self.save_masks_and_importance(domain_masks,domain_importance,save_dir)

        
            


        # if False: # 存一下各个域的掩码方便研究
        #     combined_domain_mask = {}

        #     # 遍历每个域，保存对应的掩码
        #     for domain_idx, domain_mask in enumerate(domain_masks):
        #         # 对每个域的掩码进行转换，确保它们在 CPU 上
        #         cpu_domain_mask = {name: mask.cpu() for name, mask in domain_mask.items()}
        #         combined_domain_mask[f'domain_{domain_idx}'] = cpu_domain_mask

        #     # 保存所有域的掩码为一个文件
        #     torch.save(combined_domain_mask, '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_masks.pth')
        #     print("Domain masks have been saved as 'combined_domain_masks.pth'.")
        
            # save_masks_and_importance(domain_masks, domain_importance, combined_save_path)
        
        final_mask = {}
        # overlap_ratios = {}  # 存储每一层的重叠率

        for name, param in self.network.named_parameters():
            # 判断是否属于指定层
            if any(layer_name in name for layer_name in self.selected_layers):
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])
                
                # 依次与其他domain_masks相乘
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])
                
                # 保存到final_mask
                final_mask[name] = layer_mask

                # # 计算剩余部分占比原mask的比值，实际上就是重叠率
                # original_count = torch.sum(domain_masks[0][name])  # 全1矩阵的总和
                # final_mask_count = torch.sum(layer_mask == 1)       # final_mask中1的总数
                # overlap_ratios[name] = (final_mask_count / original_count).item()  # 计算比率并保存

        # 打印每一层的重叠率,等要用的时候再打开
        # for layer_name, overlap in overlap_ratios.items():
        #     print(f"Layer: {layer_name}, Overlap Ratio: {overlap:.4f}")

        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                if final_mask[name].sum() == 0:
                    print(f"Layer {name} is excluded from training (mask is all zeros).")
                    del final_mask[name]
                else:
                    param.requires_grad = True
                    selected_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.mask = final_mask
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

        # 第二步：基于梯度方差优化最终掩码
        if self.way == 'var':
            self.optimize_final_mask_by_variance(data_list, labels_list)
        elif self.way == 'dir':
            self.optimize_final_mask_by_center_direction(data_list, labels_list)
        elif self.way == 'mean':
            self.optimize_final_mask_by_mean(data_list, labels_list)
        elif self.way == 'qua':
            self.optimize_final_mask_by_quantile(data_list, labels_list)
        else:
            print('Select withou step2.')

    def optimize_final_mask_by_variance(self, data_list, labels_list,save = False):
        save = True
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
                # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')
        if save:
            saved_masked_grad_vars = {}

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                 
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差
                    grad_var = grad_var.to(original_mask.device)
                    # print(f"grad_var device: {grad_var.device}")
                    # print(f"original_mask device: {original_mask.device}")
                    masked_grad_var = grad_var[original_mask > 0]
                    if save:
                        saved_masked_grad_vars[name] = masked_grad_var.cpu().numpy()

                    # 计算被掩码部分的均值
                    masked_mean = masked_grad_var.mean()

                    # 根据方差生成掩码（不使用 for 循环）
                    optimized_mask = (grad_var < masked_mean).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask
        # save_path = "/home/home_node7/wzb/自己算法/miro-main2/result/draw/masked_grad_vars.pkl"
        # if save:
        #     with open(save_path, "wb") as f:
        #         pickle.dump(saved_masked_grad_vars, f)

        #     print(f"Saved masked_grad_var to {save_path}")
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')
    
    def optimize_final_mask_by_quantile(self, data_list, labels_list):
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差

                    # 计算分位数阈值并生成掩码
                    quantile_threshold = self.args['quantile_threshold']  # 需要提前定义的超参数
                    threshold = torch.quantile(grad_var, quantile_threshold)
                    optimized_mask = (grad_var < threshold).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

    
    def optimize_final_mask_by_mean(self, data_list, labels_list):
        """
        通过平均梯度大小进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的梯度
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        # Step 2: 计算梯度平均值并根据大小设置掩码
        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_stack = torch.stack(all_importance[name])  # 堆叠所有域的梯度
                    grad_mean = param_grad_stack.mean(dim=0)  # 计算每个维度的平均值

                    # 打印设备信息，检查 grad_mean 和 original_mask 是否在同一设备上
                    # print(f"grad_mean device: {grad_mean.device}")
                    # print(f"original_mask device: {original_mask.device}")

                    # 将 grad_mean 移动到与其他张量相同的设备（CUDA）上
                    grad_mean = grad_mean.to(original_mask.device)

                    # 计算平均值的排序
                    sorted_grad_values, sorted_indices = torch.sort(grad_mean, dim=0, descending=True)

                    # 获取阈值，前50%设为1，后50%设为0.5
                    threshold = sorted_grad_values[int(len(sorted_grad_values) * 0.5)]

                    # 确保阈值移到正确的设备上
                    threshold = threshold.to(original_mask.device)

                    # 根据平均梯度值对掩码进行更新
                    optimized_mask = torch.where(grad_mean >= threshold, 
                                                torch.tensor(1.0, device=original_mask.device), 
                                                torch.tensor(0.5, device=original_mask.device))

                    # 更新掩码
                    # print(optimized_mask.device)
                    # print(original_mask.device)
                    self.mask[name] = optimized_mask * original_mask

        # 打印更新后的掩码中 1 的数量
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')




    def optimize_final_mask_by_center_direction(self, data_list, labels_list):
        """
        通过梯度方向与中心向量的相似度进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性（通过中心向量）
        all_center_vectors = {}  # 用于存储每层的中心向量

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 计算每层的中心向量
            for name, importance in importance_scores.items():
                if name not in all_center_vectors:
                    all_center_vectors[name] = []
                all_center_vectors[name].append(importance.detach().cpu())

        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的中心向量
                if name in all_center_vectors:
                    center_vectors = torch.stack(all_center_vectors[name])  # 计算得到所有域的梯度中心向量
                    center_vector = center_vectors.mean(dim=0)  # 对所有域的梯度进行平均，得到中心向量

                    # Step 2: 计算每个梯度向量与中心向量的余弦相似度
                    grad_stack = torch.stack([importance.detach().cpu() for importance in all_center_vectors[name]])  # 收集每个域的梯度
                    grad_norm = grad_stack / grad_stack.norm(dim=-1, keepdim=True)  # 对梯度向量进行归一化

                    # 计算所有域的梯度与中心向量的余弦相似度
                    center_vector_norm = center_vector / center_vector.norm()

                    # **广播中心向量**，确保它的形状为 (batch_size, feature_dim)
                    # 如果center_vector是[768]，则扩展为[batch_size, 768]
                    center_vector_norm_expanded = center_vector_norm.unsqueeze(0).expand(grad_norm.size(0), -1)

                    # 计算余弦相似度
                    cosine_similarities = torch.sum(grad_norm * center_vector_norm_expanded, dim=-1)  # 计算余弦相似度

                    # Step 3: 根据相似度筛选维度
                    grad_sim_mask = cosine_similarities.mean(dim=0)  # 对每个维度的相似度取平均
                    grad_sim_mask = (grad_sim_mask >= grad_sim_mask.mean()).float()  # 根据相似度的平均值生成掩码

                    # 更新掩码
                    self.mask[name] = grad_sim_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')



    def train_step(self, data, labels):
        """
        执行一次训练步骤。 

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                # mask = self.mask[name].to('cuda') # param.grad.device
                if param.grad is None:
                    print(f"Parameter {name} has no gradient. requires_grad={param.requires_grad}")
                else:
                    param.grad = param.grad * self.mask[name]
        memory1 = torch.cuda.memory_allocated() / (1024 ** 3)
        
        # 更新网络参数
        
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        return [loss.item(),memory1]

    def predict(self, data):
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y
    
    def analyze_mask(self):
        results = {}
        for name, matrix in self.mask.items():
            if not isinstance(matrix, torch.Tensor):
                raise ValueError(f"The value associated with key '{name}' is not a tensor.")
            
            # 跳过第二个维度不大于 2 的矩阵
            if matrix.dim() < 2 or matrix.size(1) <= 2:
                continue
            
            # 计算矩阵中1的个数
            ones_count = torch.sum(matrix).item()
            
            # 确保矩阵是浮点类型
            matrix_float = matrix.to(dtype=torch.float32)
            
            # 计算矩阵的秩
            _, singular_values, _ = torch.linalg.svd(matrix_float)
            rank = torch.sum(singular_values > 1e-10).item()
            
            # 保存结果
            results[name] = {"ones_count": ones_count, "rank": rank}
            print(name,ones_count,rank)
        
        # return results

    def save_masks_and_importance(self,domain_masks, domain_importance, save_path):
        """
        保存域的掩码和重要性信息到指定路径。
        
        参数:
        - domain_masks: List[Dict[str, Tensor]]，每个域的掩码字典。
        - domain_importance: List[Dict[str, Tensor]]，每个域的梯度/重要性字典。
        - save_path: str，保存文件的路径。
        """
        combined_data = {}

        # 遍历每个域，保存对应的掩码和重要性信息
        for domain_idx, (domain_mask, domain_imp) in enumerate(zip(domain_masks, domain_importance)):
            # 对每个域的掩码和重要性信息进行转换，确保它们在 CPU 上
            cpu_domain_mask = {name: mask.cpu() for name, mask in domain_mask.items()}
            cpu_domain_importance = {name: imp.cpu() for name, imp in domain_imp.items()}
            combined_data[f'domain_{domain_idx}'] = {
                'masks': cpu_domain_mask,
                'importance': cpu_domain_importance
            }

        # 保存所有域的数据为一个文件
        torch.save(combined_data, save_path)
        print(f"Masks and importance have been saved to '{save_path}'.")







class SPU_step:  # 匹配的是一个列表不再是一个值
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.selected_layers = args['selected_layers'].split()
        self.way = args['way']

    def compute_score(self, data_list, labels_list, name=None):
        # 初始化 importance 字典
        importance = {name: torch.zeros_like(param, device='cpu') 
                    for name, param in self.network.named_parameters() 
                    if any(layer_name in name for layer_name in self.selected_layers)}

        # 设置网络为训练模式
        self.network.train()
        self.network.zero_grad()

        # 遍历每个数据批次
        for data, labels in zip(data_list, labels_list):
            # 前向传播
            logits_per_image = self.network(data)  # 获取当前批次的输出
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度

            # 累加梯度
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.detach().cpu().clone()  # 累加梯度

            # 清除梯度，避免累加
            self.network.zero_grad()

        return importance

    def compute_importance(self, data_list, labels_list, name=None, save = False ):
        print('Compute importance for the current task...')
        save_impor = False
        domain_masks = []
        if save_impor:
            domain_importance = []
        # print(len(data_list))
        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance = self.compute_score(current_data, current_labels)
            if save_impor:
                domain_importance.append(importance)

            mask = {}
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
                    mask[name] = mask_tensor
            domain_masks.append(mask)
        if save_impor:
            save_dir = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_r_pacs_vit.pth'
            self.save_masks_and_importance(domain_masks,domain_importance,save_dir)

        
            


        # if False: # 存一下各个域的掩码方便研究
        #     combined_domain_mask = {}

        #     # 遍历每个域，保存对应的掩码
        #     for domain_idx, domain_mask in enumerate(domain_masks):
        #         # 对每个域的掩码进行转换，确保它们在 CPU 上
        #         cpu_domain_mask = {name: mask.cpu() for name, mask in domain_mask.items()}
        #         combined_domain_mask[f'domain_{domain_idx}'] = cpu_domain_mask

        #     # 保存所有域的掩码为一个文件
        #     torch.save(combined_domain_mask, '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_masks.pth')
        #     print("Domain masks have been saved as 'combined_domain_masks.pth'.")
        
            # save_masks_and_importance(domain_masks, domain_importance, combined_save_path)
        
        final_mask = {}
        # overlap_ratios = {}  # 存储每一层的重叠率

        for name, param in self.network.named_parameters():
            # 判断是否属于指定层
            if any(layer_name in name for layer_name in self.selected_layers):
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])
                
                # 依次与其他domain_masks相乘
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])
                
                # 保存到final_mask
                final_mask[name] = layer_mask

                # # 计算剩余部分占比原mask的比值，实际上就是重叠率
                # original_count = torch.sum(domain_masks[0][name])  # 全1矩阵的总和
                # final_mask_count = torch.sum(layer_mask == 1)       # final_mask中1的总数
                # overlap_ratios[name] = (final_mask_count / original_count).item()  # 计算比率并保存

        # 打印每一层的重叠率,等要用的时候再打开
        # for layer_name, overlap in overlap_ratios.items():
        #     print(f"Layer: {layer_name}, Overlap Ratio: {overlap:.4f}")

        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                if final_mask[name].sum() == 0:
                    print(f"Layer {name} is excluded from training (mask is all zeros).")
                    del final_mask[name]
                else:
                    param.requires_grad = True
                    selected_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.mask = final_mask
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

        # 第二步：基于梯度方差优化最终掩码
        if self.way == 'var':
            self.optimize_final_mask_by_variance(data_list, labels_list)
        elif self.way == 'dir':
            self.optimize_final_mask_by_center_direction(data_list, labels_list)
        elif self.way == 'mean':
            self.optimize_final_mask_by_mean(data_list, labels_list)
        elif self.way == 'qua':
            self.optimize_final_mask_by_quantile(data_list, labels_list)
        else:
            print('Select withou step2.')

    def optimize_final_mask_by_variance(self, data_list, labels_list,save = False):
        save = True
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
                # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')
        if save:
            saved_masked_grad_vars = {}

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                 
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差
                    grad_var = grad_var.to(original_mask.device)
                    # print(f"grad_var device: {grad_var.device}")
                    # print(f"original_mask device: {original_mask.device}")
                    masked_grad_var = grad_var[original_mask > 0]
                    if save:
                        saved_masked_grad_vars[name] = masked_grad_var.cpu().numpy()

                    # 计算被掩码部分的均值
                    masked_mean = masked_grad_var.mean()

                    # 根据方差生成掩码（不使用 for 循环）
                    optimized_mask = (grad_var < masked_mean).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask
        # save_path = "/home/home_node7/wzb/自己算法/miro-main2/result/draw/masked_grad_vars.pkl"
        # if save:
        #     with open(save_path, "wb") as f:
        #         pickle.dump(saved_masked_grad_vars, f)

        #     print(f"Saved masked_grad_var to {save_path}")
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')
    
    def optimize_final_mask_by_quantile(self, data_list, labels_list):
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差

                    # 计算分位数阈值并生成掩码
                    quantile_threshold = self.args['quantile_threshold']  # 需要提前定义的超参数
                    threshold = torch.quantile(grad_var, quantile_threshold)
                    optimized_mask = (grad_var < threshold).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

    
    def optimize_final_mask_by_mean(self, data_list, labels_list):
        """
        通过平均梯度大小进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的梯度
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        # Step 2: 计算梯度平均值并根据大小设置掩码
        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_stack = torch.stack(all_importance[name])  # 堆叠所有域的梯度
                    grad_mean = param_grad_stack.mean(dim=0)  # 计算每个维度的平均值

                    # 打印设备信息，检查 grad_mean 和 original_mask 是否在同一设备上
                    # print(f"grad_mean device: {grad_mean.device}")
                    # print(f"original_mask device: {original_mask.device}")

                    # 将 grad_mean 移动到与其他张量相同的设备（CUDA）上
                    grad_mean = grad_mean.to(original_mask.device)

                    # 计算平均值的排序
                    sorted_grad_values, sorted_indices = torch.sort(grad_mean, dim=0, descending=True)

                    # 获取阈值，前50%设为1，后50%设为0.5
                    threshold = sorted_grad_values[int(len(sorted_grad_values) * 0.5)]

                    # 确保阈值移到正确的设备上
                    threshold = threshold.to(original_mask.device)

                    # 根据平均梯度值对掩码进行更新
                    optimized_mask = torch.where(grad_mean >= threshold, 
                                                torch.tensor(1.0, device=original_mask.device), 
                                                torch.tensor(0.5, device=original_mask.device))

                    # 更新掩码
                    # print(optimized_mask.device)
                    # print(original_mask.device)
                    self.mask[name] = optimized_mask * original_mask

        # 打印更新后的掩码中 1 的数量
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')




    def optimize_final_mask_by_center_direction(self, data_list, labels_list):
        """
        通过梯度方向与中心向量的相似度进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性（通过中心向量）
        all_center_vectors = {}  # 用于存储每层的中心向量

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 计算每层的中心向量
            for name, importance in importance_scores.items():
                if name not in all_center_vectors:
                    all_center_vectors[name] = []
                all_center_vectors[name].append(importance.detach().cpu())

        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的中心向量
                if name in all_center_vectors:
                    center_vectors = torch.stack(all_center_vectors[name])  # 计算得到所有域的梯度中心向量
                    center_vector = center_vectors.mean(dim=0)  # 对所有域的梯度进行平均，得到中心向量

                    # Step 2: 计算每个梯度向量与中心向量的余弦相似度
                    grad_stack = torch.stack([importance.detach().cpu() for importance in all_center_vectors[name]])  # 收集每个域的梯度
                    grad_norm = grad_stack / grad_stack.norm(dim=-1, keepdim=True)  # 对梯度向量进行归一化

                    # 计算所有域的梯度与中心向量的余弦相似度
                    center_vector_norm = center_vector / center_vector.norm()

                    # **广播中心向量**，确保它的形状为 (batch_size, feature_dim)
                    # 如果center_vector是[768]，则扩展为[batch_size, 768]
                    center_vector_norm_expanded = center_vector_norm.unsqueeze(0).expand(grad_norm.size(0), -1)

                    # 计算余弦相似度
                    cosine_similarities = torch.sum(grad_norm * center_vector_norm_expanded, dim=-1)  # 计算余弦相似度

                    # Step 3: 根据相似度筛选维度
                    grad_sim_mask = cosine_similarities.mean(dim=0)  # 对每个维度的相似度取平均
                    grad_sim_mask = (grad_sim_mask >= grad_sim_mask.mean()).float()  # 根据相似度的平均值生成掩码

                    # 更新掩码
                    self.mask[name] = grad_sim_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')



    def train_step(self, data, labels):
        """
        执行一次训练步骤。 

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                # mask = self.mask[name].to('cuda') # param.grad.device
                if param.grad is None:
                    print(f"Parameter {name} has no gradient. requires_grad={param.requires_grad}")
                else:
                    param.grad = param.grad * self.mask[name]
        memory1 = torch.cuda.memory_allocated() / (1024 ** 3)
        
        # 更新网络参数
        
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        return [loss.item(),memory1]

    def predict(self, data):
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y
    
    def analyze_mask(self):
        results = {}
        for name, matrix in self.mask.items():
            if not isinstance(matrix, torch.Tensor):
                raise ValueError(f"The value associated with key '{name}' is not a tensor.")
            
            # 跳过第二个维度不大于 2 的矩阵
            if matrix.dim() < 2 or matrix.size(1) <= 2:
                continue
            
            # 计算矩阵中1的个数
            ones_count = torch.sum(matrix).item()
            
            # 确保矩阵是浮点类型
            matrix_float = matrix.to(dtype=torch.float32)
            
            # 计算矩阵的秩
            _, singular_values, _ = torch.linalg.svd(matrix_float)
            rank = torch.sum(singular_values > 1e-10).item()
            
            # 保存结果
            results[name] = {"ones_count": ones_count, "rank": rank}
            print(name,ones_count,rank)
        
        # return results

    def save_masks_and_importance(self,domain_masks, domain_importance, save_path):
        """
        保存域的掩码和重要性信息到指定路径。
        
        参数:
        - domain_masks: List[Dict[str, Tensor]]，每个域的掩码字典。
        - domain_importance: List[Dict[str, Tensor]]，每个域的梯度/重要性字典。
        - save_path: str，保存文件的路径。
        """
        combined_data = {}

        # 遍历每个域，保存对应的掩码和重要性信息
        for domain_idx, (domain_mask, domain_imp) in enumerate(zip(domain_masks, domain_importance)):
            # 对每个域的掩码和重要性信息进行转换，确保它们在 CPU 上
            cpu_domain_mask = {name: mask.cpu() for name, mask in domain_mask.items()}
            cpu_domain_importance = {name: imp.cpu() for name, imp in domain_imp.items()}
            combined_data[f'domain_{domain_idx}'] = {
                'masks': cpu_domain_mask,
                'importance': cpu_domain_importance
            }

        # 保存所有域的数据为一个文件
        torch.save(combined_data, save_path)
        print(f"Masks and importance have been saved to '{save_path}'.")






class SPU_step_random:  # 匹配的是一个列表不再是一个值
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.selected_layers = args['selected_layers'].split()
        self.way = args['way']
        
    def compute_score(self, data_list, labels_list, name=None):
        # 初始化 importance 字典
        importance = {name: torch.zeros_like(param, device='cpu') 
                    for name, param in self.network.named_parameters() 
                    if any(layer_name in name for layer_name in self.selected_layers)}

        # 设置网络为训练模式
        self.network.train()
        self.network.zero_grad()

        # 遍历每个数据批次
        for data, labels in zip(data_list, labels_list):
            # 前向传播
            logits_per_image = self.network(data)  # 获取当前批次的输出
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度

            # 累加梯度
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.detach().cpu().clone()  # 累加梯度

            # 清除梯度，避免累加
            self.network.zero_grad()

        return importance

    def compute_importance(self, data_list, labels_list, name=None, save=False):
        print('Compute RANDOM importance for the current task...')
        save_impor = False
        domain_masks = []
        if save_impor:
            domain_importance = []
        
        # 生成各domain的随机mask
        for domain_idx in range(self.args['k']):
            mask = {}
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers):
                    # 创建伯努利分布的随机mask
                    mask_tensor = torch.rand_like(param, device=self.args['device']) < self.args['selection_rate']
                    mask_tensor = mask_tensor.float()  # 转换为0/1的浮点tensor
                    mask[name] = mask_tensor
                    if  torch.all(mask_tensor == 0):
                        print('wrong tensor!')

            domain_masks.append(mask)
        
        # 合并各domain的mask（根据merge_method选择交集或并集）
        final_mask = {}
        merge_method = self.args.get('mask_merge_method', 'union')  # 默认为并集

        for name, param in self.network.named_parameters():
            if any(layer_name in name for layer_name in self.selected_layers):
                # 收集所有domain的该层mask（缺失的根据merge_method补全）
                all_masks = []
                for domain_mask in domain_masks:
                    if name in domain_mask:
                        all_masks.append(domain_mask[name])
                    else:
                        # 缺失mask时：交集视为1，并集视为0
                        fill_value = 1.0 if merge_method == 'intersection' else 0.0
                        all_masks.append(torch.full_like(param, fill_value, device=param.device))
                
                # 合并所有mask
                if merge_method == 'intersection':
                    layer_mask = torch.ones_like(param, device=param.device)
                    for mask in all_masks:
                        layer_mask = layer_mask * mask  # 逐元素相乘
                elif merge_method == 'union':
                    layer_mask = torch.zeros_like(param, device=param.device)
                    for mask in all_masks:
                        layer_mask = torch.max(layer_mask, mask)  # 逐元素取大
                
                # 强制至少保留一个参数（防止全0）
                if torch.all(layer_mask == 0):
                    random_idx = torch.randint(0, layer_mask.numel(), (1,))
                    layer_mask.view(-1)[random_idx] = 1
                
                final_mask[name] = layer_mask
        
        # 选择参数并设置优化器
        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                if final_mask[name].sum() == 0:
                    print(f"Layer {name} is excluded from training (mask is all zeros).")
                    del final_mask[name]
                else:
                    param.requires_grad = True
                    selected_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.mask = final_mask
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')
        print(f'Mask merge method used: {merge_method}')

        # # 保留后续的优化步骤
        # if self.way == 'var':
        #     self.optimize_final_mask_by_variance(data_list, labels_list)
        # elif self.way == 'dir':
        #     self.optimize_final_mask_by_center_direction(data_list, labels_list)
        # elif self.way == 'mean':
        #     self.optimize_final_mask_by_mean(data_list, labels_list)
        # elif self.way == 'qua':
        #     self.optimize_final_mask_by_quantile(data_list, labels_list)
        # else:
        #     print('Select without step2.')

    def optimize_final_mask_by_variance(self, data_list, labels_list,save = False):
        save = True
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
                # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')
        if save:
            saved_masked_grad_vars = {}

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                 
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差
                    grad_var = grad_var.to(original_mask.device)
                    # print(f"grad_var device: {grad_var.device}")
                    # print(f"original_mask device: {original_mask.device}")
                    masked_grad_var = grad_var[original_mask > 0]
                    if save:
                        saved_masked_grad_vars[name] = masked_grad_var.cpu().numpy()

                    # 计算被掩码部分的均值
                    masked_mean = masked_grad_var.mean()

                    # 根据方差生成掩码（不使用 for 循环）
                    optimized_mask = (grad_var < masked_mean).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask
        # save_path = "/home/home_node7/wzb/自己算法/miro-main2/result/draw/masked_grad_vars.pkl"
        # if save:
        #     with open(save_path, "wb") as f:
        #         pickle.dump(saved_masked_grad_vars, f)

        #     print(f"Saved masked_grad_var to {save_path}")
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')
    
    def optimize_final_mask_by_quantile(self, data_list, labels_list):
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差

                    # 计算分位数阈值并生成掩码
                    quantile_threshold = self.args['quantile_threshold']  # 需要提前定义的超参数
                    threshold = torch.quantile(grad_var, quantile_threshold)
                    optimized_mask = (grad_var < threshold).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

    
    def optimize_final_mask_by_mean(self, data_list, labels_list):
        """
        通过平均梯度大小进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的梯度
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        # Step 2: 计算梯度平均值并根据大小设置掩码
        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_stack = torch.stack(all_importance[name])  # 堆叠所有域的梯度
                    grad_mean = param_grad_stack.mean(dim=0)  # 计算每个维度的平均值

                    # 打印设备信息，检查 grad_mean 和 original_mask 是否在同一设备上
                    # print(f"grad_mean device: {grad_mean.device}")
                    # print(f"original_mask device: {original_mask.device}")

                    # 将 grad_mean 移动到与其他张量相同的设备（CUDA）上
                    grad_mean = grad_mean.to(original_mask.device)

                    # 计算平均值的排序
                    sorted_grad_values, sorted_indices = torch.sort(grad_mean, dim=0, descending=True)

                    # 获取阈值，前50%设为1，后50%设为0.5
                    threshold = sorted_grad_values[int(len(sorted_grad_values) * 0.5)]

                    # 确保阈值移到正确的设备上
                    threshold = threshold.to(original_mask.device)

                    # 根据平均梯度值对掩码进行更新
                    optimized_mask = torch.where(grad_mean >= threshold, 
                                                torch.tensor(1.0, device=original_mask.device), 
                                                torch.tensor(0.5, device=original_mask.device))

                    # 更新掩码
                    # print(optimized_mask.device)
                    # print(original_mask.device)
                    self.mask[name] = optimized_mask * original_mask

        # 打印更新后的掩码中 1 的数量
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')




    def optimize_final_mask_by_center_direction(self, data_list, labels_list):
        """
        通过梯度方向与中心向量的相似度进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性（通过中心向量）
        all_center_vectors = {}  # 用于存储每层的中心向量

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 计算每层的中心向量
            for name, importance in importance_scores.items():
                if name not in all_center_vectors:
                    all_center_vectors[name] = []
                all_center_vectors[name].append(importance.detach().cpu())

        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的中心向量
                if name in all_center_vectors:
                    center_vectors = torch.stack(all_center_vectors[name])  # 计算得到所有域的梯度中心向量
                    center_vector = center_vectors.mean(dim=0)  # 对所有域的梯度进行平均，得到中心向量

                    # Step 2: 计算每个梯度向量与中心向量的余弦相似度
                    grad_stack = torch.stack([importance.detach().cpu() for importance in all_center_vectors[name]])  # 收集每个域的梯度
                    grad_norm = grad_stack / grad_stack.norm(dim=-1, keepdim=True)  # 对梯度向量进行归一化

                    # 计算所有域的梯度与中心向量的余弦相似度
                    center_vector_norm = center_vector / center_vector.norm()

                    # **广播中心向量**，确保它的形状为 (batch_size, feature_dim)
                    # 如果center_vector是[768]，则扩展为[batch_size, 768]
                    center_vector_norm_expanded = center_vector_norm.unsqueeze(0).expand(grad_norm.size(0), -1)

                    # 计算余弦相似度
                    cosine_similarities = torch.sum(grad_norm * center_vector_norm_expanded, dim=-1)  # 计算余弦相似度

                    # Step 3: 根据相似度筛选维度
                    grad_sim_mask = cosine_similarities.mean(dim=0)  # 对每个维度的相似度取平均
                    grad_sim_mask = (grad_sim_mask >= grad_sim_mask.mean()).float()  # 根据相似度的平均值生成掩码

                    # 更新掩码
                    self.mask[name] = grad_sim_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')



    def train_step(self, data, labels):
        """
        执行一次训练步骤。 

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                # mask = self.mask[name].to('cuda') # param.grad.device
                if param.grad is None:
                    print(f"Parameter {name} has no gradient. requires_grad={param.requires_grad}")
                else:
                    param.grad = param.grad * self.mask[name]
        memory1 = torch.cuda.memory_allocated() / (1024 ** 3)
        
        # 更新网络参数
        
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        return [loss.item(),memory1]

    def predict(self, data):
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y
    
    def analyze_mask(self):
        results = {}
        for name, matrix in self.mask.items():
            if not isinstance(matrix, torch.Tensor):
                raise ValueError(f"The value associated with key '{name}' is not a tensor.")
            
            # 跳过第二个维度不大于 2 的矩阵
            if matrix.dim() < 2 or matrix.size(1) <= 2:
                continue
            
            # 计算矩阵中1的个数
            ones_count = torch.sum(matrix).item()
            
            # 确保矩阵是浮点类型
            matrix_float = matrix.to(dtype=torch.float32)
            
            # 计算矩阵的秩
            _, singular_values, _ = torch.linalg.svd(matrix_float)
            rank = torch.sum(singular_values > 1e-10).item()
            
            # 保存结果
            results[name] = {"ones_count": ones_count, "rank": rank}
            print(name,ones_count,rank)
        
        # return results

    def save_masks_and_importance(self,domain_masks, domain_importance, save_path):
        """
        保存域的掩码和重要性信息到指定路径。
        
        参数:
        - domain_masks: List[Dict[str, Tensor]]，每个域的掩码字典。
        - domain_importance: List[Dict[str, Tensor]]，每个域的梯度/重要性字典。
        - save_path: str，保存文件的路径。
        """
        combined_data = {}

        # 遍历每个域，保存对应的掩码和重要性信息
        for domain_idx, (domain_mask, domain_imp) in enumerate(zip(domain_masks, domain_importance)):
            # 对每个域的掩码和重要性信息进行转换，确保它们在 CPU 上
            cpu_domain_mask = {name: mask.cpu() for name, mask in domain_mask.items()}
            cpu_domain_importance = {name: imp.cpu() for name, imp in domain_imp.items()}
            combined_data[f'domain_{domain_idx}'] = {
                'masks': cpu_domain_mask,
                'importance': cpu_domain_importance
            }

        # 保存所有域的数据为一个文件
        torch.save(combined_data, save_path)
        print(f"Masks and importance have been saved to '{save_path}'.")









class SPU_step_du:  # 匹配的是一个列表不再是一个值
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.selected_layers = args['selected_layers'].split()
        self.way = args['way']
        self.stored_masked_grads = {}  # 存储历史mask=0处的梯度
        self.storage_interval = 50      # 存储间隔步数
        self.step_counter = 0      

    def compute_score(self, data_list, labels_list, name=None):
        # 初始化 importance 字典
        importance = {name: torch.zeros_like(param, device='cpu') 
                    for name, param in self.network.named_parameters() 
                    if any(layer_name in name for layer_name in self.selected_layers)}

        # 设置网络为训练模式
        self.network.train()
        self.network.zero_grad()

        # 遍历每个数据批次
        for data, labels in zip(data_list, labels_list):
            # 前向传播
            logits_per_image = self.network(data)  # 获取当前批次的输出
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度

            # 累加梯度
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.detach().cpu().clone()  # 累加梯度

            # 清除梯度，避免累加
            self.network.zero_grad()

        return importance

    def compute_importance(self, data_list, labels_list, name=None, save = False ):
        print('Compute importance for the current task...')
        save_impor = False
        domain_masks = []
        if save_impor:
            domain_importance = []
        # print(len(data_list))
        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance = self.compute_score(current_data, current_labels)
            if save_impor:
                domain_importance.append(importance)

            mask = {}
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
                    mask[name] = mask_tensor
            domain_masks.append(mask)
        if save_impor:
            save_dir = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_r_pacs_vit.pth'
            self.save_masks_and_importance(domain_masks,domain_importance,save_dir)
        
        final_mask = {}
        # overlap_ratios = {}  # 存储每一层的重叠率

        for name, param in self.network.named_parameters():
            # 判断是否属于指定层
            if any(layer_name in name for layer_name in self.selected_layers):
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])
                
                # 依次与其他domain_masks相乘
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])
                
                # 保存到final_mask
                final_mask[name] = layer_mask

                # # 计算剩余部分占比原mask的比值，实际上就是重叠率
                # original_count = torch.sum(domain_masks[0][name])  # 全1矩阵的总和
                # final_mask_count = torch.sum(layer_mask == 1)       # final_mask中1的总数
                # overlap_ratios[name] = (final_mask_count / original_count).item()  # 计算比率并保存

        # 打印每一层的重叠率,等要用的时候再打开
        # for layer_name, overlap in overlap_ratios.items():
        #     print(f"Layer: {layer_name}, Overlap Ratio: {overlap:.4f}")

        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                if final_mask[name].sum() == 0:
                    print(f"Layer {name} is excluded from training (mask is all zeros).")
                    del final_mask[name]
                else:
                    param.requires_grad = True
                    selected_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.mask = final_mask
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

        # 第二步：基于梯度方差优化最终掩码
        if self.way == 'var':
            self.optimize_final_mask_by_variance(data_list, labels_list)
        elif self.way == 'dir':
            self.optimize_final_mask_by_center_direction(data_list, labels_list)
        elif self.way == 'mean':
            self.optimize_final_mask_by_mean(data_list, labels_list)
        elif self.way == 'qua':
            self.optimize_final_mask_by_quantile(data_list, labels_list)
        else:
            print('Select withou step2.')

    def optimize_final_mask_by_variance(self, data_list, labels_list,save = False):
        save = True
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
                # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')
        if save:
            saved_masked_grad_vars = {}

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                 
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差
                    grad_var = grad_var.to(original_mask.device)
                    # print(f"grad_var device: {grad_var.device}")
                    # print(f"original_mask device: {original_mask.device}")
                    masked_grad_var = grad_var[original_mask > 0]
                    if save:
                        saved_masked_grad_vars[name] = masked_grad_var.cpu().numpy()

                    # 计算被掩码部分的均值
                    masked_mean = masked_grad_var.mean()

                    # 根据方差生成掩码（不使用 for 循环）
                    optimized_mask = (grad_var < masked_mean).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask
        # save_path = "/home/home_node7/wzb/自己算法/miro-main2/result/draw/masked_grad_vars.pkl"
        # if save:
        #     with open(save_path, "wb") as f:
        #         pickle.dump(saved_masked_grad_vars, f)

        #     print(f"Saved masked_grad_var to {save_path}")
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')
    
    def optimize_final_mask_by_quantile(self, data_list, labels_list):
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差

                    # 计算分位数阈值并生成掩码
                    quantile_threshold = self.args['quantile_threshold']  # 需要提前定义的超参数
                    threshold = torch.quantile(grad_var, quantile_threshold)
                    optimized_mask = (grad_var < threshold).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

    
    def optimize_final_mask_by_mean(self, data_list, labels_list):
        """
        通过平均梯度大小进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的梯度
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        # Step 2: 计算梯度平均值并根据大小设置掩码
        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_stack = torch.stack(all_importance[name])  # 堆叠所有域的梯度
                    grad_mean = param_grad_stack.mean(dim=0)  # 计算每个维度的平均值

                    # 打印设备信息，检查 grad_mean 和 original_mask 是否在同一设备上
                    # print(f"grad_mean device: {grad_mean.device}")
                    # print(f"original_mask device: {original_mask.device}")

                    # 将 grad_mean 移动到与其他张量相同的设备（CUDA）上
                    grad_mean = grad_mean.to(original_mask.device)

                    # 计算平均值的排序
                    sorted_grad_values, sorted_indices = torch.sort(grad_mean, dim=0, descending=True)

                    # 获取阈值，前50%设为1，后50%设为0.5
                    threshold = sorted_grad_values[int(len(sorted_grad_values) * 0.5)]

                    # 确保阈值移到正确的设备上
                    threshold = threshold.to(original_mask.device)

                    # 根据平均梯度值对掩码进行更新
                    optimized_mask = torch.where(grad_mean >= threshold, 
                                                torch.tensor(1.0, device=original_mask.device), 
                                                torch.tensor(0.5, device=original_mask.device))

                    # 更新掩码
                    # print(optimized_mask.device)
                    # print(original_mask.device)
                    self.mask[name] = optimized_mask * original_mask

        # 打印更新后的掩码中 1 的数量
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')




    def optimize_final_mask_by_center_direction(self, data_list, labels_list):
        """
        通过梯度方向与中心向量的相似度进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性（通过中心向量）
        all_center_vectors = {}  # 用于存储每层的中心向量

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 计算每层的中心向量
            for name, importance in importance_scores.items():
                if name not in all_center_vectors:
                    all_center_vectors[name] = []
                all_center_vectors[name].append(importance.detach().cpu())

        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的中心向量
                if name in all_center_vectors:
                    center_vectors = torch.stack(all_center_vectors[name])  # 计算得到所有域的梯度中心向量
                    center_vector = center_vectors.mean(dim=0)  # 对所有域的梯度进行平均，得到中心向量

                    # Step 2: 计算每个梯度向量与中心向量的余弦相似度
                    grad_stack = torch.stack([importance.detach().cpu() for importance in all_center_vectors[name]])  # 收集每个域的梯度
                    grad_norm = grad_stack / grad_stack.norm(dim=-1, keepdim=True)  # 对梯度向量进行归一化

                    # 计算所有域的梯度与中心向量的余弦相似度
                    center_vector_norm = center_vector / center_vector.norm()

                    # **广播中心向量**，确保它的形状为 (batch_size, feature_dim)
                    # 如果center_vector是[768]，则扩展为[batch_size, 768]
                    center_vector_norm_expanded = center_vector_norm.unsqueeze(0).expand(grad_norm.size(0), -1)

                    # 计算余弦相似度
                    cosine_similarities = torch.sum(grad_norm * center_vector_norm_expanded, dim=-1)  # 计算余弦相似度

                    # Step 3: 根据相似度筛选维度
                    grad_sim_mask = cosine_similarities.mean(dim=0)  # 对每个维度的相似度取平均
                    grad_sim_mask = (grad_sim_mask >= grad_sim_mask.mean()).float()  # 根据相似度的平均值生成掩码

                    # 更新掩码
                    self.mask[name] = grad_sim_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')



    def train_step(self, data, labels):
        """
        执行一次训练步骤。 

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                if param.grad is None:
                    print(f"参数 {name} 无梯度. requires_grad={param.requires_grad}")
                else:
                    # ============= 前面所有原有代码保持不变 =============
                    inverted_mask = 1 - self.mask[name]
                    current_masked_grad = param.grad * inverted_mask
                    valid_pixels = (inverted_mask > 0).sum().item()
                    
                    # 原始统计量计算
                    orig_mean = orig_max = 0.0
                    if valid_pixels > 0:
                        orig_mean = current_masked_grad.sum().item() / valid_pixels
                        orig_max = current_masked_grad.abs().max().item()
                    
                    # 差值统计量计算
                    delta_mean = delta_max = 0.0
                    if name in self.stored_masked_grads and valid_pixels > 0:
                        grad_delta = current_masked_grad - self.stored_masked_grads[name]
                        delta_mean = grad_delta.sum().item() / valid_pixels
                        delta_max = grad_delta.abs().max().item()
                    
                    # 定期存储
                    self.step_counter += 1
                    if self.step_counter % self.storage_interval == 0:
                        self.stored_masked_grads[name] = current_masked_grad.detach().clone()
                    
                    # 原有mask应用
                    param.grad = param.grad * self.mask[name]
                # ============= 前面所有代码到此结束 =============
                
                # 只修改这里：返回Python列表而不是Tensor
                
        
        # 更新网络参数
        
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        # mean_masked_grad = float(mean_masked_grad.cpu().numpy())
        # max_abs_masked_grad = float(max_abs_masked_grad.cpu().numpy())
        return [orig_mean, orig_max, delta_mean, delta_max]
    
    
    def predict(self, data):
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y

class SPU_step_du2:  # 匹配的是一个列表不再是一个值
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.selected_layers = args['selected_layers'].split()
        self.way = args['way']
        self.stored_masked_grads = {}  # 存储历史mask=0处的梯度
        self.storage_interval = 50      # 存储间隔步数
        self.step_counter = 0    
        self.stored_other_grads = {}  

    def compute_score(self, data_list, labels_list, name=None):
        # 初始化 importance 字典
        importance = {name: torch.zeros_like(param, device='cpu') 
                    for name, param in self.network.named_parameters() 
                    if any(layer_name in name for layer_name in self.selected_layers)}

        # 设置网络为训练模式
        self.network.train()
        self.network.zero_grad()

        # 遍历每个数据批次
        for data, labels in zip(data_list, labels_list):
            # 前向传播
            logits_per_image = self.network(data)  # 获取当前批次的输出
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度

            # 累加梯度
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.detach().cpu().clone()  # 累加梯度

            # 清除梯度，避免累加
            self.network.zero_grad()

        return importance

    def compute_importance(self, data_list, labels_list, name=None, save = False ):
        print('Compute importance for the current task...')
        save_impor = False
        domain_masks = []
        if save_impor:
            domain_importance = []
        # print(len(data_list))
        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance = self.compute_score(current_data, current_labels)
            if save_impor:
                domain_importance.append(importance)

            mask = {}
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
                    mask[name] = mask_tensor
            domain_masks.append(mask)
        if save_impor:
            save_dir = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_r_pacs_vit.pth'
            self.save_masks_and_importance(domain_masks,domain_importance,save_dir)
        
        final_mask = {}
        # overlap_ratios = {}  # 存储每一层的重叠率

        for name, param in self.network.named_parameters():
            # 判断是否属于指定层
            if any(layer_name in name for layer_name in self.selected_layers):
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])
                
                # 依次与其他domain_masks相乘
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])
                
                # 保存到final_mask
                final_mask[name] = layer_mask

                # # 计算剩余部分占比原mask的比值，实际上就是重叠率
                # original_count = torch.sum(domain_masks[0][name])  # 全1矩阵的总和
                # final_mask_count = torch.sum(layer_mask == 1)       # final_mask中1的总数
                # overlap_ratios[name] = (final_mask_count / original_count).item()  # 计算比率并保存

        # 打印每一层的重叠率,等要用的时候再打开
        # for layer_name, overlap in overlap_ratios.items():
        #     print(f"Layer: {layer_name}, Overlap Ratio: {overlap:.4f}")

        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                if final_mask[name].sum() == 0:
                    print(f"Layer {name} is excluded from training (mask is all zeros).")
                    del final_mask[name]
                else:
                    param.requires_grad = True
                    selected_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.mask = final_mask
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

        # 第二步：基于梯度方差优化最终掩码
        if self.way == 'var':
            self.optimize_final_mask_by_variance(data_list, labels_list)
        elif self.way == 'dir':
            self.optimize_final_mask_by_center_direction(data_list, labels_list)
        elif self.way == 'mean':
            self.optimize_final_mask_by_mean(data_list, labels_list)
        elif self.way == 'qua':
            self.optimize_final_mask_by_quantile(data_list, labels_list)
        else:
            print('Select withou step2.')

    def optimize_final_mask_by_variance(self, data_list, labels_list,save = False):
        save = True
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
                # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')
        if save:
            saved_masked_grad_vars = {}

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                 
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差
                    grad_var = grad_var.to(original_mask.device)
                    # print(f"grad_var device: {grad_var.device}")
                    # print(f"original_mask device: {original_mask.device}")
                    masked_grad_var = grad_var[original_mask > 0]
                    if save:
                        saved_masked_grad_vars[name] = masked_grad_var.cpu().numpy()

                    # 计算被掩码部分的均值
                    masked_mean = masked_grad_var.mean()

                    # 根据方差生成掩码（不使用 for 循环）
                    optimized_mask = (grad_var < masked_mean).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask
        # save_path = "/home/home_node7/wzb/自己算法/miro-main2/result/draw/masked_grad_vars.pkl"
        # if save:
        #     with open(save_path, "wb") as f:
        #         pickle.dump(saved_masked_grad_vars, f)

        #     print(f"Saved masked_grad_var to {save_path}")
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')
    
    def optimize_final_mask_by_quantile(self, data_list, labels_list):
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差

                    # 计算分位数阈值并生成掩码
                    quantile_threshold = self.args['quantile_threshold']  # 需要提前定义的超参数
                    threshold = torch.quantile(grad_var, quantile_threshold)
                    optimized_mask = (grad_var < threshold).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

    
    def optimize_final_mask_by_mean(self, data_list, labels_list):
        """
        通过平均梯度大小进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的梯度
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        # Step 2: 计算梯度平均值并根据大小设置掩码
        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_stack = torch.stack(all_importance[name])  # 堆叠所有域的梯度
                    grad_mean = param_grad_stack.mean(dim=0)  # 计算每个维度的平均值

                    # 打印设备信息，检查 grad_mean 和 original_mask 是否在同一设备上
                    # print(f"grad_mean device: {grad_mean.device}")
                    # print(f"original_mask device: {original_mask.device}")

                    # 将 grad_mean 移动到与其他张量相同的设备（CUDA）上
                    grad_mean = grad_mean.to(original_mask.device)

                    # 计算平均值的排序
                    sorted_grad_values, sorted_indices = torch.sort(grad_mean, dim=0, descending=True)

                    # 获取阈值，前50%设为1，后50%设为0.5
                    threshold = sorted_grad_values[int(len(sorted_grad_values) * 0.5)]

                    # 确保阈值移到正确的设备上
                    threshold = threshold.to(original_mask.device)

                    # 根据平均梯度值对掩码进行更新
                    optimized_mask = torch.where(grad_mean >= threshold, 
                                                torch.tensor(1.0, device=original_mask.device), 
                                                torch.tensor(0.5, device=original_mask.device))

                    # 更新掩码
                    # print(optimized_mask.device)
                    # print(original_mask.device)
                    self.mask[name] = optimized_mask * original_mask

        # 打印更新后的掩码中 1 的数量
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')




    def optimize_final_mask_by_center_direction(self, data_list, labels_list):
        """
        通过梯度方向与中心向量的相似度进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性（通过中心向量）
        all_center_vectors = {}  # 用于存储每层的中心向量

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 计算每层的中心向量
            for name, importance in importance_scores.items():
                if name not in all_center_vectors:
                    all_center_vectors[name] = []
                all_center_vectors[name].append(importance.detach().cpu())

        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的中心向量
                if name in all_center_vectors:
                    center_vectors = torch.stack(all_center_vectors[name])  # 计算得到所有域的梯度中心向量
                    center_vector = center_vectors.mean(dim=0)  # 对所有域的梯度进行平均，得到中心向量

                    # Step 2: 计算每个梯度向量与中心向量的余弦相似度
                    grad_stack = torch.stack([importance.detach().cpu() for importance in all_center_vectors[name]])  # 收集每个域的梯度
                    grad_norm = grad_stack / grad_stack.norm(dim=-1, keepdim=True)  # 对梯度向量进行归一化

                    # 计算所有域的梯度与中心向量的余弦相似度
                    center_vector_norm = center_vector / center_vector.norm()

                    # **广播中心向量**，确保它的形状为 (batch_size, feature_dim)
                    # 如果center_vector是[768]，则扩展为[batch_size, 768]
                    center_vector_norm_expanded = center_vector_norm.unsqueeze(0).expand(grad_norm.size(0), -1)

                    # 计算余弦相似度
                    cosine_similarities = torch.sum(grad_norm * center_vector_norm_expanded, dim=-1)  # 计算余弦相似度

                    # Step 3: 根据相似度筛选维度
                    grad_sim_mask = cosine_similarities.mean(dim=0)  # 对每个维度的相似度取平均
                    grad_sim_mask = (grad_sim_mask >= grad_sim_mask.mean()).float()  # 根据相似度的平均值生成掩码

                    # 更新掩码
                    self.mask[name] = grad_sim_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')



    def train_step(self, data, labels):
        """
        执行一次训练步骤。 

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            s1 = [ float("-inf")]
            s2 = [ float("-inf")]
            s3 = [ float("-inf")]
            s4 = [ float("-inf")]
            if param.grad is None:
                # print(f"参数 {name} 无梯度. requires_grad={param.requires_grad}")
                continue
                
            if name in self.mask:
                # ============= 原有mask内参数的计算保持不变 =============
                inverted_mask = 1 - self.mask[name]
                current_masked_grad = param.grad * inverted_mask
                valid_pixels = (inverted_mask > 0).sum().item()
                
                # 原始统计量计算
                orig_mean = orig_max = 0.0
                if valid_pixels > 0:
                    orig_mean = current_masked_grad.sum().item() / valid_pixels
                    orig_max = current_masked_grad.abs().max().item()
                
                # 差值统计量计算
                delta_mean = delta_max = 0.0
                if name in self.stored_masked_grads and valid_pixels > 0:
                    grad_delta = current_masked_grad - self.stored_masked_grads[name]
                    delta_mean = grad_delta.sum().item() / valid_pixels
                    delta_max = grad_delta.abs().max().item()
                   
                
                # 定期存储
                self.step_counter += 1
                if self.step_counter % self.storage_interval == 0:
                    self.stored_masked_grads[name] = current_masked_grad.detach().clone()
                
                # 原有mask应用
                param.grad = param.grad * self.mask[name]
            else:
                # ============= 新增：不在mask中的参数统计 =============
                current_grad = param.grad
                valid_pixels = current_grad.numel()
                
                # 原始统计量计算
                other_orig_mean = current_grad.sum().item() / valid_pixels
                other_orig_max = current_grad.abs().max().item()
                if name not in self.stored_other_grads:
                    self.stored_other_grads[name] = torch.zeros_like(current_grad)
                grad_delta = current_grad - self.stored_other_grads[name]
                
                # 差值统计量计算
                other_delta_mean = other_delta_max = 0.0
                s1.append(other_orig_mean)
                s2.append(other_orig_max)

                # grad_delta = current_grad - self.stored_other_grads[name]
                other_delta_mean = grad_delta.sum().item() / valid_pixels
                other_delta_max = grad_delta.abs().max().item()
                s3.append(other_delta_mean)
                s4.append(other_delta_max)
                
                # 定期存储
                if self.step_counter % self.storage_interval == 0:
                    self.stored_other_grads[name] = current_grad.detach().clone()
                        # ============= 前面所有代码到此结束 =============
                
                # 只修改这里：返回Python列表而不是Tensor
                
        
        # 更新网络参数
        
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        # mean_masked_grad = float(mean_masked_grad.cpu().numpy())
        # max_abs_masked_grad = float(max_abs_masked_grad.cpu().numpy())
        return [max(s1), max(s2), max(s3), max(s4)]

    def predict(self, data):
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y
    
    def analyze_mask(self):
        results = {}
        for name, matrix in self.mask.items():
            if not isinstance(matrix, torch.Tensor):
                raise ValueError(f"The value associated with key '{name}' is not a tensor.")
            
            # 跳过第二个维度不大于 2 的矩阵
            if matrix.dim() < 2 or matrix.size(1) <= 2:
                continue
            
            # 计算矩阵中1的个数
            ones_count = torch.sum(matrix).item()
            
            # 确保矩阵是浮点类型
            matrix_float = matrix.to(dtype=torch.float32)
            
            # 计算矩阵的秩
            _, singular_values, _ = torch.linalg.svd(matrix_float)
            rank = torch.sum(singular_values > 1e-10).item()
            
            # 保存结果
            results[name] = {"ones_count": ones_count, "rank": rank}
            print(name,ones_count,rank)
        
        # return results

    def save_masks_and_importance(self,domain_masks, domain_importance, save_path):
        """
        保存域的掩码和重要性信息到指定路径。
        
        参数:
        - domain_masks: List[Dict[str, Tensor]]，每个域的掩码字典。
        - domain_importance: List[Dict[str, Tensor]]，每个域的梯度/重要性字典。
        - save_path: str，保存文件的路径。
        """
        combined_data = {}

        # 遍历每个域，保存对应的掩码和重要性信息
        for domain_idx, (domain_mask, domain_imp) in enumerate(zip(domain_masks, domain_importance)):
            # 对每个域的掩码和重要性信息进行转换，确保它们在 CPU 上
            cpu_domain_mask = {name: mask.cpu() for name, mask in domain_mask.items()}
            cpu_domain_importance = {name: imp.cpu() for name, imp in domain_imp.items()}
            combined_data[f'domain_{domain_idx}'] = {
                'masks': cpu_domain_mask,
                'importance': cpu_domain_importance
            }

        # 保存所有域的数据为一个文件
        torch.save(combined_data, save_path)
        print(f"Masks and importance have been saved to '{save_path}'.")






class SPU_stepg:  # 增加高斯分布噪声
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.selected_layers = args['selected_layers'].split()
        self.way = args['way']

    # def compute_score(self, data, labels, name=None):
    #     importance = {name: torch.zeros_like(param, device='cpu') 
    #                  for name, param in self.network.named_parameters() 
    #                  if any(layer_name in name for layer_name in self.selected_layers)}
    #     self.network.train()
    #     self.network.zero_grad()

    #     # 前向传播
    #     # print(len(data))
    #     logits_per_image = self.network(data)
    #     loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
    #     loss.backward()

    #     for name, param in self.network.named_parameters():
    #         if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
    #             importance[name] += param.grad.detach().cpu().clone()

    #     self.network.zero_grad()  # 清除梯度，避免累加

    #     return importance
    def compute_score(self, data_list, labels_list, gaussian_num=5, variance=0.01):
        """
        计算选定层的重要性分数，支持高斯噪声数据增强，包含原始图像梯度。

        Args:
            data_list (list[torch.Tensor]): 输入数据的批次列表。
            labels_list (list[torch.Tensor]): 对应的标签批次列表。
            gaussian_num (int): 每张图像生成的高斯增强版本数量，默认为 5。
            variance (float): 高斯噪声的方差，默认为 0.01。

        Returns:
            dict: 各层参数的重要性分数。
        """
        # 初始化 importance 字典
        importance = {
            name: torch.zeros_like(param, device='cpu') 
            for name, param in self.network.named_parameters() 
            if any(layer_name in name for layer_name in self.selected_layers)
        }

        # 设置网络为训练模式
        self.network.train()
        self.network.zero_grad()

        # 遍历每个数据批次
        for data, labels in zip(data_list, labels_list):
            # 初始化批次梯度累加器
            batch_gradients = {
                name: torch.zeros_like(param, device='cpu') 
                for name, param in self.network.named_parameters() 
                if any(layer_name in name for layer_name in self.selected_layers)
            }

            # 包括原始图像的增强数据版本
            all_augmented_data = [data]  # 原始图像
            for _ in range(gaussian_num):
                noise = torch.normal(mean=0.0, std=torch.sqrt(torch.tensor(variance)), size=data.shape).to(data.device)
                all_augmented_data.append(data + noise)

            # 遍历所有增强版本并计算梯度
            for augmented_data in all_augmented_data:
                logits_per_image = self.network(augmented_data)
                loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
                loss.backward()  # 反向传播计算梯度

                for name, param in self.network.named_parameters():
                    if (
                        any(layer_name in name for layer_name in self.selected_layers) 
                        and param.requires_grad 
                        and param.grad is not None
                    ):
                        batch_gradients[name] += param.grad.detach().cpu().clone()

                self.network.zero_grad()

            # 对增强版本的梯度取均值并更新 importance
            for name in batch_gradients.keys():
                batch_gradients[name] /= (gaussian_num + 1)  # 包含原始图像
                importance[name] += batch_gradients[name]

        return importance

    def compute_importance(self, data_list, labels_list, name=None):
        print('Compute importance for the current task...')
        
        domain_masks = []
        # print(len(data_list))
        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance = self.compute_score(current_data, current_labels)

            mask = {}
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
                    mask[name] = mask_tensor
            domain_masks.append(mask)
        
        final_mask = {}
        for name, param in self.network.named_parameters():
            # print(name)
            if any(layer_name in name for layer_name in self.selected_layers):
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])
                final_mask[name] = layer_mask

        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                if final_mask[name].sum() == 0:
                    print(f"Layer {name} is excluded from training (mask is all zeros).")
                    del final_mask[name]
                else:
                    param.requires_grad = True
                    selected_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.mask = final_mask
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

        # 第二步：基于梯度方差优化最终掩码
        if self.way == 'var':
            self.optimize_final_mask_by_variance(data_list, labels_list)
        elif self.way == 'dir':
            self.optimize_final_mask_by_center_direction(data_list, labels_list)
        elif self.way == 'mean':
            self.optimize_final_mask_by_mean(data_list, labels_list)
        elif self.way == 'qua':
            self.optimize_final_mask_by_quantile(data_list, labels_list)

    def optimize_final_mask_by_variance(self, data_list, labels_list):
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
                # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                 
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差
                    grad_var = grad_var.to(original_mask.device)
                    # print(f"grad_var device: {grad_var.device}")
                    # print(f"original_mask device: {original_mask.device}")
                    masked_grad_var = grad_var[original_mask > 0]

                    # 计算被掩码部分的均值
                    masked_mean = masked_grad_var.mean()

                    # 根据方差生成掩码（不使用 for 循环）
                    optimized_mask = (grad_var < masked_mean).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')
    
    def optimize_final_mask_by_quantile(self, data_list, labels_list):
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差

                    # 计算分位数阈值并生成掩码
                    quantile_threshold = self.args['quantile_threshold']  # 需要提前定义的超参数
                    threshold = torch.quantile(grad_var, quantile_threshold)
                    optimized_mask = (grad_var < threshold).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

    
    def optimize_final_mask_by_mean(self, data_list, labels_list):
        """
        通过平均梯度大小进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的梯度
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        # Step 2: 计算梯度平均值并根据大小设置掩码
        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_stack = torch.stack(all_importance[name])  # 堆叠所有域的梯度
                    grad_mean = param_grad_stack.mean(dim=0)  # 计算每个维度的平均值

                    # 打印设备信息，检查 grad_mean 和 original_mask 是否在同一设备上
                    # print(f"grad_mean device: {grad_mean.device}")
                    # print(f"original_mask device: {original_mask.device}")

                    # 将 grad_mean 移动到与其他张量相同的设备（CUDA）上
                    grad_mean = grad_mean.to(original_mask.device)

                    # 计算平均值的排序
                    sorted_grad_values, sorted_indices = torch.sort(grad_mean, dim=0, descending=True)

                    # 获取阈值，前50%设为1，后50%设为0.5
                    threshold = sorted_grad_values[int(len(sorted_grad_values) * 0.5)]

                    # 确保阈值移到正确的设备上
                    threshold = threshold.to(original_mask.device)

                    # 根据平均梯度值对掩码进行更新
                    optimized_mask = torch.where(grad_mean >= threshold, 
                                                torch.tensor(1.0, device=original_mask.device), 
                                                torch.tensor(0.5, device=original_mask.device))

                    # 更新掩码
                    # print(optimized_mask.device)
                    # print(original_mask.device)
                    self.mask[name] = optimized_mask * original_mask

        # 打印更新后的掩码中 1 的数量
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')




    def optimize_final_mask_by_center_direction(self, data_list, labels_list):
        """
        通过梯度方向与中心向量的相似度进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性（通过中心向量）
        all_center_vectors = {}  # 用于存储每层的中心向量

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 计算每层的中心向量
            for name, importance in importance_scores.items():
                if name not in all_center_vectors:
                    all_center_vectors[name] = []
                all_center_vectors[name].append(importance.detach().cpu())

        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的中心向量
                if name in all_center_vectors:
                    center_vectors = torch.stack(all_center_vectors[name])  # 计算得到所有域的梯度中心向量
                    center_vector = center_vectors.mean(dim=0)  # 对所有域的梯度进行平均，得到中心向量

                    # Step 2: 计算每个梯度向量与中心向量的余弦相似度
                    grad_stack = torch.stack([importance.detach().cpu() for importance in all_center_vectors[name]])  # 收集每个域的梯度
                    grad_norm = grad_stack / grad_stack.norm(dim=-1, keepdim=True)  # 对梯度向量进行归一化

                    # 计算所有域的梯度与中心向量的余弦相似度
                    center_vector_norm = center_vector / center_vector.norm()

                    # **广播中心向量**，确保它的形状为 (batch_size, feature_dim)
                    # 如果center_vector是[768]，则扩展为[batch_size, 768]
                    center_vector_norm_expanded = center_vector_norm.unsqueeze(0).expand(grad_norm.size(0), -1)

                    # 计算余弦相似度
                    cosine_similarities = torch.sum(grad_norm * center_vector_norm_expanded, dim=-1)  # 计算余弦相似度

                    # Step 3: 根据相似度筛选维度
                    grad_sim_mask = cosine_similarities.mean(dim=0)  # 对每个维度的相似度取平均
                    grad_sim_mask = (grad_sim_mask >= grad_sim_mask.mean()).float()  # 根据相似度的平均值生成掩码

                    # 更新掩码
                    self.mask[name] = grad_sim_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')



    def train_step(self, data, labels):
        """
        执行一次训练步骤。 

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to('cuda') # param.grad.device
                if param.grad is None:
                    print(f"Parameter {name} has no gradient. requires_grad={param.requires_grad}")
                else:
                    param.grad = param.grad * mask
        
        # 更新网络参数
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        return loss.item()

    def predict(self, data):
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y







class SPU_step2:  # 匹配的是一个列表不再是一个值
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.selected_layers = args['selected_layers'].split()
        self.way = args['way']

    # def compute_score(self, data, labels, name=None):
    #     importance = {name: torch.zeros_like(param, device='cpu') 
    #                  for name, param in self.network.named_parameters() 
    #                  if any(layer_name in name for layer_name in self.selected_layers)}
    #     self.network.train()
    #     self.network.zero_grad()

    #     # 前向传播
    #     # print(len(data))
    #     logits_per_image = self.network(data)
    #     loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
    #     loss.backward()

    #     for name, param in self.network.named_parameters():
    #         if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
    #             importance[name] += param.grad.detach().cpu().clone()

    #     self.network.zero_grad()  # 清除梯度，避免累加

    #     return importance
    def compute_score(self, data_list, labels_list, name=None):
        # 初始化 importance 字典
        importance = {name: torch.zeros_like(param, device='cpu') 
                    for name, param in self.network.named_parameters() 
                    if any(layer_name in name for layer_name in self.selected_layers)}

        # 设置网络为训练模式
        self.network.train()
        self.network.zero_grad()

        # 遍历每个数据批次
        for data, labels in zip(data_list, labels_list):
            # 前向传播
            logits_per_image = self.network(data)  # 获取当前批次的输出
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度

            # 累加梯度
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.detach().cpu().clone()  # 累加梯度

            # 清除梯度，避免累加
            self.network.zero_grad()

        return importance

    def compute_importance(self, data_list, labels_list, name=None):
        print('Compute importance for the current task...')

        domain_masks = []
        self.layer_mean_values = {}  # 用于存储每个层的均值

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance = self.compute_score(current_data, current_labels)

            mask = {}
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and name in importance:
                    magnitudes = importance[name].abs()
                    k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                    topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                    
                    # 获取 topk_values 中最小的那个值
                    min_topk_value = topk_values.min().item()  # 取最小的一个值
                    self.layer_mean_values[name] = self.layer_mean_values.get(name, 0) + min_topk_value

                    mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                    mask_tensor.view(-1)[topk_indices] = 1
                    mask[name] = mask_tensor

            domain_masks.append(mask)
        
        final_mask = {}
        for name, param in self.network.named_parameters():
            if any(layer_name in name for layer_name in self.selected_layers):
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])
                final_mask[name] = layer_mask

        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                if final_mask[name].sum() == 0:
                    print(f"Layer {name} is excluded from training (mask is all zeros).")
                    del final_mask[name]
                else:
                    param.requires_grad = True
                    selected_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 计算所有层均值的最终值（可选）
        for name, mean_value in self.layer_mean_values.items():
            # 如果有多个domain，取平均
            # num_domains = len(data_list)
            self.layer_mean_values[name] = mean_value / self.args['k']
            # print(f"Layer {name} mean topk value (min of top k): {self.layer_mean_values[name]}")
        self.mask = final_mask
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

            # 第二步：基于梯度方差优化最终掩码
        if self.way == 'var':
                self.optimize_final_mask_by_variance(data_list, labels_list)
        elif self.way == 'dir':
                self.optimize_final_mask_by_center_direction(data_list, labels_list)
        elif self.way == 'mean':
                self.optimize_final_mask_by_mean(data_list, labels_list)
        elif self.way == 'qua':
                self.optimize_final_mask_by_quantile(data_list, labels_list)

    def optimize_final_mask_by_variance(self, data_list, labels_list):
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
                # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                 
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差
                    grad_var = grad_var.to(original_mask.device)
                    # print(f"grad_var device: {grad_var.device}")
                    # print(f"original_mask device: {original_mask.device}")
                    masked_grad_var = grad_var[original_mask > 0]

                    # 计算被掩码部分的均值
                    masked_mean = masked_grad_var.mean()

                    # 根据方差生成掩码（不使用 for 循环）
                    optimized_mask = (grad_var < masked_mean).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')
    
    def optimize_final_mask_by_quantile(self, data_list, labels_list):
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差

                    # 计算分位数阈值并生成掩码
                    quantile_threshold = self.args['quantile_threshold']  # 需要提前定义的超参数
                    threshold = torch.quantile(grad_var, quantile_threshold)
                    optimized_mask = (grad_var < threshold).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

    
    def optimize_final_mask_by_mean(self, data_list, labels_list):
        """
        通过平均梯度大小进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的梯度
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')

        # Step 2: 计算梯度平均值并根据大小设置掩码
        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_stack = torch.stack(all_importance[name])  # 堆叠所有域的梯度
                    grad_mean = param_grad_stack.mean(dim=0)  # 计算每个维度的平均值

                    # 打印设备信息，检查 grad_mean 和 original_mask 是否在同一设备上
                    # print(f"grad_mean device: {grad_mean.device}")
                    # print(f"original_mask device: {original_mask.device}")

                    # 将 grad_mean 移动到与其他张量相同的设备（CUDA）上
                    grad_mean = grad_mean.to(original_mask.device)

                    # 计算平均值的排序
                    sorted_grad_values, sorted_indices = torch.sort(grad_mean, dim=0, descending=True)

                    # 获取阈值，前50%设为1，后50%设为0.5
                    threshold = sorted_grad_values[int(len(sorted_grad_values) * 0.5)]

                    # 确保阈值移到正确的设备上
                    threshold = threshold.to(original_mask.device)

                    # 根据平均梯度值对掩码进行更新
                    optimized_mask = torch.where(grad_mean >= threshold, 
                                                torch.tensor(1.0, device=original_mask.device), 
                                                torch.tensor(0.5, device=original_mask.device))

                    # 更新掩码
                    # print(optimized_mask.device)
                    # print(original_mask.device)
                    self.mask[name] = optimized_mask * original_mask

        # 打印更新后的掩码中 1 的数量
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')




    def optimize_final_mask_by_center_direction(self, data_list, labels_list):
        """
        通过梯度方向与中心向量的相似度进一步优化最终掩码。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
        # Step 1: 预先计算每一层在每个域上的重要性（通过中心向量）
        all_center_vectors = {}  # 用于存储每层的中心向量

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 计算每层的中心向量
            for name, importance in importance_scores.items():
                if name not in all_center_vectors:
                    all_center_vectors[name] = []
                all_center_vectors[name].append(importance.detach().cpu())

        print('finish compute score')

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的中心向量
                if name in all_center_vectors:
                    center_vectors = torch.stack(all_center_vectors[name])  # 计算得到所有域的梯度中心向量
                    center_vector = center_vectors.mean(dim=0)  # 对所有域的梯度进行平均，得到中心向量

                    # Step 2: 计算每个梯度向量与中心向量的余弦相似度
                    grad_stack = torch.stack([importance.detach().cpu() for importance in all_center_vectors[name]])  # 收集每个域的梯度
                    grad_norm = grad_stack / grad_stack.norm(dim=-1, keepdim=True)  # 对梯度向量进行归一化

                    # 计算所有域的梯度与中心向量的余弦相似度
                    center_vector_norm = center_vector / center_vector.norm()

                    # **广播中心向量**，确保它的形状为 (batch_size, feature_dim)
                    # 如果center_vector是[768]，则扩展为[batch_size, 768]
                    center_vector_norm_expanded = center_vector_norm.unsqueeze(0).expand(grad_norm.size(0), -1)

                    # 计算余弦相似度
                    cosine_similarities = torch.sum(grad_norm * center_vector_norm_expanded, dim=-1)  # 计算余弦相似度

                    # Step 3: 根据相似度筛选维度
                    grad_sim_mask = cosine_similarities.mean(dim=0)  # 对每个维度的相似度取平均
                    grad_sim_mask = (grad_sim_mask >= grad_sim_mask.mean()).float()  # 根据相似度的平均值生成掩码

                    # 更新掩码
                    self.mask[name] = grad_sim_mask * original_mask

        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')



    def train_step(self, data, labels):
        """
        执行一次训练步骤。 

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                mask = self.mask[name].to('cuda') # param.grad.device
                if param.grad is None:
                    print(f"Parameter {name} has no gradient. requires_grad={param.requires_grad}")
                else:
                    # 获取该层的均值
                    mean_value = self.layer_mean_values.get(name, None)
                    if mean_value is not None:
                        # 如果均值存在，应用均值阈值
                        grad_mask = (param.grad.abs() >= mean_value).to(param.grad.device)  # 确保 grad_mask 在 param.grad 的设备上
                        param.grad = param.grad * grad_mask.float()  # 梯度小于均值的部分置为0

                    # 再应用原本的 mask，确保只有被选择的部分参与梯度更新
                    param.grad = param.grad * mask
        
        # 更新网络参数
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        return loss.item()

    def predict(self, data):
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y
# 残差修改

from torchvision.models.resnet import Bottleneck




# 这个算的是全部的掩码，比较难看
# def optimize_final_mask_by_variance(self, data_list, labels_list):
#         """
#         进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

#         Args:
#             data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
#             labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
#         """
#                 # Step 1: 预先计算每一层在每个域上的重要性
#         all_importance = {}  # 用于存储所有层在不同域上的重要性分数

#         for domain_idx in range(self.args['k']):
#             current_data = data_list[domain_idx]
#             current_labels = labels_list[domain_idx]
#             importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

#             # 将重要性分数存入 all_importance 字典
#             for name, importance in importance_scores.items():
#                 if name not in all_importance:
#                     all_importance[name] = []
#                 all_importance[name].append(importance.detach().cpu())
#         print('finish compute score')

#         for name, mask_tensor in self.mask.items():
#             if any(layer_name in name for layer_name in self.selected_layers):
#                 original_mask = mask_tensor.clone()

#                 # 从预先计算好的重要性分数中读取目标层的值
#                 if name in all_importance:
#                     param_grad_var = all_importance[name]
#                     grad_stack = torch.stack(param_grad_var)
#                     grad_var = grad_stack.var(dim=0)  # 计算梯度的方差

#                     # 根据方差生成掩码（不使用 for 循环）
#                     optimized_mask = (grad_var < grad_var.mean()).float().to(original_mask.device)

#                     # 更新掩码
#                     self.mask[name] = optimized_mask * original_mask

#         total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
#         print(f'Total number of ones in the mask: {total_count}')

# 得定义一个类来完成这一项,但是那个该死的bottelneck有问题
def custom_forward(self, x, extra_param=None): # 这里extra 有问题，应该是没训练的 因为下一次就没有调用了
        # 在这里实现你自己的前向传播逻辑
            identity = x  # 保存输入以便进行残差连接

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)  # 如果有下采样，调整输入的形状
            if extra_param is not None:
                # print(f"当前 extra_param 的 ID: {id(extra_param)}")
                extra_param1 = extra_param.view(1, -1, 1, 1)
                identity = identity * extra_param1   # 示例：将输出乘以额外参数

            out += identity  # 残差连接
            out = self.relu(out)  # 最后的激活
            return out


def custom_forward2(mod, x, extra_param=None):
    identity = x

    out = mod.conv1(x)
    out = mod.bn1(out)
    out = mod.relu(out)

    out = mod.conv2(out)
    out = mod.bn2(out)
    out = mod.relu(out)

    out = mod.conv3(out)
    out = mod.bn3(out)

    if mod.downsample is not None:
        identity = mod.downsample(x)

    if extra_param is not None:
        # print(extra_param)
        extra_param1 = F.relu(extra_param.view(1, -1, 1, 1))
        identity = identity * extra_param1

    out += identity
    out = mod.relu(out)

    return out





class SPU_step_direct:  # 匹配的是一个列表不再是一个值
    def __init__(self, args, Tnetwork, lr, weight_decay, classifier):
        self.args = args  # args 现在是字典
        self.network = Tnetwork  # 模型
        self.mask = {}
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = None  # 优化器将在 compute_importance 中初始化
        self.classifier = classifier
        self.optimizer2 = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.selection_rate = args['selection_rate']
        self.selected_layers = args['selected_layers'].split()
        self.way = args['way']

    def compute_score(self, data_list, labels_list, name=None):
        # 初始化 importance 字典
        importance = {name: torch.zeros_like(param, device='cpu') 
                    for name, param in self.network.named_parameters() 
                    if any(layer_name in name for layer_name in self.selected_layers)}

        # 设置网络为训练模式
        self.network.train()
        self.network.zero_grad()

        # 遍历每个数据批次
        for data, labels in zip(data_list, labels_list):
            # 前向传播
            logits_per_image = self.network(data)  # 获取当前批次的输出
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度

            # 累加梯度
            for name, param in self.network.named_parameters():
                if any(layer_name in name for layer_name in self.selected_layers) and param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.detach().cpu().clone()  # 累加梯度

            # 清除梯度，避免累加
            self.network.zero_grad()

        return importance

    def compute_importance(self, data_list, labels_list, name=None, save = False ):
        print('Compute importance for the current task...')
        save_impor = False
        domain_masks = []
        if save_impor:
            domain_importance = []
        # print(len(data_list))
            # First ensure all elements in data_list and labels_list are tensors with compatible shapes
        combined_data = torch.cat([x if torch.is_tensor(x) else torch.stack(x) for x in data_list], dim=0)
        combined_labels = torch.cat([x if torch.is_tensor(x) else torch.stack(x) for x in labels_list], dim=0)

        # Compute importance score once for the combined batch
        importance = self.compute_score(combined_data, combined_labels)

        if save_impor:
            domain_importance.append(importance)  # Or handle differently if needed

        # Create a single mask based on the combined importance
        mask = {}
        for name, param in self.network.named_parameters():
            if any(layer_name in name for layer_name in self.selected_layers) and name in importance:
                magnitudes = importance[name].abs()
                k = max(1, int(magnitudes.numel() * self.args['selection_rate']))
                topk_values, topk_indices = torch.topk(magnitudes.view(-1), k=k)
                mask_tensor = torch.zeros_like(magnitudes).to(self.args['device'])
                mask_tensor.view(-1)[topk_indices] = 1
                mask[name] = mask_tensor

        # If you need to maintain compatibility with the original structure 
        # where domain_masks is expected to have k elements, you can replicate
        # the same mask k times (though they'll all be identical)
        domain_masks = [mask.copy() for _ in range(self.args['k'])]
        if save_impor:
            save_dir = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_r_pacs_vit.pth'
            self.save_masks_and_importance(domain_masks,domain_importance,save_dir)

        
            


        # if False: # 存一下各个域的掩码方便研究
        #     combined_domain_mask = {}

        #     # 遍历每个域，保存对应的掩码
        #     for domain_idx, domain_mask in enumerate(domain_masks):
        #         # 对每个域的掩码进行转换，确保它们在 CPU 上
        #         cpu_domain_mask = {name: mask.cpu() for name, mask in domain_mask.items()}
        #         combined_domain_mask[f'domain_{domain_idx}'] = cpu_domain_mask

        #     # 保存所有域的掩码为一个文件
        #     torch.save(combined_domain_mask, '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_masks.pth')
        #     print("Domain masks have been saved as 'combined_domain_masks.pth'.")
        
            # save_masks_and_importance(domain_masks, domain_importance, combined_save_path)
        
        final_mask = {}
        # overlap_ratios = {}  # 存储每一层的重叠率

        for name, param in self.network.named_parameters():
            # 判断是否属于指定层
            if any(layer_name in name for layer_name in self.selected_layers):
                if name in domain_masks[0]:
                    layer_mask = domain_masks[0][name].clone()
                else:
                    layer_mask = torch.ones_like(param, device=self.args['device'])
                
                # 依次与其他domain_masks相乘
                for domain_idx in range(1, len(domain_masks)):
                    if name in domain_masks[domain_idx]:
                        layer_mask = layer_mask * domain_masks[domain_idx][name]
                    else:
                        layer_mask = layer_mask * torch.ones_like(param, device=self.args['device'])
                
                # 保存到final_mask
                final_mask[name] = layer_mask

                # # 计算剩余部分占比原mask的比值，实际上就是重叠率
                # original_count = torch.sum(domain_masks[0][name])  # 全1矩阵的总和
                # final_mask_count = torch.sum(layer_mask == 1)       # final_mask中1的总数
                # overlap_ratios[name] = (final_mask_count / original_count).item()  # 计算比率并保存

        # 打印每一层的重叠率,等要用的时候再打开
        # for layer_name, overlap in overlap_ratios.items():
        #     print(f"Layer: {layer_name}, Overlap Ratio: {overlap:.4f}")

        selected_params = []
        for name, param in self.network.named_parameters():
            if name in final_mask:
                if final_mask[name].sum() == 0:
                    print(f"Layer {name} is excluded from training (mask is all zeros).")
                    del final_mask[name]
                else:
                    param.requires_grad = True
                    selected_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(
            selected_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.mask = final_mask
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')

        # 第二步：基于梯度方差优化最终掩码
        if self.way == 'var':
            self.optimize_final_mask_by_variance(data_list, labels_list)
        elif self.way == 'dir':
            self.optimize_final_mask_by_center_direction(data_list, labels_list)
        elif self.way == 'mean':
            self.optimize_final_mask_by_mean(data_list, labels_list)
        elif self.way == 'qua':
            self.optimize_final_mask_by_quantile(data_list, labels_list)
        else:
            print('Select withou step2.')

    def optimize_final_mask_by_variance(self, data_list, labels_list,save = False):
        save = True
        """
        进一步优化最终掩码，通过梯度方差排除对梯度影响较大的维度。

        Args:
            data_list (List[Tensor]): 数据列表，每个元素为一个 batch 的数据。
            labels_list (List[Tensor]): 标签列表，每个元素为一个 batch 的标签。
        """
                # Step 1: 预先计算每一层在每个域上的重要性
        all_importance = {}  # 用于存储所有层在不同域上的重要性分数

        for domain_idx in range(self.args['k']):
            current_data = data_list[domain_idx]
            current_labels = labels_list[domain_idx]
            importance_scores = self.compute_score(current_data, current_labels)  # 计算所有层的分数

            # 将重要性分数存入 all_importance 字典
            for name, importance in importance_scores.items():
                if name not in all_importance:
                    all_importance[name] = []
                all_importance[name].append(importance.detach().cpu())
        print('finish compute score')
        if save:
            saved_masked_grad_vars = {}

        for name, mask_tensor in self.mask.items():
            if any(layer_name in name for layer_name in self.selected_layers):
                original_mask = mask_tensor.clone()

                # 从预先计算好的重要性分数中读取目标层的值
                if name in all_importance:
                    param_grad_var = all_importance[name]
                    grad_stack = torch.stack(param_grad_var)
                 
                    grad_var = grad_stack.var(dim=0)  # 计算梯度的方差
                    grad_var = grad_var.to(original_mask.device)
                    # print(f"grad_var device: {grad_var.device}")
                    # print(f"original_mask device: {original_mask.device}")
                    masked_grad_var = grad_var[original_mask > 0]
                    if save:
                        saved_masked_grad_vars[name] = masked_grad_var.cpu().numpy()

                    # 计算被掩码部分的均值
                    masked_mean = masked_grad_var.mean()

                    # 根据方差生成掩码（不使用 for 循环）
                    optimized_mask = (grad_var < masked_mean).float().to(original_mask.device)

                    # 更新掩码
                    self.mask[name] = optimized_mask * original_mask
        # save_path = "/home/home_node7/wzb/自己算法/miro-main2/result/draw/masked_grad_vars.pkl"
        # if save:
        #     with open(save_path, "wb") as f:
        #         pickle.dump(saved_masked_grad_vars, f)

        #     print(f"Saved masked_grad_var to {save_path}")
        total_count = sum(mask_tensor.sum().item() for mask_tensor in self.mask.values())
        print(f'Total number of ones in the mask: {total_count}')
    

    def train_step(self, data, labels):
        """
        执行一次训练步骤。 

        Args:
            data (Tensor): 输入数据，形状为 [B, C, H, W]。
            labels (Tensor): 标签，形状为 [B]。

        Returns:
            float: 损失值。
        """
        if self.optimizer is None:
            raise ValueError("Optimizer has not been initialized. Call compute_importance first.")

        self.network.train()
        self.classifier.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()
        
        # Forward pass through network
        features = self.network(data)
        y = self.classifier(features)
        
        loss = torch.nn.functional.cross_entropy(y, labels)
        loss.backward()
        
        
        # 应用掩码到梯度
        for name, param in self.network.named_parameters():
            if name in self.mask:
                # mask = self.mask[name].to('cuda') # param.grad.device
                if param.grad is None:
                    print(f"Parameter {name} has no gradient. requires_grad={param.requires_grad}")
                else:
                    param.grad = param.grad * self.mask[name]
        memory1 = torch.cuda.memory_allocated() / (1024 ** 3)
        
        # 更新网络参数
        
        self.optimizer.step()
        
        # 更新分类器参数
        self.optimizer2.step()
        return [loss.item(),memory1]

    def predict(self, data):
        self.network.eval()
        self.classifier.eval()
        with torch.no_grad():
            features = self.network(data)
            y = self.classifier(features)
        return y
    
    def analyze_mask(self):
        results = {}
        for name, matrix in self.mask.items():
            if not isinstance(matrix, torch.Tensor):
                raise ValueError(f"The value associated with key '{name}' is not a tensor.")
            
            # 跳过第二个维度不大于 2 的矩阵
            if matrix.dim() < 2 or matrix.size(1) <= 2:
                continue
            
            # 计算矩阵中1的个数
            ones_count = torch.sum(matrix).item()
            
            # 确保矩阵是浮点类型
            matrix_float = matrix.to(dtype=torch.float32)
            
            # 计算矩阵的秩
            _, singular_values, _ = torch.linalg.svd(matrix_float)
            rank = torch.sum(singular_values > 1e-10).item()
            
            # 保存结果
            results[name] = {"ones_count": ones_count, "rank": rank}
            print(name,ones_count,rank)
        
        # return results

    def save_masks_and_importance(self,domain_masks, domain_importance, save_path):
        """
        保存域的掩码和重要性信息到指定路径。
        
        参数:
        - domain_masks: List[Dict[str, Tensor]]，每个域的掩码字典。
        - domain_importance: List[Dict[str, Tensor]]，每个域的梯度/重要性字典。
        - save_path: str，保存文件的路径。
        """
        combined_data = {}

        # 遍历每个域，保存对应的掩码和重要性信息
        for domain_idx, (domain_mask, domain_imp) in enumerate(zip(domain_masks, domain_importance)):
            # 对每个域的掩码和重要性信息进行转换，确保它们在 CPU 上
            cpu_domain_mask = {name: mask.cpu() for name, mask in domain_mask.items()}
            cpu_domain_importance = {name: imp.cpu() for name, imp in domain_imp.items()}
            combined_data[f'domain_{domain_idx}'] = {
                'masks': cpu_domain_mask,
                'importance': cpu_domain_importance
            }

        # 保存所有域的数据为一个文件
        torch.save(combined_data, save_path)
        print(f"Masks and importance have been saved to '{save_path}'.")
