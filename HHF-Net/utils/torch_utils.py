import torch
import torch.nn as nn

class JointBoneLoss(nn.Module):
    def __init__(self, joint_num):
        super(JointBoneLoss, self).__init__()
        id_i, id_j = [], []
        for i in range(joint_num):
            for j in range(i+1, joint_num):
                id_i.append(i)
                id_j.append(j)
        self.id_i = id_i
        self.id_j = id_j

    def forward(self, joint_out, joint_gt):
        # 判断输入的形状
        if len(joint_out.shape) == 4:  # (b, n, h, w) heatmap-based featuremap
            calc_dim = [2, 3]  # 计算维度为 height 和 width
        elif len(joint_out.shape) == 3:  # (b, n, 2) 或 (b, n, 3) regression-based result
            calc_dim = -1  # 计算最后一个维度（x,y 或 x,y,z）

        # 计算预测关节之间的距离 (J)
        J = torch.norm(joint_out[:, self.id_i, :] - joint_out[:, self.id_j, :], p=2, dim=calc_dim, keepdim=False)

        # 计算目标关节之间的距离 (Y)
        Y = torch.norm(joint_gt[:, self.id_i, :] - joint_gt[:, self.id_j, :], p=2, dim=calc_dim, keepdim=False)

        # 计算骨骼损失 (关节预测距离与目标距离的绝对差)
        loss = torch.abs(J - Y)
        return loss.mean()

