# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """包装了一个现有的损失函数(loss_fcn)，并对原始损失函数的输出应用了Focal Loss修正。
    """
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        """pos_weight 是用于调整正类别的权重的参数。在二分类问题中，通常情况下负类别（negative class）的样本数量远远大于正类别（positive class）的样本数量。
        这样一来，模型可能会更倾向于预测为负类别，因为这样可以降低整体的损失。
        """
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device)) # 分类损失的损失函数 nn.BCEWithLogitsLoss，这是一个二分类交叉熵损失函数
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device)) # 目标损失的损失函数 pos_weight 参数用于对正样本和负样本的权重进行加权，这在处理样本不均衡的情况下很有用。

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3  类别标签平滑 (Class Label Smoothing):一种正则化技巧，用于训练分类模型时减缓模型的过拟合
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets 根据超参数中的设定，进行了类别平滑处理。  label_smoothing 是一个超参数，用于控制平滑的程度
        # self.cp 和 self.cn 分别是用于平滑的正类别和负类别的 BCE（二元交叉熵）目标。

        # Focal loss https://arxiv.org/pdf/1708.02002.pdf 一种损失函数，旨在解决类别不平衡问题，尤其是在存在大量负样本的情况下。
        g = h['fl_gamma']  # focal loss gamma fl_gamma 是 Focal Loss 中的一个超参数，控制了易分类样本的权重，通常设为正值以降低易分类样本的权重。
        if g > 0: # 如果 fl_gamma 大于 0，则会创建一个 Focal Loss 的实例，并将其用于分类损失 BCEcls 和目标损失 BCEobj 中
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module 从传入的模型 model 中获取最后一层，这里是一个Detect()模块。
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors # anchors 表示锚框的尺寸，它等于 m.anchors，即从模型 m 中获取的锚框尺寸。
        print(f"-------- m.anchors: {m.anchors}")
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        """类的主要调用函数，用于计算总的损失。

        Args:
            p (_type_): 模型的预测结果
            targets (_type_): 真实标签

        Returns:
            _type_: 返回加权后的总损失
        """
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets 调用build_targets函数构建目标

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions 遍历每个预测层，计算分类损失、目标框损失和目标损失，并累加到对应的变量中。
            print(f"---------------i: {i}, pi: {pi.shape}")
            """ 训练时
            ---------------i: 0, pi: torch.Size([16, 3, 80, 80, 10]) # 检测小目标 16 batch_size; 80x80 特征图块; 3 每个特征图块预测三个框; 10=5(类别)+4(位置)+1(目标置信度)
            ---------------i: 1, pi: torch.Size([16, 3, 40, 40, 10])
            ---------------i: 2, pi: torch.Size([16, 3, 20, 20, 10]) # 检测大目标
            """
            """ 模型评估（推断）
            ---------------i: 0, pi: torch.Size([32, 3, 64, 84, 10])
            ---------------i: 1, pi: torch.Size([32, 3, 32, 42, 10])
            ---------------i: 2, pi: torch.Size([32, 3, 16, 21, 10])  
            """
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                #  pxy（预测的中心坐标）、pwh（预测的宽高）、_（这个部分在计算目标框损失时没有用到）、pcls（预测的类别）
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression 目标框损失
                pxy = pxy.sigmoid() * 2 - 0.5 # 对预测的中心坐标进行了变换。它首先对坐标进行了 sigmoid 操作，将其范围限制在 (0, 1) 之间，然后乘以 2 并减去 0.5，将范围映射到 (-0.5, 0.5)。
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i] # 对预测的宽高进行了变换。类似于中心坐标，首先进行了 sigmoid 操作，然后乘以 2 的平方，最后再乘以对应的锚框尺寸。
                pbox = torch.cat((pxy, pwh), 1)  # predicted box 将处理后的中心坐标和宽高拼接在一起，得到了预测的目标框。
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target) 计算了预测框和目标框之间的 IoU（交并比）
                lbox += (1.0 - iou).mean()  # iou loss 计算了目标框回归的损失。它将 1 减去 IoU 得到的值，然后取平均，累加到 lbox 变量中。

                # Objectness 目标损失
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio 将处理后的 IoU 值赋值给 tobj，即目标张量，用于计算目标损失。

                # Classification 分类损失
                if self.nc > 1:  # cls loss (only if multiple classes) 只有在有多类别时才计算分类损失。
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets 创建一个与预测的类别张量 pcls 有相同形状的全1张量作为 targets。self.cn 是负类别权重，self.cp 是正类别权重。
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE 二元交叉熵损失函数计算分类损失。

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj) # 计算目标置信度的二元交叉熵损失。
            lobj += obji * self.balance[i]  # obj loss 将损失乘以相应的权重，然后加到总的目标置信度损失中。
            if self.autobalance: # 检查是否需要自动平衡权重
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box'] # 损失乘以超参数中的权重
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """构建目标

        Args:
            p (_type_): 模型的预测结果
            targets (_type_): 真实标签

        Returns:
            _type_: _description_
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h) 根据 anchor 数量和目标数量初始化了用于保存分类、目标框、索引和锚框信息的列表
        na, nt = self.na, targets.shape[0]  # number of anchors, targets na 表示锚框的数量，nt 表示目标的数量
        tcls, tbox, indices, anch = [], [], [], [] # 保存分类、目标框、索引和锚框信息。
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain grid空间的增益 gain，将目标坐标映射到 grid 空间
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt) ai 张量，它是一个形状为 (na, nt) 的张量，用于存储锚框的索引。
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices 目标张量 targets 与 ai 水平拼接，形成一个形状为 (na * nt, 7) 的张量，其中包含了目标的类别、坐标和锚框的索引信息。
        # print(f"--------------gain :{gain.shape},  --- ai: {ai.shape}, ----- targets: {targets.shape}")
        # iteration1 --------------gain :torch.Size([7]),  --- ai: torch.Size([3, 16]), ----- targets: torch.Size([3, 16, 7])
        # iteration2 --------------gain :torch.Size([7]),  --- ai: torch.Size([3, 14]), ----- targets: torch.Size([3, 14, 7])
        # ... nt是会变化的
        # iteration7 --------------gain :torch.Size([7]),  --- ai: torch.Size([3, 23]), ----- targets: torch.Size([3, 23, 7])

        g = 0.5  # bias 初始化了一个偏移量 off，它是一个形状为 (5, 2) 的张量，用于计算目标的偏移
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets
        # print(f"------------- self.nl: {self.nl}")
        # ------------- self.nl: 3
        for i in range(self.nl): # 遍历模型的不同层，对每一层进行处理
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors 将目标坐标与锚框的宽高比进行比较
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare  布尔张量 j，表示目标是否与锚框匹配。
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter 根据 j 对目标进行过滤

                # Offsets 计算目标在 grid 空间的坐标，并根据偏移量 off 计算目标的偏移
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append 目标的类别、坐标、索引和锚框信息
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
