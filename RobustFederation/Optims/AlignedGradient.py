import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from Optims.utils.federated_optim import FederatedOptim
import numpy as np
import torch
import torch.nn.functional as F

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

# def prograd_backward_and_update(model, optimizer, loss_a, loss_b, count,lambda_= 1):
#     # 检查损失是否有效
#     if not torch.isfinite(loss_b).all() or not torch.isfinite(loss_a).all():
#         raise FloatingPointError("Loss is infinite or NaN!")
#
#     # 反向传播 loss_b 并保留计算图
#     loss_b.backward(retain_graph=True)
#
#     # 获取梯度
#     b_grads = []
#     for p in model.parameters():
#         b_grads.append(p.grad.clone().detach())
#
#     # 清除梯度以准备下一次反向传播
#     optimizer.zero_grad()
#
#     # 反向传播 loss_a
#     loss_a.backward()
#
#
#     # 调整梯度方向
#     for p, b_grad in zip(model.parameters(), b_grads):
#         # 归一化梯度
#         b_grad_norm = b_grad / torch.linalg.norm(b_grad)
#         a_grad = p.grad.clone().detach()
#         a_grad_norm = a_grad / torch.linalg.norm(a_grad)
#
#         # 如果两个梯度的方向相反
#         if torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten()) < 0:
#             count += 1
#             # note:90度
#             # p.grad = a_grad - lambda_ * torch.dot(
#             #     a_grad.flatten(), b_grad_norm.flatten()
#             # ) * b_grad_norm
#             p.grad = a_grad - lambda_ * torch.dot(
#                 a_grad.flatten(), b_grad_norm.flatten()
#             ) * b_grad_norm
#             # # note：60度
#             # # 计算新的梯度方向
#             # # 新的梯度方向应该与 b_grad 成60度角
#             # # 计算调整后的梯度方向
#             # proj_component = torch.dot(a_grad.flatten(), b_grad_norm.flatten())
#             # adjusted_grad = a_grad - proj_component * b_grad_norm + \
#             #                 (proj_component * torch.cos(
#             #                     torch.tensor(60.0) * torch.pi / 180.0) - proj_component) * b_grad_norm
#             # adjusted_grad = adjusted_grad / torch.linalg.norm(adjusted_grad) * torch.linalg.norm(a_grad)
#             #
#             # # 更新梯度方向
#             # p.grad = adjusted_grad
#         else:
#             p.grad = a_grad
#     # print('lamda:%d' % lambda_)
#     # 更新参数
#     optimizer.step()
#     print(count)




class AlignedGradient(FederatedOptim):
    NAME = 'AlignedGradient'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(AlignedGradient, self).__init__(nets_list, client_domain_list, args, cfg)
        self.bad_scale = int(cfg.DATASET.parti_num * cfg['attack'].bad_client_rate)
        self.good_scale = cfg.DATASET.parti_num - self.bad_scale
        self.client_type = np.repeat(True, self.good_scale).tolist() + (np.repeat(False, self.bad_scale)).tolist()
        self.lamda = cfg.Optim[self.NAME].lamda
        self.count = 0


    def ProGradLoss(self, output, teacher_output, target, T=1):
        xe_loss = F.cross_entropy(output, target)
        xe_loss.to(self.device)
        # print(output.equal(teacher_output))
        # print(output)
        # print(teacher_output)

        # 假设 teacher_logits 为Output
        tea_prob = F.softmax(teacher_output / T, dim=-1)
        # kl_loss = -tea_prob * F.log_softmax(output / T, dim=-1) * T * T
        kl_loss = -tea_prob * F.log_softmax(output, dim=-1) * T * T
        kl_loss.to(self.device)
        kl_loss = kl_loss.sum(1).mean()

        return xe_loss, kl_loss

    def prograd_backward_and_update(self, model, optimizer, loss_a, loss_b, lambda_=1):
        # 检查损失是否有效
        if not torch.isfinite(loss_b).all() or not torch.isfinite(loss_a).all():
            raise FloatingPointError("Loss is infinite or NaN!")

        # 反向传播 loss_b 并保留计算图
        loss_b.backward(retain_graph=True)

        # 获取梯度
        b_grads = {}
        for name, para in model.named_parameters():
            b_grads[name] = para.grad.clone()
            # print(b_grads[name])

        p_grad = {}
        for p in model.parameters():
            a = p.grad

        # 清除梯度以准备下一次反向传播
        optimizer.zero_grad()

        # 反向传播 loss_a
        loss_a.backward()
        a_grads = {}
        for name, para in model.named_parameters():
            a_grads[name] = para.grad.clone()
            b_grad_norm = b_grads[name] / torch.linalg.norm(b_grads[name])
            a_grad_norm = a_grads[name] / torch.linalg.norm(a_grads[name])
            if torch.dot(a_grad_norm.flatten(), b_grad_norm.flatten()) < 0:
                self.count += 1
                # note:90度
                # p.grad = a_grad - lambda_ * torch.dot(
                #     a_grad.flatten(), b_grad_norm.flatten()
                # ) * b_grad_norm
                para.grad = a_grads[name] - lambda_ * torch.dot(
                    a_grads[name].flatten(), b_grad_norm.flatten()
                ) * b_grad_norm
            else:
                para.grad = a_grads[name]
        optimizer.step()


    def load_model_weight(self, model, params):
        # 加载向量化的参数到模型
        return torch.nn.utils.vector_to_parameters(torch.from_numpy(params), model.parameters())



    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)



    def loc_update(self,priloader_list,freq = None, epoch = None, excepted_models_list = None, bad_loader = None, good_loader = None):
        total_clients = list(range(self.cfg.DATASET.parti_num))
        self.online_clients_list  = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        for i in self.online_clients_list:
            self._train_net(i,self.nets_list[i], priloader_list[i],excepted_models_list[i])
        return  None

    def _train_net(self,index,net,train_loader,others_nets):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9,weight_decay=self.weight_decay)

        # criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        self.count = 0
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                tea_output = others_nets(images)
                # loss = criterion(outputs, labels)
                xe_loss, kl_loss = self.ProGradLoss(outputs, tea_output, labels)
                optimizer.zero_grad()
                self.prograd_backward_and_update(net, optimizer, xe_loss, kl_loss,lambda_= self.lamda)
                iterator.desc = "Local Participant %d celoss = %0.3f, klloss = %0.3f" % (index,xe_loss,kl_loss)
                # optimizer.step()
        # print(self.count)

    def _Fedavg_train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
                optimizer.step()

    def _val_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        criterion.to(self.device)
        # iterator = tqdm(range(self.local_epoch))
        # for _ in iterator:
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            # iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
            optimizer.step()
