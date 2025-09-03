from Server.utils.server_methods import ServerMethod
from utils.utils import row_into_parameters,row_into_state_dict
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import torch
import copy


class FLTrust(ServerMethod):
    NAME = 'FLTrust'

    def __init__(self, args, cfg):
        super(FLTrust, self).__init__(args, cfg)
        self.fltrust_style = 'per_client'
    # def server_update(self, **kwargs):
    #
    #     online_clients_list = kwargs['online_clients_list']
    #     global_net = kwargs['global_net']
    #     nets_list = kwargs['nets_list']
    #     val_loader = kwargs['val_loader']
    #
    #     temp_net = copy.deepcopy(global_net)
    #     with torch.no_grad():
    #         all_delta = []
    #         global_net_para = []
    #         add_global = True
    #         for i in online_clients_list:
    #
    #             net_all_delta = []
    #             for name, param0 in temp_net.state_dict().items():
    #                 param1 = nets_list[i].state_dict()[name]
    #                 delta = (param1.detach() - param0.detach())
    #
    #                 net_all_delta.append(copy.deepcopy(delta.view(-1)))
    #                 if add_global:
    #                     weights = copy.deepcopy(param0.detach().view(-1))
    #                     global_net_para.append(weights)
    #
    #             add_global = False
    #             net_all_delta = torch.cat(net_all_delta, dim=0).cpu().numpy()
    #             all_delta.append(net_all_delta)
    #
    #         all_delta = np.array(all_delta)
    #         global_net_para = np.array(torch.cat(global_net_para, dim=0).cpu().numpy())
    #
    #     criterion = nn.CrossEntropyLoss()
    #     iterator = tqdm(range(self.cfg.Server.FLTrust.public_epoch))
    #     optimizer = optim.SGD(temp_net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr,
    #                           momentum=self.cfg.OPTIMIZER.momentum, weight_decay=self.cfg.OPTIMIZER.weight_decay)
    #     for _ in iterator:
    #         for batch_idx, (images, labels) in enumerate(val_loader):
    #             images = images
    #             images = images.to(self.device)
    #             labels = labels.to(self.device)
    #             outputs = temp_net(images)
    #             loss = criterion(outputs, labels)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #     with torch.no_grad():
    #         global_delta = []
    #         for name, param0 in temp_net.state_dict().items():
    #             param1 = global_net.state_dict()[name]
    #             delta = (param0.detach() - param1.detach())
    #             global_delta.append(copy.deepcopy(delta.view(-1)))
    #
    #         global_delta = torch.cat(global_delta, dim=0).cpu().numpy()
    #         global_delta = np.array(global_delta)
    #
    #     total_TS = 0
    #     TSnorm = []
    #     for d in all_delta:
    #         tmp_weight = copy.deepcopy(d)
    #
    #         TS = np.dot(tmp_weight, global_delta) / (np.linalg.norm(tmp_weight) * np.linalg.norm(global_delta) + 1e-5)
    #         # print(TS)
    #         if TS < 0:
    #             TS = 0
    #         total_TS += TS
    #
    #         norm = np.linalg.norm(global_delta) / (np.linalg.norm(tmp_weight) + 1e-5)
    #         TSnorm.append(TS * norm)
    #
    #     delta_weight = np.sum(np.array(TSnorm).reshape(-1, 1) * all_delta, axis=0) / (total_TS + 1e-5)
    #     new_global_net_para = global_net_para + delta_weight
    #     # row_into_parameters(new_global_net_para, global_net.parameters())
    #     row_into_state_dict(new_global_net_para, global_net.state_dict())
    #     for _, net in enumerate(nets_list):
    #         net.load_state_dict(global_net.state_dict())

    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        val_loader = kwargs['val_loader']

        temp_net = copy.deepcopy(global_net).to(self.device)

        state_keys = list(global_net.state_dict().keys())

        # === 客户端 delta 按层收集 ===
        client_deltas = {name: [] for name in state_keys}
        for i in online_clients_list:
            client_state = nets_list[i].state_dict()
            for name in state_keys:
                delta = (client_state[name] - global_net.state_dict()[name]).detach().to(torch.float32).view(
                    -1).cpu().numpy()
                client_deltas[name].append(delta)

        # === 使用验证集训练 clean 模型（temp_net） ===
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            temp_net.parameters(),
            lr=self.cfg.OPTIMIZER.local_train_lr,
            momentum=self.cfg.OPTIMIZER.momentum,
            weight_decay=self.cfg.OPTIMIZER.weight_decay
        )
        temp_net.train()
        for _ in range(self.cfg.Server.FLTrust.public_epoch):
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                loss = criterion(temp_net(images), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # === 获取 clean model 的 global delta（参考） ===
        global_deltas = {}
        for name in state_keys:
            delta = (temp_net.state_dict()[name] - global_net.state_dict()[name]).detach().to(torch.float32).view(
                -1).cpu().numpy()
            global_deltas[name] = delta

        updated_dict = {}

        # === 聚合逻辑 ===
        if self.fltrust_style == 'per_client':
            # 计算每个客户端的整体 delta 向量
            client_flat = []
            for i in online_clients_list:
                vec = []
                for name in state_keys:
                    delta = (nets_list[i].state_dict()[name] - global_net.state_dict()[name]).detach().to(
                        torch.float32).view(-1).cpu().numpy()
                    vec.append(delta)
                client_flat.append(np.concatenate(vec))
            global_flat = np.concatenate([global_deltas[name] for name in state_keys])

            # 计算 trust score
            trust_scores = []
            total_ts = 0.0
            for vec in client_flat:
                cos = np.dot(vec, global_flat) / (np.linalg.norm(vec) * np.linalg.norm(global_flat) + 1e-5)
                cos = max(cos, 0.0)
                norm = np.linalg.norm(global_flat) / (np.linalg.norm(vec) + 1e-5)
                ts = cos * norm
                trust_scores.append(ts)
                total_ts += ts

            # 聚合每层
            for name in state_keys:
                layer_sum = np.zeros_like(global_deltas[name], dtype=np.float32)
                for i in range(len(online_clients_list)):
                    layer_sum += trust_scores[i] * client_deltas[name][i]
                avg_delta = layer_sum / (total_ts + 1e-5)
                avg_tensor = torch.from_numpy(avg_delta).view(global_net.state_dict()[name].shape).to(
                    global_net.state_dict()[name].device).to(global_net.state_dict()[name].dtype)
                updated_dict[name] = global_net.state_dict()[name] + avg_tensor

        elif self.fltrust_style == 'per_layer':
            # 每层各自计算 trust score 并加权
            for name in state_keys:
                layer_client_deltas = np.array(client_deltas[name], dtype=np.float32)
                global_delta = global_deltas[name]
                total_ts = 0.0
                weighted_sum = np.zeros_like(global_delta, dtype=np.float32)

                for i in range(len(online_clients_list)):
                    client_delta = layer_client_deltas[i]
                    cos = np.dot(client_delta, global_delta) / (
                                np.linalg.norm(client_delta) * np.linalg.norm(global_delta) + 1e-5)
                    cos = max(cos, 0.0)
                    norm = np.linalg.norm(global_delta) / (np.linalg.norm(client_delta) + 1e-5)
                    ts = cos * norm
                    weighted_sum += ts * client_delta
                    total_ts += ts

                avg_delta = weighted_sum / (total_ts + 1e-5)
                avg_tensor = torch.from_numpy(avg_delta).view(global_net.state_dict()[name].shape).to(
                    global_net.state_dict()[name].device).to(global_net.state_dict()[name].dtype)
                updated_dict[name] = global_net.state_dict()[name] + avg_tensor

        else:
            raise ValueError(f"Unsupported fltrust_style: {self.fltrust_style}")

        # === 更新 global_net 并同步 ===
        global_net.load_state_dict(updated_dict)
        for net in nets_list:
            net.load_state_dict(global_net.state_dict())


