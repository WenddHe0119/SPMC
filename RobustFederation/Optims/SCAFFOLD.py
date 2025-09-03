import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from Optims.utils.federated_optim import FederatedOptim


class SCAFFOLD(FederatedOptim):
    NAME = 'SCAFFOLD'

    def __init__(self, nets_list, client_domain_list, args, cfg):
        super(SCAFFOLD, self).__init__(nets_list, client_domain_list, args, cfg)
        # 初始化每个客户端的控制变量c_i和全局控制变量c_0
        self.control_vars_clients = [copy.deepcopy(net).to(self.device) for net in nets_list]
        self.control_var_server = copy.deepcopy(nets_list[0]).to(self.device)
        for param in self.control_var_server.parameters():
            param.data.zero_()

    def ini(self):

        super().ini()
        # 为所有客户端网络初始化控制变量为零
        for cv in self.control_vars_clients:
            for param in cv.parameters():
                param.data.zero_()
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list, freq=None, epoch=None, excepted_models_list=None, bad_loader=None,
                   good_loader=None):
        total_clients = list(range(self.cfg.DATASET.parti_num))
        self.online_clients_list = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        if epoch > 1:
            for i in self.online_clients_list:
                self._train_net(i, self.nets_list[i], priloader_list[i], self.control_vars_clients[i])
        else:
            for i in self.online_clients_list:
                self._avg_train_net(i, self.nets_list[i], priloader_list[i])
        return None

    def _train_net(self, index, net, train_loader, control_var_client):
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss().to(self.device)

        iterator = tqdm(range(self.local_epoch))
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

                # 更新步骤：应用梯度并调整控制变量
                for p_net, p_c_local, p_c_global in zip(net.parameters(), control_var_client.parameters(),
                                                        self.control_var_server.parameters()):
                    adjust = (p_c_local.data - p_c_global.data).clamp_(-1.0, 1.0)
                    p_net.grad.data.sub_(adjust)

                optimizer.step()

                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss.item())

        # 更新客户端的控制变量
        for p_net, p_c_local, p_c_global in zip(net.parameters(), control_var_client.parameters(),
                                                self.control_var_server.parameters()):
            p_c_local.data.add_(p_net.grad.data / (len(train_loader) * self.local_lr) - p_c_global.data)

    def _avg_train_net(self, index, net, train_loader):
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
