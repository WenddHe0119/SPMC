import copy
import numpy as np
import torch
from Server.utils.server_methods import ServerMethod
from utils.utils import row_into_parameters


class FedCPA(ServerMethod):
    NAME = 'FedCPA'

    def __init__(self, args, cfg):
        super(FedCPA, self).__init__(args, cfg)
        self.top_rate = cfg.Server[self.NAME].top_rate

    def get_update_static(self, net_list, global_net):
        model_weight_list = []
        net_id_list = []

        glboal_net_para = global_net.state_dict()
        global_weight = self.get_weight(glboal_net_para).unsqueeze(0)

        for net_id, net in enumerate(net_list):
            net_id_list.append(net_id)
            net_para = net.state_dict()
            model_weight = self.get_weight(net_para).unsqueeze(0)
            model_update = model_weight - global_weight
            model_weight_list.append(model_update)
        model_weight_cat = torch.cat(model_weight_list, dim=0)
        model_std, model_mean = torch.std_mean(model_weight_cat, unbiased=False, dim=0)

        return model_mean, model_std, model_weight_cat, global_weight

    def get_weight(self, model_weight):
        weight_tensor_result = []
        for k, v in model_weight.items():
            weight_tensor_result.append(v.reshape(-1).float())
        weights = torch.cat(weight_tensor_result)
        return weights

    def get_foolsgold_score(self, total_score, grads, global_weight):
        n_clients = total_score.shape[0]
        norm_score = total_score

        wv = (norm_score - np.min(norm_score)) / (np.max(norm_score) - np.min(norm_score))
        wv[(wv == 1)] = .99
        wv[(wv == 0)] = .01

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        model_weight_list = []
        for i in range(0, n_clients):
            if wv[i] != 0:
                current_weight = global_weight + wv[i] * grads[i]
                model_weight_list.append(current_weight)
        fools_gold_weight = torch.cat(model_weight_list).mean(0, keepdims=True)

        return fools_gold_weight.view(-1), wv

    def server_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']
        temp_net = copy.deepcopy(global_net)

        import scipy
        self.prev_prev_global_w = copy.deepcopy(global_net.state_dict())
        self.prev_global_w = copy.deepcopy(global_net.state_dict())

        for net_id, net in enumerate(nets_list):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    self.prev_global_w[key] = net_para[key] / len(nets_list)
            else:
                for key in net_para:
                    self.prev_global_w[key] += net_para[key] / len(nets_list)

        local_global_w_list = []

        global_para = global_net.state_dict()
        global_critical_dict = {}
        for name, val in global_para.items():
            if val.dim() in [2, 4]:
                critical_weight = torch.abs(
                    (self.prev_global_w[name] - self.prev_prev_global_w[name]) * self.prev_global_w[name])
                global_critical_dict[name] = critical_weight

        global_w_stacked = self.get_weight(global_critical_dict).view(1, -1)
        global_topk_indices = torch.abs(global_w_stacked).topk(int(global_w_stacked.shape[1] * self.top_rate)).indices
        global_bottomk_indices = torch.abs(global_w_stacked).topk(int(global_w_stacked.shape[1] * self.top_rate),
                                                                  largest=False).indices

        for net in nets_list:
            net_para = net.state_dict()
            critical_dict = {}
            for name, val in net_para.items():
                if val.dim() in [2, 4]:
                    critical_weight = torch.abs((val - self.prev_global_w[name]) * val)
                    critical_dict[name] = critical_weight

            local_global_w_list.append(self.get_weight(critical_dict))

        w_stacked = torch.stack(local_global_w_list, dim=0)
        local_topk_indices = torch.abs(w_stacked).topk(int(w_stacked.shape[1] * self.top_rate)).indices
        local_bottomk_indices = torch.abs(w_stacked).topk(int(w_stacked.shape[1] * self.top_rate),
                                                          largest=False).indices

        pairwise_score = np.zeros((len(nets_list), len(nets_list)))
        for i in range(len(nets_list)):
            for j in range(len(nets_list)):
                if i == j:
                    pairwise_score[i][j] = 1
                elif i < j:
                    continue

                topk_intersection = list(set(local_topk_indices[i].tolist()) & set(local_topk_indices[j].tolist()))
                topk_corr_dist = ((scipy.stats.pearsonr(w_stacked[i, topk_intersection].cpu().numpy(),
                                                        w_stacked[j, topk_intersection].cpu().numpy())[0]) + 1) / 2
                topk_jaccard_dist = len(topk_intersection) / (
                        len(local_topk_indices[i]) + len(local_topk_indices[j]) - len(topk_intersection))

                bottomk_intersection = list(
                    set(local_bottomk_indices[i].tolist()) & set(local_bottomk_indices[j].tolist()))
                bottomk_corr_dist = ((scipy.stats.pearsonr(w_stacked[i, bottomk_intersection].cpu().numpy(),
                                                           w_stacked[j, bottomk_intersection].cpu().numpy())[
                    0]) + 1) / 2
                bottomk_jaccard_dist = len(bottomk_intersection) / (
                        len(local_bottomk_indices[i]) + len(local_bottomk_indices[j]) - len(bottomk_intersection))

                pairwise_score[i][j] = (topk_corr_dist + bottomk_corr_dist) / 2 + (
                        topk_jaccard_dist + bottomk_jaccard_dist) / 2
                pairwise_score[j][i] = (topk_corr_dist + bottomk_corr_dist) / 2 + (
                        topk_jaccard_dist + bottomk_jaccard_dist) / 2

        global_score = np.zeros(len(nets_list))
        for i in range(len(nets_list)):
            topk_intersection = list(set(local_topk_indices[i].tolist()) & set(global_topk_indices[0].tolist()))
            topk_corr_dist = ((scipy.stats.pearsonr(w_stacked[i, topk_intersection].cpu().numpy(),
                                                    global_w_stacked[0, topk_intersection].cpu().numpy())[0]) + 1) / 2
            topk_jaccard_dist = len(topk_intersection) / (
                    len(local_topk_indices[i]) + len(global_topk_indices[0]) - len(topk_intersection))

            bottomk_intersection = list(
                set(local_bottomk_indices[i].tolist()) & set(global_bottomk_indices[0].tolist()))
            bottomk_corr_dist = ((scipy.stats.pearsonr(w_stacked[i, bottomk_intersection].cpu().numpy(),
                                                       global_w_stacked[0, bottomk_intersection].cpu().numpy())[
                0]) + 1) / 2
            bottomk_jaccard_dist = len(bottomk_intersection) / (
                    len(local_bottomk_indices[i]) + len(global_bottomk_indices[0]) - len(bottomk_intersection))

            global_score[i] = (topk_corr_dist + bottomk_corr_dist) / 2 + (topk_jaccard_dist + bottomk_jaccard_dist) / 2

        total_score = np.mean(pairwise_score, axis=1) + global_score

        update_mean, update_std, update_cat, global_weight = self.get_update_static(nets_list, global_net)
        model_weight_foolsgold, wv = self.get_foolsgold_score(total_score, update_cat, global_weight)

        global_w = global_net.state_dict()
        current_idx = 0
        for key in net_para:
            length = len(net_para[key].reshape(-1))
            global_w[key] = model_weight_foolsgold[current_idx:current_idx + length].reshape(net_para[key].shape)
            current_idx += length

        self.prev_prev_global_w = copy.deepcopy(self.prev_global_w)
        self.prev_global_w = copy.deepcopy(global_net.state_dict())

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())
