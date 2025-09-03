import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from Server.utils.server_methods import ServerMethod


class RLR(ServerMethod):
    NAME = 'RLR'

    def __init__(self, args, cfg):
        super(RLR, self).__init__(args, cfg)

        self.server_lr = cfg.Server[self.NAME].server_lr
        self.robustLR_threshold = cfg.Server[self.NAME].robustLR_threshold

    def compute_robustLR(self, agent_updates_dict):
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_of_signs = torch.abs(sum(agent_updates_sign))

        sm_of_signs[sm_of_signs < self.robustLR_threshold] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.robustLR_threshold] = self.server_lr
        return sm_of_signs.to(self.device)

    def agg_avg(self, agent_updates_dict, freq, online_clients_list):
        """ classic fed avg """
        update = 0
        for index, net_id in enumerate(online_clients_list):
            delta = agent_updates_dict[net_id]
            update += delta * freq[index]

        return update

    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        with torch.no_grad():
            update_dict = {
                online_clients_list[i]:
                    parameters_to_vector(nets_list[i].parameters()) - parameters_to_vector(global_net.parameters())
                for i in range(len(nets_list))
            }

        freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)
        lr_vector = self.compute_robustLR(update_dict)
        # cur_global_params = parameters_to_vector(global_net.parameters())
        cur_global_params = parameters_to_vector(global_net.parameters())
        for i in online_clients_list:
            local_param = (cur_global_params + lr_vector * update_dict[i]).float()
            vector_to_parameters(local_param, nets_list[i].parameters())
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=global_net, freq=freq,
                       except_part=[], global_only=False)

        # aggregated_updates = self.agg_avg(update_dict, freq, online_clients_list)
        #
        #
        # new_global_params = (cur_global_params + lr_vector * aggregated_updates).float()
        # # new_global_params = (cur_global_params + aggregated_updates).float()
        # global_params = global_net.parameters()
        # vector_to_parameters(new_global_params, global_net.parameters())
        # new_global_params = global_net.parameters()
        # for net in nets_list:
        #     net.load_state_dict(global_net.state_dict(), strict=False)
