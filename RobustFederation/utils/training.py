import copy
from Attack.byzantine.utils import attack_net_para
from Optims.utils.federated_optim import FederatedOptim
from utils.logger import CsvWriter
from torch.utils.data import DataLoader
import torch
import numpy as np
from utils.utils import log_msg
from typing import Tuple
from Attack.backdoor.utils import Badpfl




def cal_top_one_five(net, test_dl, device):
    net.eval()
    correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
    for batch_idx, (images, labels) in enumerate(test_dl):
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)
            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)
    net.train()
    top1acc = round(100 * top1 / total, 2)
    top5acc = round(100 * top5 / total, 2)
    return top1acc, top5acc

def global_in_evaluation(model: FederatedOptim, test_loader: dict, in_domain_list: list):
    in_domain_accs = []
    for in_domain in in_domain_list:
        global_net = model.global_net
        global_net.eval()

        test_domain_dl = test_loader[in_domain]
        top1acc, _ = cal_top_one_five(net=global_net, test_dl=test_domain_dl, device=model.device)
        in_domain_accs.append(top1acc)
        global_net.train()
    mean_in_domain_acc = round(np.mean(in_domain_accs, axis=0), 3)
    return in_domain_accs, mean_in_domain_acc
def fill_blank(net_cls_counts,classes):
    class1 = [i for i in range(classes)]

    for client, dict_i in net_cls_counts.items():
        if len(dict_i.keys()) == 10:
            continue
        else:
            for i in class1:
                if i not in dict_i.keys():
                    dict_i[i] = 0

    return net_cls_counts


def train(fed_method, fed_server, private_dataset, args, cfg, total_clean_dataset) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, cfg)
    with open('./data/a.txt', 'r') as f:
        for line in f.readlines():
            print(line)
    if hasattr(fed_method, 'ini'):
        fed_method.ini()
        fed_server.ini()


    if args.task == 'label_skew':
        mean_in_domain_acc_list = []
        if args.attack_type == 'None':
            contribution_match_degree_list = []
        fed_method.net_cls_counts = fill_blank(private_dataset.net_cls_counts,cfg.DATASET.n_classes)
    elif args.task == 'domain_skew':
        in_domain_accs_dict = {}  # Query-Client Accuracy \bm{\mathcal{A}}}^{u}
        mean_in_domain_acc_list = []  # Cross-Client Accuracy A^U \bm{\mathcal{A}}}^{\mathcal{U}
        performance_variane_list = []
        if args.attack_type == 'None':
            contribution_match_degree_list = []
    if args.attack_type == 'backdoor':
        attack_success_rate = []

    communication_epoch = cfg.DATASET.communication_epoch
    old_net = copy.deepcopy(fed_method.nets_list)
    excepted_models_list = copy.deepcopy(fed_method.nets_list)
    # bad_models = copy.deepcopy(fed_method.nets_list[0])
    for epoch_index in range(communication_epoch):
        if epoch_index == 0:
            # print(fed_method)
            freq = [1/10 for i in range(10)]
            fed_method.global_net = copy.deepcopy(fed_method.nets_list[0])
        fed_method.epoch_index = epoch_index
        print(fed_method.NAME)

        if hasattr(fed_method, 'loc_update'):
            fed_method.val_loader = private_dataset.val_loader
            bad_models = copy.deepcopy(fed_method.nets_list[0])
            if args.attack_name == 'Badpfl':
                Badpfl(cfg, fed_method.nets_list, fed_method.global_net, fed_method.client_type, private_dataset, total_clean_dataset,
                       target_label=cfg.attack.backdoor.backdoor_label,
                       poison_ratio=cfg.attack.noise_data_rate)
            if fed_method.NAME == 'BadAlignv2':
                fed_method.loc_update(total_clean_dataset.train_loaders, freq=freq, epoch=epoch_index,
                                      excepted_models_list=excepted_models_list,
                                      bad_loader=private_dataset.backdoor_train_loader,
                                      good_loader= private_dataset.good_train_loader)
            else:
                fed_method.loc_update(private_dataset.train_loaders,freq=freq, epoch = epoch_index,
                                      excepted_models_list = excepted_models_list, bad_loader = private_dataset.backdoor_train_loader,
                                      good_loader= private_dataset.good_train_loader)


        old_net = copy.deepcopy(fed_method.nets_list)
        if args.attack_type == 'byzantine':
            attack_net_para(args, cfg, fed_method)
        freq = fed_server.server_update(online_clients_list=fed_method.online_clients_list,
                                 priloader_list=private_dataset.train_loaders,
                                 client_domain_list=fed_method.client_domain_list, global_net=fed_method.global_net,
                                 nets_list=fed_method.nets_list, val_loader=private_dataset.val_loader,
                                 epoch_index =epoch_index, fish_diff_dict=fed_method.fish_diff_dict, fish_svd_dict=fed_method.fish_svd_dict, hessian_ma = fed_method.hessian_ma)
        print('--------')
        print(fed_method.online_clients_list)
        print(freq)
        print('--------')
        if fed_method.NAME == 'AlignedGradient':
            for i in range(len(fed_method.nets_list)):
                ecpt_w = {}
                net_para = fed_method.nets_list[i].state_dict()
                ecpt_para = old_net[i].state_dict()
                for k,v in net_para.items():
                    ecpt_w[k] = (v - freq[fed_method.online_clients_list.index(i)]*ecpt_para[k])/(1-freq[fed_method.online_clients_list.index(i)])
                excepted_models_list[i].load_state_dict(ecpt_w, strict=False)

        # if fed_method.NAME == 'BadAlign':
        #     bad_scale = int(fed_method.cfg['DATASET'].parti_num * fed_method.cfg['attack'].bad_client_rate)
        #     bad_w = {}
        #     net_para = old_net[0].state_dict()
        #     for k, v in net_para.items():
        #         bad_w[k] = 0
        #     for i in range(len(fed_method.nets_list)):
        #         if not fed_method.client_type[i]:
        #
        #             # net_para = fed_method.nets_list[i].state_dict()
        #             net_para = old_net[i].state_dict()
        #             for k,v in net_para.items():
        #                 bad_w[k] += 1/bad_scale * v
        #     bad_models.load_state_dict(bad_w, strict=False)



        # Server

        if 'mean_in_domain_acc_list' in locals() and args.task == 'label_skew':
            print("eval mean_in_domain_acc_list")
            top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.test_loader, fed_method.device)
            mean_in_domain_acc_list.append(top1acc)
            if args.csv_name == None:
                print(log_msg(f'The {epoch_index} Epoch: Acc:{top1acc} Optim:{args.optim} Server:{args.server}', "TEST"))
            else:
                print(log_msg(f'The {epoch_index} Epoch: Acc:{top1acc} Optim:{args.optim} Server:{args.server} CSV:{args.csv_name}', "TEST"))

        # if 'in_domain_accs_dict' in locals():
        #     print("eval in_domain_accs_dict")
        #     domain_accs, mean_in_domain_acc = global_in_evaluation(fed_method, private_dataset.test_loader, private_dataset.domain_list)
        #     perf_var = np.var(domain_accs, ddof=0)
        #     performance_variane_list.append(perf_var)
        #     mean_in_domain_acc_list.append(mean_in_domain_acc)
        #
        #     for index, in_domain in enumerate(private_dataset.domain_list):
        #         if in_domain in in_domain_accs_dict:
        #             in_domain_accs_dict[in_domain].append(domain_accs[index])
        #         else:
        #             in_domain_accs_dict[in_domain] = [domain_accs[index]]
        #     print(log_msg(f"The {epoch_index} Epoch: Mean Acc: {mean_in_domain_acc} Method: {args.method} Per Var: {perf_var} ", "TEST"))

        if 'attack_success_rate' in locals():
            top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.backdoor_test_loader, fed_method.device)
            attack_success_rate.append(top1acc)
            if args.csv_name == None:
                print(log_msg(f'The {epoch_index} Epoch: attack success rate:{top1acc} Optim:{args.optim} Server:{args.server}',"ROBUST"))
            else:
                print(log_msg(f'The {epoch_index} Epoch: attack success rate:{top1acc} Optim:{args.optim} Server:{args.server} CSV:{args.csv_name}',"ROBUST"))
        if args.csv_log:
            if args.save_checkpoint:
                torch.save(fed_method.global_net.state_dict(), csv_writer.para_path + '/model.pth')
                print('SAVE!')
    if args.csv_log:
        print("ok for args.csv_log\n")
        if args.task == 'label_skew':

            csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
            print(csv_writer.para_path)
            if args.attack_type == 'None':
                csv_writer.write_acc(contribution_match_degree_list, name='contribution_fairness', mode='MEAN')

        # elif args.task == 'domain_skew':
        #     csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
        #     csv_writer.write_acc(in_domain_accs_dict, name='in_domain', mode='ALL')
        #     csv_writer.write_acc(contribution_match_degree_list, name='contribution_fairness', mode='MEAN')
        #     csv_writer.write_acc(performance_variane_list, name='performance_variance', mode='MEAN')
        if args.attack_type == 'backdoor':
            print("ok for backdoor save\n")
            csv_writer.write_acc(attack_success_rate, name='attack_success_rate', mode='MEAN')

        if args.save_checkpoint:
            torch.save(fed_method.global_net.state_dict(), csv_writer.para_path + '/model_final.pth')
