from torch import optim, nn
from tqdm import tqdm

from Server.utils.server_methods import ServerMethod


class Finetuning(ServerMethod):
    NAME = 'Finetuning'

    def __init__(self, args, cfg):
        super(Finetuning, self).__init__(args, cfg)

    def weight_calculate(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        freq = [1 / len(online_clients_list) for _ in range(len(online_clients_list))]
        return freq

    def server_update(self, **kwargs):

        val_loader = kwargs['val_loader']

        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=global_net, freq=freq, except_part=[], global_only=False)

        criterion = nn.CrossEntropyLoss()
        iterator = tqdm(range(1))
        optimizer = optim.SGD(global_net.parameters(), lr=self.cfg.OPTIMIZER.local_train_lr, momentum=0.9,
                              weight_decay=self.cfg.OPTIMIZER.weight_decay)
        for _ in iterator:
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = global_net(images)

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
