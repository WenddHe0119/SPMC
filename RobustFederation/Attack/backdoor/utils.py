import copy
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from functools import partial

def base_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        target = cfg.attack.backdoor.backdoor_label
        for pos_index in range(0, len(cfg.attack.backdoor.trigger_position)):
            pos = cfg.attack.backdoor.trigger_position[pos_index]
            img[pos[0]][pos[1]][pos[2]] = cfg.attack.backdoor.trigger_value[pos_index]
    return img, target


def semantic_backdoor(cfg, img, target, noise_data_rate):
    if torch.rand(1) < noise_data_rate:
        if target == cfg.attack.backdoor.semantic_backdoor_label:
            target = cfg.attack.backdoor.backdoor_label
            img = img + torch.randn(img.size()) * 0.05



def backdoor_attack(args, cfg, client_type, private_dataset, is_train):
    noise_data_rate = cfg.attack.noise_data_rate if is_train else 1.0
    clean_dataset = copy.deepcopy(private_dataset)
    if is_train:
        dataset = copy.deepcopy(private_dataset.train_loaders[0].dataset)

        all_targets = []
        all_imgs = []

        for i in range(len(dataset)):
            img, target = dataset.__getitem__(i)
            if cfg.attack.backdoor.evils == 'base_backdoor':
                img, target = base_backdoor(cfg, (img), (target), noise_data_rate)

            if cfg.attack.backdoor.evils == 'semantic_backdoor':
                img, target = semantic_backdoor(cfg, (img), (target), noise_data_rate)

            all_targets.append(target)
            all_imgs.append(img.numpy())

        new_dataset = BackdoorDataset(all_imgs, all_targets)

        for client_index in tqdm(range(cfg.DATASET.parti_num), desc="Processing Clients"):
            if not client_type[client_index]:
                train_sampler = private_dataset.train_loaders[client_index].batch_sampler.sampler

                if args.task == 'label_skew':
                    private_dataset.train_loaders[client_index] = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch,
                                                                             sampler=train_sampler, num_workers=4, drop_last=True)

        for client_index in range(cfg.DATASET.parti_num):
            bad_imgs = []
            bad_targets = []
            good_imgs = []
            good_targets = []
            if not client_type[client_index]:
                # 遍历 DataLoader 并逐个访问 img 和 target
                for batch in private_dataset.train_loaders[client_index]:
                    images, targets = batch
                    if images.is_cuda:
                        images = images.cpu()
                        targets = targets.cpu()
                    for img, target in zip(images, targets):
                        bad_imgs.append(img.numpy())
                        bad_targets.append(target.item())
                for batch in clean_dataset.train_loaders[client_index]:
                    images, targets = batch
                    if images.is_cuda:
                        images = images.cpu()
                        targets = targets.cpu()
                    for img, target in zip(images, targets):
                        good_imgs.append(img.numpy())
                        good_targets.append(target.item())
        parti_dataset = BackdoorDataset(bad_imgs, bad_targets)
        good_dataset = BackdoorDataset(good_imgs, good_targets)
        private_dataset.backdoor_train_loader = DataLoader(parti_dataset, batch_size=cfg.OPTIMIZER.local_train_batch,
                                                           num_workers=4)
        private_dataset.good_train_loader = DataLoader(good_dataset, batch_size=cfg.OPTIMIZER.local_train_batch,
                                                           num_workers=4)



    else:
        if args.task == 'label_skew':
            dataset = copy.deepcopy(private_dataset.test_loader.dataset)

            all_targets = []
            all_imgs = []

            for i in range(len(dataset)):
                img, target = dataset.__getitem__(i)
                if cfg.attack.backdoor.evils == 'base_backdoor':
                    img, target = base_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), 1.0)

                    all_targets.append(target)
                    all_imgs.append(img.numpy())
                elif cfg.attack.backdoor.evils == 'semantic_backdoor':
                    if target == cfg.attack.backdoor.semantic_backdoor_label:
                        img, target = semantic_backdoor(cfg, copy.deepcopy(img), copy.deepcopy(target), 1.0)
                        all_targets.append(target)
                        all_imgs.append(img.numpy())
            new_dataset = BackdoorDataset(all_imgs, all_targets)
            private_dataset.backdoor_test_loader = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch, num_workers=4)

def pgd_attack(model, images, labels, epsilon=4. / 255., alpha=4. / 255., num_iter=1):
    adv_images = images.clone().detach() + torch.zeros_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=0, max=1)

    # 保证 adv_images 在和模型一样的设备上
    device = next(model.parameters()).device
    adv_images = adv_images.to(device)
    images = images.to(device)
    labels = labels.to(device)

    for _ in range(num_iter):
        torch.cuda.empty_cache()
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_images = adv_images + alpha * torch.sign(adv_images.grad)
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + eta, min=0, max=1)
        adv_images = adv_images.detach().clone()
        torch.cuda.empty_cache()

    return adv_images.detach().clone()

def Badpfl(cfg, net_list, global_model, client_type, private_dataset, clean_dataset,target_label=2, poison_ratio=0.3):
    class Autoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
                torch.nn.ReLU(),
            )
            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
                torch.nn.Sigmoid()
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    trigger_gen = Autoencoder()
    gen_optimizer = torch.optim.Adam(trigger_gen.parameters(), lr=1e-2)
    loss_func = torch.nn.CrossEntropyLoss()

    def trigger_gen_trainer(client, loacl_dataset, target_label):
        client.eval()
        for batch_idx, (clean_data, clean_label) in enumerate(loacl_dataset):
            device = next(client.parameters()).device
            torch.cuda.empty_cache()
            gen_optimizer.zero_grad()
            adv_imgs = pgd_attack(client, clean_data, clean_label)
            gen_trigger = trigger_gen(clean_data) / 255. * 4.
            clean_data = clean_data.to(device)
            clean_label = clean_label.to(device)
            gen_trigger = gen_trigger.to(device)
            pred = client(adv_imgs + gen_trigger).to(device)
            loss = loss_func(pred, torch.full([clean_label.size(0)], target_label, device=clean_label.device).to(
                torch.long))
            loss.backward()
            gen_optimizer.step()
            torch.cuda.empty_cache()

    def use_badpfl(net_list, index, dataset_loader, target_label, poison_ratio):
        trigger_gen_trainer(net_list[index], dataset_loader, target_label)
        # all_img = []
        # all_label = []
        # images, labels = dataset_loader
        # for i in range(len(dataset_loader)):
        #     img, label = dataset_loader.__getitem__(i)
        for batch_idx, (images, labels) in enumerate(dataset_loader):
            device = next(net_list[index].parameters()).device
            poison_mask = torch.rand(labels.size(0), device=labels.device) <= poison_ratio
            if poison_mask.sum().item() == 0:
                return
            else:
                poison_data, poison_label = images.clone(), torch.full([labels.size(0)], target_label,
                                                                       device=labels.device)
            poison_data = pgd_attack(net_list[index], poison_data, labels).detach().clone()
            gen_trigger = trigger_gen(images) / 255. * 4.
            images = images.to(device)
            labels = labels.to(device)
            gen_trigger = gen_trigger.to(device)
            poison_mask = poison_mask.to(device)
            poison_label = poison_label.to(device)
            poison_data = poison_mask.view(-1, 1, 1, 1).float() * (poison_data + gen_trigger) + (
                ~poison_mask.view(-1, 1, 1, 1)).float() * images
            poison_label = poison_mask.float() * poison_label + (~poison_mask).float() * labels
            images = poison_data.cpu()
            labels = poison_label.cpu()

    test_all_img, test_all_label = [], []
    for index, client in enumerate(net_list):
        if not client_type[index]:
            # train_sampler = clean_dataset.train_loaders[index].batch_sampler.sampler
            use_badpfl(net_list, index, clean_dataset.train_loaders[index], target_label, poison_ratio)
            # new_dataset = BackdoorDataset(train_img, train_label)
            # private_dataset.train_loaders[index] = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch,
            #                                                          sampler=train_sampler, num_workers=4,
            #                                                          drop_last=True)
            # test_img, test_label = use_badpfl(net_list, index, clean_dataset.test_loader, target_label, 1.0)
            # test_all_img.append(test_img)
            # test_all_label.append(test_label)
    use_badpfl(net_list, index, clean_dataset.test_loader, target_label, 1.0)
    # new_dataset = BackdoorDataset(test_all_img, test_all_label)
    # private_dataset.backdoor_test_loader = DataLoader(new_dataset, batch_size=cfg.OPTIMIZER.local_train_batch, num_workers=4)
    private_dataset = copy.deepcopy(clean_dataset)


class BackdoorDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = np.array(data)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
