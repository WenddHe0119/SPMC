import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np
from Backbones import get_private_backbones
from utils.cfg import CFG as cfg, simplify_cfg,show_cfg
import datetime
import socket
import uuid
from argparse import ArgumentParser
from Optims import Fed_Optim_NAMES, get_fed_method
from Backbones import get_private_backbones
from Server import get_server_method,Server_NAME
import argparse
from Backbones.SimpleCNN import SimpleCNN, SimpleCNN_sr
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Here is the code ：

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import utils


def parse_args():
    parser = ArgumentParser(description='Federated Learning', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    '''
    Task: label_skew domain_skew
    '''
    parser.add_argument('--task', type=str, default='label_skew')
    '''
    label_skew:   fl_cifar10 fl_cifar100 fl_mnist fl_usps fl_fashionmnist fl_tinyimagenet
    '''
    parser.add_argument('--dataset', type=str, default='fl_cifar10',
                        help='Which scenario to perform experiments on.')

    '''
    Attack: byzantine backdoor None
    '''
    parser.add_argument('--attack_type', type=str, default='backdoor')

    '''
    Federated Method:  fedavg  fedprox  PrevAbsFedFish
    '''
    parser.add_argument('--optim', type=str, default='FedAvG',
                        help='Federated Method name.', choices=Fed_Optim_NAMES)
    # FedFish FedAvG FedProx
    parser.add_argument('--rand_domain_select', type=bool, default=False, help='The Local Domain Selection')
    '''
    Aggregations Strategy Hyper-Parameter
    '''
    parser.add_argument('--server', type=str, default='FTSAM', choices=Server_NAME, help='The Option for averaging strategy')
    # MultiKrum Weight Equal FLTrust ExpNegFishDiff FTSAM
    parser.add_argument('--seed', type=int, default=0, help='The random seed.')

    parser.add_argument('--csv_log', action='store_true', default=False, help='Enable csv logging')
    parser.add_argument('--csv_name', type=str, default=None, help='Predefine the csv name')
    parser.add_argument('--save_checkpoint', action='store_true', default=False)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def save_tensor_as_image(tensor, path, original_shape):
    """
    将一个给定的 tensor 转换为图像并恢复到原始形状，然后保存。

    参数:
        tensor (torch.Tensor): 形状为 (C, H, W) 的张量。
        original_shape (tuple): 原始图像的形状 (H, W, C)。
        path (str): 图像保存的路径。
    """
    # 如果张量在 GPU 上，先转移到 CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # 如果张量的值在 [0, 1] 范围内，则需要乘以 255 转换为 [0, 255] 范围内的整数值
    if tensor.max() <= 1:
        tensor = tensor * 255

    # 转换张量的数据类型为 uint8
    tensor = tensor.byte()

    # 选择前3个通道
    tensor = tensor[:3]

    # 调整张量的维度顺序从 (C, H, W) 到 (H, W, C)，这一步对于 PIL.Image 是必需的
    # tensor = tensor.permute(1, 2, 0)

    # 使用 torchvision.transforms.ToPILImage 将张量转换为 PIL 图像
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)

    # 恢复到原始形状
    original_height, original_width, _ = original_shape
    image = image.resize((original_width, original_height), Image.Resampling.LANCZOS)

    # 保存图像
    image.save(path)

def base_backdoor(cfg, img):
    for pos_index in range(0, len(cfg.attack.backdoor.trigger_position)):
        pos = cfg.attack.backdoor.trigger_position[pos_index]
        img[pos[0]][pos[1]][pos[2]] = cfg.attack.backdoor.trigger_value[pos_index]
    return img


def main():
    trigger = True
    args = parse_args()
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    particial_cfg = simplify_cfg(args, cfg)

    model_path = 'data/label_skew/base_backdoor/0.3/fl_cifar10/0.5/SPMD/AlignedGradient/lamda1.0_N10/model.pth'
    # model_path = 'data/label_skew/base_backdoor/0.3/fl_cifar10/0.5/DnC/FedAvG/label_2_N10/model.pth'
    model = SimpleCNN(particial_cfg)  # 假设cfg已正确配置
    model.load_state_dict(torch.load(model_path))
    model.eval()
    target_layers = [model.feats.conv2]  # 指定目标层

    data_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Prepare image
    img_path = "Pic1.jpg"
    # img_path = "Pic2.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    if '1' in img_path:
        cate = 'fly'
    else:
        cate = 'cat'
    if 'SPMD' in model_path:
        method = 'SPMC'
    else:
        method = 'DnC'
    if trigger:
        flag = 'Y'
    else:
        flag = 'N'
    img = Image.open(img_path).convert('RGB')
    original_shape = np.array(img).shape
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (32, 32))
    img_tensor = data_transform(img)
    if trigger:
        img_tensor = base_backdoor(cfg, img_tensor)
        # tmp_tensor = img_tensor.permute(2, 1, 0)
        # save_tensor_as_image(img_tensor, 'fly_trigger_origin.png', original_shape)
        # utils.save_image(img_tensor, 'fly_trigger_origin_tensor.png')
        # zhuan
        img_tensor_np = img_tensor.numpy()

        # 逆标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_tensor_np = (img_tensor_np * std[:, None, None] + mean[:, None, None])

        # 将数据类型转换为 uint8
        img_tensor_np = (img_tensor_np * 255).astype(np.uint8)

        # 调整尺寸回到原始尺寸
        img_resized = cv2.resize(img_tensor_np.transpose(1, 2, 0), (original_shape[1], original_shape[0]))

        # 使用 PIL 库保存图片
        img_pil = Image.fromarray(img_resized)
        img_pil.save('fly_trigger_origin.png')
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # Grad CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    # targets = [ClassifierOutputTarget(281)]     # cat
    # targets = [ClassifierOutputTarget(254)]  # dog
    targets = None

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.,
                                      grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    # plt.savefig(f'output_{cate}_{method}_{flag}.jpg', format='jpg')
    plt.show()



if __name__ == '__main__':
    main()





# def load_model(path,cfg):
#     model = SimpleCNN(cfg)  # 这里以ResNet18为例，你需要替换为你实际使用的模型结构
#     model.load_state_dict(torch.load(path))
#     model.eval()
#     return model
#
#
#
#
#
# def preprocess_image(img_path, target_size=(32, 32)):
#     transform = transforms.Compose([
#         transforms.Resize(target_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     img = Image.open(img_path).convert('RGB')
#     img_tensor = transform(img)
#     print(img_tensor.shape)
#     return img_tensor # 添加批次维度
#
#
#
#
#
# class FeatureExtractor:
#     """提取特征图并注册梯度"""
#     def __init__(self, model, target_layers):
#         self.model = model
#         self.target_layers = target_layers
#         self.gradients = []
#
#     def save_gradient(self, grad):
#         self.gradients.append(grad)
#
#     def __call__(self, x):
#         outputs = []
#         self.gradients = []
#
#         x = self.model(x)
#
#         # # 遍历feats模块
#         # x = self.model.feats.conv1(x)
#         # print(f"conv1 output shape: {x.shape}")
#         # x = self.model.feats.relu(x)
#         # x = self.model.feats.pool(x)
#         # print(f"pool1 output shape: {x.shape}")
#         # x = self.model.feats.conv2(x)
#         # print(f"conv2 output shape: {x.shape}")
#         # x = self.model.feats.relu(x)
#         # x = self.model.feats.pool(x)
#         # print(f"pool2 output shape: {x.shape}")
#         #
#         # if 'conv2' in self.target_layers:
#         #     x.register_hook(self.save_gradient)
#         #     outputs += [x]
#         #
#         # # 展平特征图
#         # x = x.view(x.size(0), -1)
#         # print(f"flattened output shape: {x.shape}")
#         #
#         # x = self.model.feats.fc1(x)
#         # print(f"fc1 output shape: {x.shape}")
#         # x = self.model.feats.relu(x)
#         # x = self.model.feats.fc2(x)
#         # print(f"fc2 output shape: {x.shape}")
#         #
#         # x = self.model.l1(x)
#         # print(f"l1 output shape: {x.shape}")
#         # x = F.relu(x)
#         # x = self.model.l2(x)
#         # print(f"l2 output shape: {x.shape}")
#
#         return outputs, x
#
# class ModelOutputs:
#     """提取特征图和梯度"""
#     def __init__(self, model, target_layers):
#         self.model = model
#         self.feature_extractor = FeatureExtractor(self.model, target_layers)
#
#     def get_gradients(self):
#         return self.feature_extractor.gradients
#
#     def __call__(self, x):
#         target_activations, output = self.feature_extractor(x)
#         return target_activations, output
#
# # class GradCAM:
# #     def __init__(self, model, target_layer_names, use_cuda):
# #         self.model = model
# #         self.model.eval()
# #         self.cuda = use_cuda
# #         if self.cuda:
# #             self.model = model.cuda()
# #
# #         self.extractor = ModelOutputs(self.model, target_layer_names)
# #
# #     def forward(self, input_img):
# #         return self.__call__(input_img)
# #
# #     def __call__(self, input_img):
# #         if self.cuda:
# #             input_img = input_img.cuda()
# #
# #         features, output = self.extractor(input_img)
# #
# #         one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
# #         one_hot_output[0][torch.argmax(output)] = 1
# #         one_hot_output = Variable(one_hot_output, requires_grad=True)
# #
# #         if self.cuda:
# #             one_hot_output = one_hot_output.cuda()
# #
# #         output.backward(gradient=one_hot_output)
# #
# #         grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
# #
# #         target = features[-1]
# #         target = target.cpu().data.numpy()[0, :]
# #
# #         weights = np.mean(grads_val, axis=(2, 3))[0, :]
# #         cam = np.zeros(target.shape[1:], dtype=np.float32)
# #
# #         for i, w in enumerate(weights):
# #             cam += w * target[i, :, :]
# #
# #         cam = np.maximum(cam, 0)
# #         cam = cv2.resize(cam, (224, 224))  # 根据输入图片大小调整
# #         cam = cam - np.min(cam)
# #         cam = cam / np.max(cam)
# #         return cam
#
# args = parse_args()
# args.conf_jobnum = str(uuid.uuid4())
# args.conf_timestamp = str(datetime.datetime.now())
# args.conf_host = socket.gethostname()
# particial_cfg = simplify_cfg(args, cfg)
#
# model_path = 'data/label_skew/base_backdoor/0.3/fl_cifar10/0.5/SPMD/AlignedGradient/GradCam/model_A.pth'
# model = SimpleCNN(particial_cfg)  # 假设cfg已正确配置
# model.load_state_dict(torch.load(model_path))
# model.eval()
#
#
# # model_A = load_model('data/label_skew/base_backdoor/0.3/fl_cifar10/0.5/SPMD/AlignedGradient/GradCam/model_A.pth',particial_cfg)
# # model_B = load_model('data/label_skew/base_backdoor/0.3/fl_cifar10/0.5/SPMD/AlignedGradient/GradCam/model_B.pth',particial_cfg)
#
#
# image_path = 'data/label_skew/base_backdoor/0.3/fl_cifar10/0.5/SPMD/AlignedGradient/GradCam/img/1.jpg'
# input_tensor = preprocess_image(image_path)
#
# # grad_cam = GradCAM(model=model_A, target_layer_names=['conv2'], use_cuda=False)  # 对于ResNet，最后一层卷积通常是layer4
# # mask = grad_cam(img_tensor)
# #
# # # 将热力图与原始图片叠加
# # img = cv2.imread('data/label_skew/base_backdoor/0.3/fl_cifar10/0.5/SPMD/AlignedGradient/GradCam/img/1.jpg', 1)
# # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
# # img_with_heatmap = np.float32(heatmap) + np.float32(img)
# # img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
# # cv2.imwrite('output_A.png', np.uint8(255 * img_with_heatmap))
# #
# # # 对模型B重复上述过程
# # grad_cam_b = GradCAM(model=model_B, target_layer_names=['conv2'], use_cuda=False)
# # mask_b = grad_cam_b(img_tensor)
# # heatmap_b = cv2.applyColorMap(np.uint8(255 * mask_b), cv2.COLORMAP_JET)
# # img_with_heatmap_b = np.float32(heatmap_b) + np.float32(img)
# # img_with_heatmap_b = img_with_heatmap_b / np.max(img_with_heatmap_b)
# # cv2.imwrite('output_B.png', np.uint8(255 * img_with_heatmap_b))
#
# # 创建GradCAM对象
# target_layers = [model.feats.conv2]  # 指定目标层
# cam = GradCAM(model=model, target_layers=target_layers)
#
# # 获取图像的numpy数组形式
# rgb_img = np.array(Image.open(image_path).convert('RGB')) / 255.0
#
# # 使用GradCAM获取热图
# grayscale_cam = cam(input_tensor=input_tensor, targets=None)
#
# # 将热图与原始图像叠加
# visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#
# # 显示结果
# plt.imshow(visualization)
# plt.axis('off')  # 关闭坐标轴
# plt.show()