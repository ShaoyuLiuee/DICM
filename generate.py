import os
import json
import math
import uuid
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from ddpm_torch import *
from ddim import DDIM, get_selection_schedule
from torchvision import transforms, datasets as tvds
from argparse import ArgumentParser
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
#os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import os
import pandas as pd
import torch.nn.functional as F
import functools
from torch.utils.data import Subset
#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"  # 有几块GPU写多少

def progress_monitor(total, counter):
    pbar = tqdm(total=total)
    while pbar.n < total:
        if pbar.n < counter.value:  # non-blocking intended
            pbar.update(counter.value - pbar.n)
        time.sleep(0.1)


def generate(rank, args, counter=0):
    assert isinstance(counter, (Synchronized, int))

    is_leader = rank == 0
    dataset = args.dataset
    in_channels = DATASET_INFO[dataset]["channels"]
    image_res = DATASET_INFO[dataset]["resolution"][0]
    input_shape = (in_channels, image_res, image_res)

    config_dir = args.config_dir
    with open(os.path.join(config_dir, dataset + ".json")) as f:
        configs = json.load(f)

    diffusion_kwargs = configs["diffusion"]
    beta_schedule = diffusion_kwargs.pop("beta_schedule")
    beta_start = diffusion_kwargs.pop("beta_start")
    beta_end = diffusion_kwargs.pop("beta_end")
    num_diffusion_timesteps = diffusion_kwargs.pop("timesteps")
    betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)

    use_ddim = args.use_ddim
    if use_ddim:
        diffusion_kwargs["model_var_type"] = "fixed-small"
        skip_schedule = args.skip_schedule
        eta = args.eta
        subseq_size = args.subseq_size
        subsequence = get_selection_schedule(skip_schedule, size=subseq_size, timesteps=num_diffusion_timesteps)
        diffusion = DDIM(betas, **diffusion_kwargs, eta=eta, subsequence=subsequence)
    else:
        diffusion = GaussianDiffusion(betas, **diffusion_kwargs)

    device = torch.device(f"cuda:{rank}" if args.num_gpus > 1 else args.device)
    block_size = configs["model"].pop("block_size", 1)
    model = UNet(out_channels=10, **configs["model"])
    if block_size > 1:
        pre_transform = torch.nn.PixelUnshuffle(block_size)  # space-to-depth
        post_transform = torch.nn.PixelShuffle(block_size)  # depth-to-space
        model = ModelWrapper(model, pre_transform, post_transform)
    model.to(device)
    chkpt_dir = args.chkpt_dir
    chkpt_path = args.chkpt_path or os.path.join(chkpt_dir, f"ddpm_{dataset}.pt")
    folder_name = os.path.basename(chkpt_path)[:-3]  # truncated at file extension
    use_ema = args.use_ema
    if use_ema:
        state_dict = torch.load(chkpt_path, map_location=device)["ema"]["shadow"]
    else:
        state_dict = torch.load(chkpt_path, map_location=device)["model"]
    for k in list(state_dict.keys()):
        if k.startswith("module."):  # state_dict of DDP
            state_dict[k.split(".", maxsplit=1)[1]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    del state_dict
    model.eval()
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad_(False)

    folder_name = folder_name + args.suffix
    save_dir = os.path.join(args.save_dir, folder_name)
    if is_leader and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    local_total_size = args.local_total_size
    batch_size = args.batch_size
    if args.world_size > 1:
        if rank < args.total_size % args.world_size:
            local_total_size += 1
    local_num_batches = math.ceil(local_total_size / batch_size)
    shape = (batch_size, ) + input_shape
    # 定义CIFAR-10类别索引到文本标签的映射字典
    # class_index_to_label = {
    #     0: "airplane",
    #     1: "automobile",
    #     2: "bird",
    #     3: "cat",
    #     4: "deer",
    #     5: "dog",
    #     6: "frog",
    #     7: "horse",
    #     8: "ship",
    #     9: "truck"
    # }
    # def save_image(arr):
    #     with Image.fromarray(arr, mode="RGB") as im:
    #         im.save(f"{save_dir}/{uuid.uuid4()}.png")

    
    # if torch.backends.cudnn.is_available():  # noqa
    #     torch.backends.cudnn.benchmark = True  # noqa

    # pbar = None
    # if isinstance(counter, int):
    #     pbar = tqdm(total=local_num_batches)
    # # # 直接将标签和图像同时采样 看是否是一一对应的关系
    # with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
    #     for i in range(local_num_batches):
    #         if i == local_num_batches - 1:
    #             shape = (local_total_size - i * batch_size, 3, image_res, image_res)
    #         x, cf = diffusion.p_sample(model, shape=shape, device=device)
    #         # # 第二种可以尝试的方法
    #         probabilities = torch.softmax(cf, dim=1) #把张量中的每一个元素转化成非负值 并且所有元素之和为1，通常用于将原始分数转换成概率分布      
    #         mean_values = probabilities.mean(dim=(2, 3))     
    #         class_labels = torch.argmax(mean_values, dim=1).to('cuda:0') #[128]
    #         # 检查最大值索引范围是否在[0, num_classes-1]内
    #         num_classes = 10
    #         if torch.any(class_labels >= num_classes):
    #             raise ValueError("最大值索引超出类别范围！")            
    #         x = (x.cpu() * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    #         pool.map(lambda img: save_image(img[0], img[1]), zip(list(x), class_labels))
    #         if isinstance(counter, Synchronized):
    #             with counter.get_lock():
    #                 counter.value += 1
    #         else:
    #             pbar.update(1)
    # 定义CIFAR-10类别索引到文本标签的映射字典
    class_index_to_label = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }
    pbar = None
    if isinstance(counter, int):
        pbar = tqdm(total=local_num_batches)    
    # 反向inpainting
        #反向inpaint生成图片，计算FID
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for i in range(local_num_batches):
            if i == local_num_batches - 1:
                shape = (local_total_size - i * batch_size, 13, image_res, image_res)
                random_labels = torch.randint(low=0, high=10, size=(local_total_size - i * batch_size,)) 
                # 将随机标签转换为One-Hot向量
                one_hot_labels = F.one_hot(random_labels, num_classes=10).float()
                # 扩展One-Hot向量的维度，使其变为[batch_size, 10, 1, 1]
                one_hot_labels = one_hot_labels.view(local_total_size - i * batch_size, 10, 1, 1)
                # # 将one-hot中的0替换为0.4，将1替换为0.6
                one_hot_labels[one_hot_labels == 0] = 0.49
                one_hot_labels[one_hot_labels == 1] = 0.51                     
            else:
                random_labels = torch.randint(low=0, high=10, size=(batch_size,))
                # 将随机标签转换为One-Hot向量
                one_hot_labels = F.one_hot(random_labels, num_classes=10).float()
                # 扩展One-Hot向量的维度，使其变为[batch_size, 10, 1, 1]
                one_hot_labels = one_hot_labels.view(batch_size, 10, 1, 1)
                # 将one-hot中的0替换为0.4，将1替换为0.6
                one_hot_labels[one_hot_labels == 0] = 0.49
                one_hot_labels[one_hot_labels == 1] = 0.51      
            # 扩展One-Hot向量的维度，使其变为[batch_size, 10, 32, 32]
            one_hot_labels = one_hot_labels.expand(one_hot_labels.shape[0], 10, 32, 32)
            images = torch.randn(one_hot_labels.shape[0], 3, 32, 32)
            images = torch.cat((images, one_hot_labels), dim=1)
            gt_keep_mask = torch.ones_like(images)
            gt_keep_mask[:, :3, :, :] = 0            
            x = diffusion.p_sample(model, shape=shape, device=device, images=images, gt_keep_mask=gt_keep_mask, noise=torch.randn(shape, device=device))
            x_image = x[:, :3, :, :]
            # 归一化使用的均值和标准差
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to("cuda:3")
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to("cuda:3")  
            x_image = (x_image * 0.5 + 0.5) * 255
            x_image = x_image.round().clamp(0, 255).to(torch.uint8)
            x_image = x_image.permute(0, 2, 3, 1).cpu().numpy() 
            for j in range(one_hot_labels.shape[0]):
                # 获取单张图片和对应的类别标签
                single_image = x_image[j]
                single_class_label = random_labels[j].item()
                class_text_label = class_index_to_label[single_class_label]
                # 将张量转换为PIL图像
                image_pil = transforms.ToPILImage()(single_image)  # 转换为PIL图像对象
                filename = f"{save_dir}/{class_text_label}_{uuid.uuid4()}.png"
                image_pil.save(filename)

            if isinstance(counter, Synchronized):
                with counter.get_lock():
                    counter.value += 1
            else:
                pbar.update(1)
    # # CIFAR-10数据集文件路径
    # cifar10_dataset_folder_path = '/home/shaoyu/ddpm-torch-new-train/datasets/cifar10_test'

    # # 定义预处理步骤
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),  # 随机水平翻转        
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])
    # # 创建ImageFolder数据集
    # test_dataset = ImageFolder(root=cifar10_dataset_folder_path, transform=transform)
    # # 创建DataLoader
    # batch_size = 128  # 你可以根据需要调整batch_size
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # # 多次采样投票，最后确定最终类别
    # # 用生成的标签和真实标签计算准确度
    # correct = 0
    # total = 0    
    # for data in test_loader:
    #     start_time = time.time()  # 记录开始时间
    #     images, labels = data
    #     print(labels)
    #     # 将images传入你的生成标签的模型，得到生成的标签pred_labels
    #     # shape = (images.shape[0], 13, image_res, image_res)      
    #     shape = (images.shape[0], 13, image_res, image_res)
    #     one_hot_label = torch.nn.functional.one_hot(labels.clone().detach(), num_classes=10).float()
    #     one_hot_label = one_hot_label.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, image_res, image_res)
    #     images = torch.cat((images, one_hot_label), dim=1)
    #     gt_keep_mask = torch.ones_like(images)
    #     gt_keep_mask[:, 3:, :, :] = 0
    #     # 初始化计数器，用于存储每个样本的出现次数
    #     class_counts = torch.zeros((images.shape[0], 10), device='cuda:0')        
    #     #重复进行十次采样
    #     for _ in range(5):
    #         x = diffusion.p_sample(model, shape=shape, device=device, images=images, gt_keep_mask=gt_keep_mask, noise=torch.randn(shape, device=device))
    #         x_image, x_label = x[:, :3, :, :], x[:, 3:, :, :]
    #         # # 第二种可以尝试的方法
    #         probabilities = torch.softmax(x_label, dim=1) #把张量中的每一个元素转化成非负值 并且所有元素之和为1，通常用于将原始分数转换成概率分布      
    #         mean_values = probabilities.mean(dim=(2, 3))     
    #         class_labels = torch.argmax(mean_values, dim=1).to('cuda:0') #[128]
    #         # 检查最大值索引范围是否在[0, num_classes-1]内
    #         num_classes = 10
    #         if torch.any(class_labels >= num_classes):
    #             raise ValueError("最大值索引超出类别范围！")

    #         for i in range(images.shape[0]):
    #             class_idx = class_labels[i].item()
    #             class_counts[i, class_idx] += 1
    #     # 选择出现次数最多的类别作为最终的预测结果
    #     final_class_indices = torch.argmax(class_counts, dim=1)
    #     print(final_class_indices)
    #     labels = labels.to('cuda:0')
    #     #计算准确度
    #     total += labels.size(0)
    #     correct += (final_class_indices == labels).float()
    #     accuracy = correct.sum().item() / total
    #     print(total)
    #     print(f'Accuracy: {accuracy * 100:.2f}%')
    #     end_time = time.time()  # 记录结束时间
    #     elapsed_time = end_time - start_time  # 计算处理时间
    #     print(f"Batch Processing Time: {elapsed_time:.2f} seconds")  # 输出处理时间
    # def save_image(arr, label):
    #     class_name = class_index_to_label[label.item()]
    #     with Image.fromarray(arr, mode="RGB") as im:
    #         filename = f"{save_dir}/{uuid.uuid4()}_label_{class_name}.png"
    #         im.save(filename)

    # #采样生成图片，计算FID
    # with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
    #     for i in range(local_num_batches):
    #         if i == local_num_batches - 1:
    #             shape = (local_total_size - i * batch_size, 13, 3, 3)
    #         mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    #         std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)  
    #         x = diffusion.p_sample(model, shape=shape, device=device).cpu()
    #         x_image, x_label = x[:, :3, :, :], x[:, 3:, :, :]
    #         # # 第二种可以尝试的方法
    #         probabilities = torch.softmax(x_label, dim=1) #把张量中的每一个元素转化成非负值 并且所有元素之和为1，通常用于将原始分数转换成概率分布      
    #         mean_values = probabilities.mean(dim=(2, 3))     
    #         class_labels = torch.argmax(mean_values, dim=1).to('cuda:0') #[128]
    #         x_image = (x_image * std + mean) * 255        
    #         x_image = x_image.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    #         for j in range(shape[0]):
    #             # 获取单张图片和对应的类别标签
    #             single_image = x_image[j]
    #             single_class_label = class_labels[j].item()
    #             class_text_label = class_index_to_label[single_class_label]
    #             # 将张量转换为PIL图像
    #             image_pil = transforms.ToPILImage()(single_image)  # 转换为PIL图像对象
    #             filename = f"{save_dir}/{class_text_label}_{uuid.uuid4()}.png"
    #             image_pil.save(filename)
    #         if isinstance(counter, Synchronized):
    #             with counter.get_lock():
    #                 counter.value += 1
    #         else:
    #             pbar.update(1)

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10","stl", "celeba", "celebahq"], default="cifar10")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--total-size", default=10000, type=int)
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-path", default="", type=str)
    parser.add_argument("--save-dir", default="./images/repaint/convcifar10_0.49", type=str)
    parser.add_argument("--device", default="cuda:3", type=str)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--use-ddim", action="store_true")
    parser.add_argument("--eta", default=0., type=float)
    parser.add_argument("--skip-schedule", default="linear", type=str)
    parser.add_argument("--subseq-size", default=50, type=int)
    parser.add_argument("--suffix", default="", type=str)
    parser.add_argument("--max-workers", default=8, type=int)
    parser.add_argument("--num-gpus", default=1, type=int)

    args = parser.parse_args()

    world_size = args.world_size = args.num_gpus or 1
    local_total_size = args.local_total_size = args.total_size // world_size
    batch_size = args.batch_size
    remainder = args.total_size % world_size
    num_batches = math.ceil((local_total_size + 1) / batch_size) * remainder
    num_batches += math.ceil(local_total_size / batch_size) * (world_size - remainder)
    args.num_batches = num_batches

    if world_size > 1:
        mp.set_start_method("spawn")
        counter = mp.Value("i", 0)
        mp.Process(target=progress_monitor, args=(num_batches, counter), daemon=True).start()
        mp.spawn(generate, args=(args, counter), nprocs=world_size)
    else:
        generate(0, args)


if __name__ == "__main__":
    main()
