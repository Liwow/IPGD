import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from cisp import CISP
from cvae import CVAE
import data_utils
from tqdm import tqdm


# cisp lr 0.001 800 cvae 0.0005 100
# torch.cuda.is_available = lambda: False


def loss_fun(BCE, KLD, epoch, w_type=None):
    k = 6
    x = (epoch + 1 / 100) * 2 - 1  # 归一化并平移到(-1, 1]
    if w_type == 'nonlinear':
        weight = (1 / (1 + np.exp(-k * x))) * 1  # 计算sigmoid函数值并缩放到最大权重
    elif w_type == 'linear':
        weight = 0.1 + (1.5 - 0.1) * (epoch + 1 / 100)
    else:
        weight = 1e-6

    return BCE + weight * KLD


def lr_lambda(epoch):
    return 0.9 ** ((epoch + 1) // 15)


def train_one_epoch(model, optimizer, data_loader, device, epoch, tb_writer=None):
    accumulation_steps = 16
    model.train()
    mean_loss = torch.zeros(1).to(device)
    mean_BCE = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader)
    for iteration, mip in enumerate(data_loader):
        x = mip.sols
        if type(model) == CISP:
            accumulation_steps = 32
            logits_per_mip, logits_per_x, key_padding_mask = model(mip, x)
            labels = torch.arange(logits_per_mip.size(0), device=device).long()
            cross_entropy_loss_I = F.cross_entropy(logits_per_mip, labels)
            cross_entropy_loss_X = F.cross_entropy(logits_per_x, labels)
            loss = (cross_entropy_loss_I + cross_entropy_loss_X) / 2

        elif type(model) == CVAE:
            recon_x, mu, logvar, BCE, KLD = model(x, mip)
            loss = loss_fun(BCE, KLD, epoch)
            mean_BCE = (mean_BCE * iteration + BCE.detach()) / (iteration + 1)
        else:
            raise ValueError

        loss.backward()
        mean_loss = (mean_loss * iteration + loss.detach()) / (iteration + 1)  # update mean losses

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        if (iteration + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            data_loader.desc = "[epoch {}] loss {} BCE {}".format(epoch, round(mean_loss.item(), 6), mean_BCE.item())
        if iteration == len(data_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 6))

        if tb_writer is not None:
            tags = ["train_loss", "learning_rate"]
            # tensorboard可视化
            for tag, value in zip(tags, [mean_loss.item(), optimizer.param_groups[0]["lr"]]):
                tb_writer.add_scalars('Train %s' % tag, value, iteration)
    return mean_loss.item(), mean_BCE.item()


@torch.no_grad()
def evaluate(model, data_loader, epoch, device):
    # model.eval()
    total_val_loss = 0
    total_BCE = 0
    with torch.no_grad():
        for iteration, mip in enumerate(data_loader):
            x = mip.sols
            if type(model) == CISP:
                logits_per_mip, logits_per_x, key_padding_mask = model(mip, x)
                labels = torch.arange(logits_per_mip.size(0), device=device).long()
                cross_entropy_loss_I = F.cross_entropy(logits_per_mip, labels)
                cross_entropy_loss_X = F.cross_entropy(logits_per_x, labels)
                loss = (cross_entropy_loss_I + cross_entropy_loss_X) / 2

            elif type(model) == CVAE:
                recon_x, mu, logvar, BCE, KLD = model(x, mip)
                loss = BCE + 1e-6 * KLD
                total_BCE += BCE.item()
            else:
                raise ValueError
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / (iteration + 1)
    avg_BCE = total_BCE / (iteration + 1)
    print(f'Epoch: {epoch}, Validation Loss: {avg_val_loss}, BCE: {avg_BCE}')
    return avg_val_loss


def main(args, logger):
    if args.model == "cisp":
        model = CISP()
    elif args.model == "cvae":
        model = CVAE(embedding=emb)
    else:
        raise ValueError
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.0005)
    epochs = 100
    scheduler = LambdaLR(optimizer, lr_lambda)
    # tb_writer = SummaryWriter(log_dir=args.log_dir)
    logger.logger.info(f'{args.type} start training {args.model}......\n')
    best_val_loss = float('inf')
    for epoch in range(epochs):
        optimizer.zero_grad()
        mean_loss, mean_BCE = train_one_epoch(model, optimizer, train_loader, device, epoch)
        logger.logger.info('%d epoch train mean loss: %.6f mean BCE:%d \n' % (epoch, mean_loss, mean_BCE))

        scheduler.step()

        val_loss = evaluate(model, val_loader, epoch, device)

        if epoch + 1 > 4 and (epoch + 1) % args.save_epoch == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),  # *模型参数
                'optimizer_state_dict': optimizer.state_dict(),  # *优化器参数
                'scheduler_state_dict': scheduler.state_dict(),  # *scheduler
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(model_path, 'checkpoint-%d.pth' % epoch))
            logger.logger.info('save model %d successed......\n' % epoch)

        if epoch + 1 > 4 and val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.logger.info('best model in %d epoch, validation loss: %.6f \n' % (epoch, val_loss))
            checkpoint = {
                'model_state_dict': model.state_dict(),  # *模型参数
                'optimizer_state_dict': optimizer.state_dict(),  # *优化器参数
                'scheduler_state_dict': scheduler.state_dict(),  # *scheduler
                'epoch': epoch,
            }
            torch.save(checkpoint, os.path.join(model_path, f'best_checkpoint.pth'))
            logger.logger.info('save best model successed......\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_epoch', type=float, default=20)
    parser.add_argument('--log_dir', type=str, default='./Logs/summary/pre')
    parser.add_argument('--type', type=str, default="CA")
    parser.add_argument('--model', type=str, default="cvae", help='choose cvae or cisp to encode')
    args = parser.parse_args()
    m = args.model
    emb = True
    if emb and args.model == 'cvae':
        model_path = f'./model/{args.type}/cvae_embedding_pre'
    else:
        model_path = f'./model/{args.type}/{m}_pre'
    if not os.path.isdir(f'./model/{args.type}'):
        os.mkdir(f'./model/{args.type}')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    # dataset = data_utils.MyDataset(ptype)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True,
    #                         collate_fn=data_utils.mip_collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graphSet = data_utils.GraphDataset(args.type)
    total_size = len(graphSet)
    train_size = int(total_size * 0.9)
    val_size = total_size - train_size
    train_set, val_set = random_split(graphSet, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(1998))
    train_loader = torch_geometric.data.DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = torch_geometric.data.DataLoader(val_set, batch_size=4, shuffle=False)

    logger = data_utils.Logger(args, 'pretrain')
    main(args, logger)
