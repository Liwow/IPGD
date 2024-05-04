import argparse
import os
from utils import RoundSTE
import sys
from cvae import CVAE
import torch
import torch.nn.functional as F
import torch_geometric
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
import data_utils
import diffusion
from decoder import SolutionDecoder
from cisp import CISP
from tqdm import tqdm


def lr_lambda(epoch):
    return 0.9 ** ((epoch + 1) // 15)


def set_model_to_mode(model, mode):
    if mode == 'eval':
        if isinstance(model, list):
            for m in model:
                m.eval()
        else:
            model.eval()
    else:
        if isinstance(model, list):
            for m in model:
                m.train()
        else:
            model.train()


def Loss_CV(mip, sol_per, ptype):
    # lamda * sum(max( ax - b , 0 )); lamda = n_vars if type != IS else 0
    lamda = mip.n_vars[0]
    sol = sol_per
    if ptype == "IS":
        return torch.tensor(0)
    else:
        A, b, _ = mip.getCoff()
        result = torch.sparse.mm(A, sol.unsqueeze(1))
        Ax_minus_b = result.squeeze(1) - b
        max_violations = torch.clamp(Ax_minus_b, min=0).mean()
        loss = lamda * max_violations
        return loss


def forward_by_vae(mip, x, model, checkpoint=None):
    if checkpoint is not None:
        model.load_state_dict(checkpoint['ddpm_state_dict'])
    n_int_var = mip.n_int_vars
    zi, _ = vae.encode_mip(mip, n_int_var)
    zx, key = vae.encode_solution(x, n_int_var)
    zx_start, loss_ddpm = model(zx, zi, key)
    sols = vae.decoder(zi, zx_start, key)
    loss_CV = Loss_CV(mip, sols, args.type)
    return loss_ddpm, loss_CV, sols


def forward_by_cisp(mip, x, model, checkpoint=None):
    ddpm, decoder = model
    if checkpoint is not None:
        ddpm.load_state_dict(checkpoint['ddpm_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    n_int_var = mip.n_int_vars
    zi, _ = cisp.encode_mip(mip, n_int_var)
    zx, key = cisp.encode_solution(x, n_int_var)

    zx_start, loss_ddpm = ddpm(zx, zi, key)
    sols = decoder(zi, zx_start, key)

    loss_decoder = F.binary_cross_entropy(sols, x.float())
    # sols_round = RoundSTE.apply(sols)
    loss_CV = Loss_CV(mip, sols, args.type)
    return loss_ddpm, loss_CV, loss_decoder, sols


def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch, checkpoint=None, tb_writer=None):
    if checkpoint is not None:
        check = torch.load(checkpoint)
        epoch = check['epoch']
        optimizer.load_state_dict(check['optimizer_state_dict'])
        scheduler.load_state_dict(check['scheduler_state_dict'])
    else:
        check = None
    set_model_to_mode(model, 'train')
    mean_loss = torch.zeros(1).to(device)
    mean_loss_ddpm = torch.zeros(1).to(device)
    mean_loss_CV = torch.zeros(1).to(device)
    accumulation_steps = 8
    data_loader = tqdm(data_loader)
    for iteration, mip in enumerate(data_loader):
        x = mip.sols
        if args.vae:
            loss_ddpm, loss_CV, sols = forward_by_vae(mip, x, model, check)
            loss = loss_ddpm + loss_CV
        else:
            loss_ddpm, loss_CV, loss_decoder, sols = forward_by_cisp(mip, x, model, check)
            # 必须decoder * 10
            loss = loss_ddpm + loss_CV + loss_decoder * 20
        loss.backward()
        mean_loss_ddpm = (mean_loss_ddpm * iteration + loss_ddpm.detach()) / (iteration + 1)
        mean_loss_CV = (mean_loss_CV * iteration + loss_CV.detach()) / (iteration + 1)
        mean_loss = (mean_loss * iteration + loss.detach()) / (iteration + 1)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if (iteration + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            data_loader.desc = "[epoch {}] loss {} loss_CV {} loss_ddpm {}".format(epoch, round(mean_loss.item(), 4),
                                                                                   round(mean_loss_CV.item(), 2),
                                                                                   round(mean_loss_ddpm.item(), 4))
        if iteration == len(data_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 4))

        if tb_writer is not None:
            tags = ["train_loss", "learning_rate"]
            # tensorboard可视化
            for tag, value in zip(tags, [mean_loss.item(), optimizer.param_groups[0]["lr"]]):
                tb_writer.add_scalars('Train %s' % tag, value, iteration)

    scheduler.step()
    return mean_loss.item(), mean_loss_CV.item()


def evaluate(model, data_loader, epoch):
    # set_model_to_mode(model, 'eval')
    total_val_loss = 0
    total_ddpm_loss = 0
    total_CV_loss = 0
    total_obj = 0
    fea = 0
    with torch.no_grad():
        for iteration, mip in enumerate(data_loader):
            x = mip.sols
            if args.vae:
                loss_ddpm, loss_CV, sols = forward_by_vae(mip, x, model)
                loss = loss_ddpm + loss_CV
            else:
                loss_ddpm, loss_CV, loss_decoder, sols = forward_by_cisp(mip, x, model)
                loss = loss_ddpm + loss_CV + loss_decoder

            sols_round = sols.round()
            A, b, c = mip.getCoff()
            obj = sols_round.squeeze() @ c
            violates = torch.max((A @ sols_round).squeeze() - b,
                                 torch.tensor(0)).mean()
            if violates <= 0:
                fea += args.batch
            total_val_loss += loss.item()
            total_ddpm_loss += loss_ddpm.item()
            total_CV_loss += loss_CV.item()
            total_obj += obj.item()

    avg_val_loss = total_val_loss / (iteration + 1)
    avg_ddpm_loss = total_ddpm_loss / (iteration + 1)
    avg_CV_loss = total_CV_loss / (iteration + 1)
    avg_obj = total_obj / (iteration + 1)

    logger.logger.info(
        f'Epoch: {epoch}, Validation Loss: {avg_val_loss} Loss_ddpm: {avg_ddpm_loss} Loss_CV:{avg_CV_loss}')
    logger.logger.info(f'Epoch: {epoch}, Validation fea: {fea} / 100, obj:{avg_obj}')
    return avg_val_loss


def main(args, logger):
    # checkpoint = f'./model/{args.type}/best_checkpoint_vae_False.pth'
    # modelPath = os.path.join(f'./model/{args.type}', checkpoint)
    # if not os.path.isdir(modelPath):
    #     checkpoint = None
    #     logger.logger.info('No such checkpoint!')
    ddpm = diffusion.DDPMTrainer(attn_dim=128, n_heads=4, n_layers=1, device=device,
                                 parameterization=f'{args.p}')
    ddpm.to(device)
    if not args.vae:
        decoder = SolutionDecoder(attn_dim=128, n_heads=4, n_layers=2, attn_mask=None)
        decoder.to(device)
        optimizer = Adam([
            {'params': ddpm.parameters(), 'lr': 0.0008},
            {'params': decoder.parameters(), 'lr': 0.0005}
        ])
        model = [ddpm, decoder]
    else:
        optimizer = Adam(ddpm.parameters(), 0.0008)
        model = ddpm
    epochs = 100
    scheduler = LambdaLR(optimizer, lr_lambda)
    logger.logger.info(f'start training {args.type}...... vae {args.vae}\n')
    best_val_loss = float('inf')
    optimizer.zero_grad()
    for epoch in range(epochs):
        mean_loss, mean_loss_CV = train_one_epoch(model, optimizer, scheduler, train_loader, device, epoch,
                                                  checkpoint=None)
        logger.logger.info('%d epoch train mean loss: %.4f CV_loss: %.4f\n' % (epoch, mean_loss, mean_loss_CV))

        val_loss = evaluate(model, val_loader, epoch)

        if epoch == 30:
            print('stop')

        if epoch + 1 >= args.save_epoch and (epoch + 1) % args.save_epoch == 0:
            checkpoint = {
                'ddpm_state_dict': ddpm.state_dict(),  # *模型参数
                'decoder_state_dict': decoder.state_dict() if args.vae is not True else None,
                'optimizer_state_dict': optimizer.state_dict(),  # *优化器参数
                'scheduler_state_dict': scheduler.state_dict(),  # *scheduler
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(path, f'checkpoint-%d-{args.vae}.pth' % epoch))
            logger.logger.info('save model %d successed......\n' % epoch)

        if epoch + 1 >= args.save_epoch and val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.logger.info('best model in %d epoch, validation loss: %.6f \n' % (epoch, val_loss))
            checkpoint = {
                'ddpm_state_dict': ddpm.state_dict(),  # *模型参数
                'decoder_state_dict': decoder.state_dict() if args.vae is not True else None,
                'optimizer_state_dict': optimizer.state_dict(),  # *优化器参数
                'scheduler_state_dict': scheduler.state_dict(),  # *scheduler
                'epoch': epoch,
            }
            torch.save(checkpoint, os.path.join(path, f'best_checkpoint_vae_{args.vae}.pth'))
            logger.logger.info('save best model successed......\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_epoch', type=float, default=10)
    parser.add_argument('--log_dir', type=str, default='./Logs/summary')
    parser.add_argument('--type', type=str, default="CA")
    parser.add_argument('--vae', type=bool, default=False)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--p', type=str, default='x0', help='whether eps or x0 the ddpm predict')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    graphSet = data_utils.GraphDataset(args.type)
    total_size = len(graphSet)
    train_size = int(total_size * 0.9)
    val_size = total_size - train_size
    train_set, val_set = random_split(graphSet, [train_size, val_size], generator=torch.Generator().manual_seed(2024))
    train_loader = torch_geometric.data.DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = torch_geometric.data.DataLoader(val_set, batch_size=args.batch, shuffle=False)

    path = f'./model/{args.type}'
    cisp = CISP()
    cisp.to(device)
    cisp.load_state_dict(torch.load(f'./model/{args.type}/cisp_pre/best_checkpoint.pth')['model_state_dict'])
    vae = CVAE(embedding=True)
    vae.to(device)
    vae.load_state_dict(torch.load(f'./model/{args.type}/cvae_embedding_pre/best_checkpoint.pth')['model_state_dict'])
    logger = data_utils.Logger(args)
    main(args, logger)
