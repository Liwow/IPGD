import argparse
import os
import pickle
import torch
import torch_geometric
from pyscipopt.scip import Model

import data_utils
from cvae import CVAE
from data_utils import BipartiteNodeData
from utils import get_a_new2, normTorch
import cisp
from diffusion import DDPMSampler, DDIMSampler, DDPMTrainer
from decoder import SolutionDecoder


def ins2mip(ins):
    A, v_map, v_nodes, c_nodes, b_vars, n_int_vars, int_indices = get_a_new2(ins)
    v_nodes_norm = normTorch(v_nodes)
    c_nodes_norm = normTorch(c_nodes)
    c = v_nodes[:, 0]
    b = c_nodes[:, 2]
    constraint_features = c_nodes_norm
    variable_features = v_nodes_norm
    n_vars = variable_features.shape[0]

    edge_indices = A._indices()
    edge_features = A._values().unsqueeze(1).float()
    # edge_features = torch.ones(edge_features.shape)

    graph = BipartiteNodeData(
        torch.FloatTensor(constraint_features),
        torch.LongTensor(edge_indices),
        torch.FloatTensor(edge_features),
        torch.FloatTensor(variable_features),
        n_vars,
        b_vars.shape[0],
        b_vars,
        c,
        b
    )
    return graph.to(device)


def getSolsBySCIP(insFile, m, settings):
    ins_sol_file = f'./dataset/test/{args.type}/solution/{insFile}.sol'
    if os.path.exists(ins_sol_file):
        with open(ins_sol_file, "rb") as f:
            sol_data = pickle.load(f)
    else:
        if not os.path.isdir(f'./dataset/test/{args.type}'):
            os.mkdir(f'./dataset/test/{args.type}')
        if not os.path.isdir(f'./dataset/test/{args.type}/solution'):
            os.mkdir(f'./dataset/test/{args.type}/solution')

        # 设置参数
        m.setParam('limits/maxorigsol', settings['maxsol'])
        m.setParam('limits/time', settings['maxtime'])
        m.setParam('parallel/maxnthreads', settings['threads'])
        # m.setIntParam('emphasis/memory', 1)  # 减少内存使用
        m.hideOutput()
        m.optimize()
        best_sol = m.getBestSol()
        best_obj = m.getSolObjVal(best_sol)
        sol = []
        for var in m.getVars():
            sol.append(m.getSolVal(best_sol, var))

        print(f'{insFile} solved, best obj is {best_obj} ')
        sol_data = {'sols': sol, 'objs': best_obj}

    pickle.dump(sol_data, open(ins_sol_file, 'wb'))

    return sol_data


def get_obj_v(A, b, c, zi, x):
    # x : tensor
    x = x.view(zi.shape[0], -1, 1).float()
    pred_x_reshape = x.view(-1)
    Ax_minus_b = torch.sparse.mm(A, pred_x_reshape.unsqueeze(1)).squeeze(1) - b
    violates = torch.max(Ax_minus_b, torch.tensor(0)).sum()
    obj_value = (pred_x_reshape.squeeze() @ c).sum()
    return obj_value, violates


if __name__ == '__main__':
    torch.cuda.is_available = lambda: False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='./')
    parser.add_argument('--sampler', type=str, default='DDIM')
    parser.add_argument('--maxTime', type=int, default=1000)
    parser.add_argument('--maxStoredSol', type=int, default=500)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--vae', type=bool, default=False)
    parser.add_argument('--type', type=str, default='CA')
    parser.add_argument('--p', type=str, default='x0', help='whether eps or x0 the ddpm predict')
    args = parser.parse_args()
    status = 'train'
    SETTINGS = {
        'maxtime': args.maxTime,
        'mode': 2,
        'maxsol': args.maxStoredSol,
        'threads': args.threads,
    }

    ddpm_setting = {
        's': 15000,
        'gamma': 0.1
    }

    ddim_setting = {
        's': 10000,
        'gamma': 1e-4
    }
    samplerType = args.sampler

    ModelPath = f'./model/{args.type}/best_checkpoint_vae_False3.pth'
    cispPath = f'./model/{args.type}/cisp_pre/best_checkpoint.pth'
    vaePath = f'./model/{args.type}/cvae_pre/best_checkpoint.pth'

    vae = CVAE(embedding=True)
    vae.eval()
    cisp = cisp.CISP()
    cisp.eval()
    ddpm = DDPMTrainer(attn_dim=128, n_heads=4, n_layers=1, device=device, parameterization=f'{args.p}')
    ddpm.load_state_dict(torch.load(ModelPath, map_location=device)['ddpm_state_dict'])
    solutionDecoder = SolutionDecoder(attn_dim=128, n_heads=4, n_layers=2, attn_mask=None)
    solutionDecoder.eval()
    if args.vae is not True:
        cisp.load_state_dict(torch.load(cispPath, map_location=device)['model_state_dict'])
        solutionDecoder.load_state_dict(torch.load(ModelPath, map_location=device)['decoder_state_dict'])
        decoder = solutionDecoder
    else:
        vae.load_state_dict(torch.load(vaePath, map_location=device)['model_state_dict'])
        decoder = vae.decoder

    graphSet = data_utils.GraphDataset(args.type, status=status)
    data_loader = torch_geometric.data.DataLoader(graphSet, batch_size=1, shuffle=False)

    if samplerType == 'DDPM':
        sampler = DDPMSampler(ddpm, gradient_scale=ddpm_setting['s'], obj_guided_coef=ddpm_setting['gamma']
                              , decoder=decoder, device=device)
    elif samplerType == 'DDIM':
        sampler = DDIMSampler(ddpm, gradient_scale=ddim_setting['s'], obj_guided_coef=ddim_setting['gamma']
                              , decoder=decoder, device=device)
    else:
        raise ValueError

    sampler.eval()
    sols = []
    objs = []

    logger = data_utils.Logger(args, 'predict')
    insfile = f'./instance/{status}/{args.type}'
    filenames = os.listdir(insfile)
    feasible = 0
    for it, mip in enumerate(data_loader):
        filenames = mip.insFile
        for filename in filenames:
            m = Model('model')
            m.readProblem(os.path.join(insfile, filename))
            solution = m.createSol()
            variables = m.getVars()

            sol_data = getSolsBySCIP(filename, m, SETTINGS)
            best_x = sol_data['sols']
            best_obj = sol_data['objs']

            logger.logger.info(f'{filename} is starting, best_obj:{best_obj}')
            hyper_s = ddim_setting['s'] if args.sampler == 'DDIM' else ddpm_setting['s']
            hyper_gamma = ddim_setting['gamma'] if args.sampler == 'DDIM' else ddpm_setting['gamma']
            logger.logger.info(f'sampler:{args.sampler}; s:{hyper_s}; gamma:{hyper_gamma}')

            A, b, c = mip.getCoff()
            n_int_var = mip.n_int_vars

            tensor_x = torch.Tensor(best_x).long()
            zx, _ = cisp.encode_solution(tensor_x, n_int_var)
            t = torch.Tensor([999]).to(device).long()
            # sampler.initial_noise = ddpm.q_sample(zx, t=t)
            if args.vae is not True:
                zi, key = cisp.encode_mip(mip, n_int_var)
                sol_zx = decoder.apply_model(zi, zx, key)

                zx_pred, i = sampler.ip_guided_sample(zi, key, A, b, c)
                # zx_pred, i = sampler.sample(zi, key)
                sol_sigmoid = decoder.apply_model(zi, zx_pred, key)
            else:
                zi, key = vae.encode_mip(mip, n_int_var)
                zx_pred, i = sampler.ip_guided_sample(zi, key, A, b, c)
                sol_sigmoid = vae.decoder(zi, zx_pred, key)

            sol_pred = torch.round(sol_sigmoid)
            assert len(sol_pred) == len(variables), \
                "The solution does not match the number of variables."
            for value, var in zip(sol_pred, variables):
                m.setSolVal(solution, var, value)

            if m.checkSol(solution, printreason=False):
                obj = m.getSolObjVal(solution)
                feasible += 1
                objs.append(obj)
                sols.append(sol_pred)
                logger.logger.info(
                    f'{filename} is feasible, obj is {obj}, best_obj is {best_obj}, total_fea = {feasible}/100\n')
            else:
                obj = m.getSolObjVal(solution)
                logger.logger.info(f'{filename} is not feasible, obj:{obj}')

    print(f'feasible is {feasible} / 100')
