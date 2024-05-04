import argparse

import ecole
import torch
import numpy as np
import pyscipopt as scp
from encoder import IPEncoder

# torch.cuda.is_available = lambda : False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prenorm(model, pretrain_loader):
    """
    Pre-nomalize all PreNorm layers in the model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to pre-normalize.
    pretrain_loader : torch_geometric.data.DataLoader
        Pre-loaded dataset of pre-training samples.

    Returns
    -------
    i : int
        Number of pre-trained layers.
    """
    model.pre_norm_init()

    i = 0
    while True:
        for batch in pretrain_loader:
            batch.to(device)
            pre_norm = True
            if isinstance(model, IPEncoder):
                pre_norm = model.pre_norm(
                    batch.constraint_features,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.variable_features
                )
            if not pre_norm:
                break

        if model.pre_norm_next() is None:
            break
        i += 1
    return i


def get_padding(features, n_int_vars, padding_len, pad_object):
    if isinstance(n_int_vars, int):
        split_features = torch.split(features, [n_int_vars])
    else:
        split_features = torch.split(features, n_int_vars.tolist())
    if pad_object == 'mip':
        padding_features = map(
            lambda v: torch.nn.functional.pad(v, (0, 0, 0, padding_len - v.shape[0]), 'constant', 0).unsqueeze(0),
            split_features)
    elif pad_object == 'solution':
        padding_features = map(
            lambda v: torch.nn.functional.pad(v, (0, padding_len - v.shape[0]), 'constant', 2).unsqueeze(0),
            split_features)
    else:
        raise Exception("Padding object not found!")
    key_padding_mask = map(lambda v: torch.nn.functional.pad(
        torch.zeros(v.shape[0], dtype=torch.bool, device=v.device), (0, padding_len - v.shape[0]), 'constant',
        True).unsqueeze(0),
                           split_features)

    features_out = torch.concat(list(padding_features), dim=0)
    key_padding_mask = torch.concat(list(key_padding_mask), dim=0)
    return features_out, key_padding_mask


## diffusion 
def isfunction(obj):
    return callable(obj)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract_into_tensor(a, t, x_shape):
    """
    collect the data in a at the index of t
    a:[time_step]
    t:[batch_size]
    return:[batch_size, 1]
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def to_torch(x):
    if torch.is_tensor(x):
        out = x.clone().detach().float()
    else:
        out = torch.from_numpy(x).float()
    return out


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    give the sequence of betas
    return: tensor(n_timestep)
    """
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2 ** 32 - 1:
        raise argparse.ArgumentTypeError(
            "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed

def position_get_ordered_flt(variable_features):
    lens = variable_features.shape[0]
    feature_widh = 20  # max length 4095
    sorter = variable_features[:, 1]
    position = torch.argsort(sorter)
    position = position / float(lens)

    position_feature = torch.zeros(lens, feature_widh)

    for row in range(position.shape[0]):
        flt_indx = position[row]
        divider = 1.0
        for ks in range(feature_widh):
            if divider <= flt_indx:
                position_feature[row][ks] = 1
                flt_indx -= divider
            divider /= 2.0
            # print(row,position[row],position_feature[row])
    position_feature = position_feature.to(device)
    variable_features = variable_features.to(device)
    v = torch.concat([variable_features, position_feature], dim=1)
    return v


def get_BG_from_scip(ins_name):
    epsilon = 1e-6

    # vars:  [obj coeff, norm_coeff, degree, max coeff, min coeff, Bin?]
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)

    v_nodes = []

    b_vars = []

    ori_start = 6
    emb_num = 15

    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e+20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == 'BINARY':
            tp[ori_start - 1] = 1
            b_vars.append(i)

        v_nodes.append(tp)
    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    obj_node = [0, 0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        v_nodes[v_indx][0] = v

        # print(v_indx,float(nvars),v_indx/float(nvars),v_nodes[v_indx][ori_start:ori_start+emb_num])

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]
    # quit()

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            # print(coeff,c)
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]
    # for i in range(len(cons_map)):
    #   tmp=0
    #   coeff=m.getValsLinear(cons_map[i][0])
    #    for k in coeff:
    #        tmp+=coeff[k]
    #    cons_map[i].append(tmp)
    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]
    # print(cons)
    # quit()
    A = []
    for i in range(ncons):
        A.append([])
        for j in range(nvars + 2):
            A[i].append(0)
    A.append(obj_cons)
    lcons = ncons
    c_nodes = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        A[cind][-2] = rhs
        sense = 0

        if rhs == lhs:
            sense = 2
        elif rhs >= 1e+20:
            sense = 1
            rhs = lhs

        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            A[cind][v_indx] = 1
            A[cind][-1] += 1
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff[k] / lcons
            if v_indx == 1066:
                print(coeff[k], lcons)
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            # v_nodes[v_indx][3]+=cind*coeff[k]
            summation += coeff[k]
        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])
    c_nodes.append(obj_node)
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    A = np.array(A, dtype=np.float32)

    A = A[:, :-2]
    A = torch.as_tensor(A).to(device).to_sparse()
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)
    # v_nodes=position_get_ordered(v_nodes)
    v_nodes = position_get_ordered_flt(v_nodes)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars


def get_a_new2(ins_name):
    epsilon = 1e-6
    torch.cuda.is_available = lambda: False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # vars:  [obj coeff, norm_coeff, degree, Bin?]
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()

    mvars = m.getVars()

    v_nodes = []
    int_indices = []
    b_vars = []

    ori_start = 6
    emb_num = 15

    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e+20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == 'BINARY':
            tp[ori_start - 1] = 1
            b_vars.append(i)
        if mvars[i].vtype() == 'INTEGER':
            int_indices.append(i)

        v_nodes.append(tp)
    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        # 在A里添加目标函数
        # if v != 0:
        #     indices_spr[0].append(0)
        #     indices_spr[1].append(v_indx)
        #     # values_spr.append(v)
        #     values_spr.append(1)
        v_nodes[v_indx][0] = v

        # print(v_indx,float(nvars),v_indx/float(nvars),v_nodes[v_indx][ori_start:ori_start+emb_num])

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]
    # quit()

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            # print(coeff,c)
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]

    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]

    lcons = ncons
    c_nodes = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        # A[cind][-2]=rhs
        sense = 0

        if rhs == lhs:
            sense = 2
        elif rhs >= 1e+20:
            sense = 1
            rhs = lhs

        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            # A[cind][v_indx]=1
            # A[cind][-1]+=1
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(1)
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff[k] / lcons
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            # v_nodes[v_indx][3]+=cind*coeff[k]
            summation += coeff[k]
        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])

    # ?
    # c_nodes.append(obj_node)
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32).to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32).to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32).to(device)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons + 1, nvars)).to(device)

    model = ecole.scip.Model.from_file(ins_name)
    obs = ecole.observation.MilpBipartite().extract(model, True)
    # edge_indices = np.array(obs.edge_features.indices, dtype=int)
    variable_features = obs.variable_features
    variable_features = torch.from_numpy(variable_features[:, -8:]).float()
    v_nodes = torch.cat((v_nodes[:, :5], variable_features), dim=1)

    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    # v_nodes = normTorch(v_nodes)
    v_nodes = torch.clamp(v_nodes, 1e-5, float('inf'))
    # v_nodes=position_get_ordered(v_nodes)
    # v_nodes=position_get_ordered_flt(v_nodes)

    # c_nodes = normTorch(c_nodes)
    c_nodes = torch.clamp(c_nodes, 1e-5, float('inf'))

    n_int_vars = m.getNIntVars()

    return A, v_map, v_nodes, c_nodes, b_vars, n_int_vars, int_indices


def to_norm(tensor):
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True)
    return (tensor - mean) / std

def normTorch(x):
    maxs = torch.max(x, 0)[0]
    mins = torch.min(x, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    x = x - mins
    x = x / diff
    x = torch.clamp(x, 1e-5, 1)
    return x


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output