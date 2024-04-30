import copy
import os
import pickle
import time
import logging  # 引入logging模块
import os.path
import numpy as np
import torch
import torch_geometric
from torch.utils.data import Dataset
from utils import normTorch

epsilon = 1e-5


def read(file):
    with open(file, 'rb') as file:
        data = pickle.load(file)

    return data


def mip_collate_fn(batch):
    mips, xs = zip(*batch)
    batch_constraint_features = torch.stack([item.constraint_features for item in mips])
    batch_edge_attr = torch.stack([item.edge_attr for item in mips])
    batch_variable_features = torch.stack([item.variable_features for item in mips])
    batch_edge_index = torch.stack([item.edge_index for item in mips])
    batch_int_indices = torch.stack([item.int_indices for item in mips])
    batch_xs = torch.stack(xs)
    mip = {
        'constraint_features': batch_constraint_features,
        'edge_attr': batch_edge_attr,
        'variable_features': batch_variable_features,
        'edge_index': batch_edge_index,
        'int_indices': batch_int_indices
    }
    mipModel = copy.deepcopy(mips[0])
    mipModel.set_features(mip)
    return mipModel, batch_xs


def getMipModelList(ptype, status="train"):
    mipList = []
    filename = f'./dataset/{status}/{ptype}'
    insfilenames = os.listdir(f'./instance/{status}/{ptype}')
    for ins in insfilenames:
        mip = MipModel(filename, ins, status)
        mipList.append(mip)
    return mipList


def norm(arr):
    min_obj = arr.min()
    max_obj = arr.max()
    diff = max_obj - min_obj
    arr_norm = (arr - min_obj) / diff
    return arr_norm


def getSolByObj(sols, is_min_obj):
    solution_objs = np.array(sols['objs'])
    mean_obj = solution_objs.mean()
    if is_min_obj:
        # solution_objs = solution_objs[solution_objs < mean_obj]
        solution_objs_norm = norm(solution_objs)
        # solution_probs = np.exp(-np.power(solution_objs, 1/3))  # exp(-x^ 1/3)
        solution_probs = 1 / (solution_objs_norm + 0.1)
    else:
        # solution_objs = solution_objs[solution_objs > mean_obj]
        solution_objs_norm = norm(solution_objs)
        solution_probs = np.power(solution_objs_norm, 2)
    solution_probs /= solution_probs.sum()
    chosen_index = np.random.choice(solution_objs.shape[0], p=solution_probs)
    return chosen_index


class Logger:
    def __init__(self, args, user='train', mode='w'):
        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        if user == 'train':
            rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))+f'{args.type}_ddpm_{args.vae}'
        elif user == 'pretrain':
            rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + f'{args.type}_{args.model}'
        else:
            rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time())) + f'{args.type}'
        log_path = os.getcwd() + '/Logs/'
        log_name = log_path + rq + '.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode=mode)
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)


class MipModel():
    def __init__(self, filename, ins, status="train"):
        self.key_padding_mask = None
        self.status = status
        self.mipPath = os.path.join(filename, 'BG', ins + '.pkl')
        self.solPath = os.path.join(filename, 'solution', ins + '.sol')
        mip = read(self.mipPath)
        self.constraint_features = mip['constraint_features']
        self.edge_attr = mip['edge_attr']
        self.variable_features = mip['variable_features']
        self.edge_index = mip['edge_index']
        self.n_vars = mip['n_vars']
        self.n_int_vars = mip['n_int_vars']
        self.int_indices = mip['all_integer_variable_indices']
        self.sols_data = read(self.solPath)

    def get_key_padding_mask(self):
        return self.key_padding_mask

    def set_key_padding_mask(self, key):
        self.key_padding_mask = key

    def set_features(self, mip):
        self.constraint_features = mip['constraint_features']
        self.edge_attr = mip['edge_attr']
        self.variable_features = mip['variable_features']
        self.edge_index = mip['edge_index']
        self.int_indices = mip['int_indices']


class MyDataset(Dataset):
    def __init__(self, ptype, status="train"):
        self.ptype = ptype
        self.status = status
        self.key = None
        self.mipList = getMipModelList(ptype)
        self.sols = self.read_solutions()
        self.is_min_obj = self.get_is_min_obj()

    def __len__(self):
        return len(self.sols)

    def __getitem__(self, idx):
        mip = self.mipList[idx]
        sol = self.sols[idx]
        chosen_index = getSolByObj(sol, self.is_min_obj)
        return mip, torch.tensor(sol['sols'][chosen_index], dtype=torch.float)

    def read_solutions(self):
        sols = []
        ins_filenames = os.listdir(f'./instance/{self.status}/{self.ptype}')
        sols_file_path = f'./dataset/{self.status}/{self.ptype}/solution'
        for ins_filename in ins_filenames:
            sol_file_path = os.path.join(sols_file_path, ins_filename + '.sol')
            with open(sol_file_path, 'rb') as f:
                sol_data = pickle.load(f)  # {var_names sols, objs}
                sols.append(sol_data)
        return sols

    def get_type(self):
        return self.ptype

    def get_mipList(self):
        return self.mipList

    def set_keypadding(self, key):
        self.key = key

    def get_is_min_obj(self):
        if self.ptype == "CF":
            return True
        else:
            return False


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, ptype, status="train"):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.DIR_BG = f'./dataset/{status}/{ptype}/BG'
        self.DIR_SOL = f'./dataset/{status}/{ptype}/solution'
        self.sample_names = os.listdir(self.DIR_BG)
        self.sample_files = [(os.path.join(self.DIR_BG, name),
                              os.path.join(self.DIR_SOL, name).replace('bg', 'sol')) for name in
                             self.sample_names]
        self.type = ptype
        self.status = status
        self.is_min_obj = self.get_is_min_obj(ptype)

    def len(self):
        return len(self.sample_files)

    def process_sample(self, filepath):
        BGFilepath, solFilePath = filepath
        with open(BGFilepath, "rb") as f:
            bgData = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        BG = bgData
        varNames = solData['var_names']

        sols = solData['sols']  # [0:300]
        objs = solData['objs']  # [0:300]

        sols = np.round(sols, 0)
        if self.status == 'train':
            chosen_index = getSolByObj(solData, self.is_min_obj)
            sol = sols[chosen_index]
            obj = objs[chosen_index]
        elif self.status == 'test':
            sol = sols
            obj = objs
        else:
            raise ValueError
        return BG, sol, obj, varNames

    def get_is_min_obj(self, ptype):
        if ptype == "CF" or ptype == "SC":
            return True
        else:
            return False

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        # nbp, sols, objs, varInds, varNames = self.process_sample(self.sample_files[index])
        BG, sols, objs, varNames = self.process_sample(self.sample_files[index])
        A, v_map, v_nodes, c_nodes, b_vars, n_int_vars, int_indices = BG

        insfile = self.sample_names[index][:-3]

        v_nodes_norm = normTorch(v_nodes)
        c_nodes_norm = normTorch(c_nodes)
        c = v_nodes[:, 0]
        b = c_nodes[:, 2]

        constraint_features = c_nodes_norm
        edge_indices = A._indices()

        variable_features = v_nodes_norm
        edge_features = A._values().unsqueeze(1).float()
        # edge_features = torch.ones(edge_features.shape)

        # constraint_features[np.isnan(constraint_features)] = 1
        n_vars = variable_features.shape[0]
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
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.sols = torch.LongTensor(sols)
        graph.insFile = insfile

        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.solutions = torch.FloatTensor(sols).reshape(-1)
        graph.ntvars = variable_features.shape[0]
        graph.objVals = objs
        graph.nsols = sols.shape[0]
        varname_dict = {}
        varname_map = []
        i = 0
        for iter in varNames:
            varname_dict[iter] = i
            i += 1
        for iter in v_map:
            varname_map.append(varname_dict[iter])

        varname_map = torch.tensor(varname_map)

        # graph.varInds = [[varname_map], [b_vars]]

        # constraint_features = BG['constraint_features']
        # edge_indices = BG['edge_index']
        #
        # variable_features = BG['variable_features']
        # edge_features = BG['edge_attr']
        # # edge_features = torch.ones(edge_features.shape)
        #
        # # constraint_features[np.isnan(constraint_features)] = 1
        #
        # graph = BipartiteNodeData(
        #     torch.FloatTensor(constraint_features),
        #     edge_indices,
        #     torch.FloatTensor(edge_features),
        #     torch.FloatTensor(variable_features),
        #     BG['n_vars'],
        #     BG['n_int_vars'],
        #     BG['all_integer_variable_indices'],
        # )
        # graph.sols = torch.FloatTensor(sols)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return graph.to(device)


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            n_vars,
            n_int_vars,
            int_indices,
            c,
            b
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.n_vars = n_vars
        self.n_int_vars = n_int_vars
        self.int_indices = int_indices
        self.sols = None
        self.b = b
        self.c = c

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def getCoff(self):
        nc = self.constraint_features.shape[0]
        nv = self.variable_features.shape[0]
        A = torch.sparse_coo_tensor(self.edge_index, self.edge_attr.squeeze(), size=(nc, nv))

        b = self.b
        c = self.c
        return A, b, c
