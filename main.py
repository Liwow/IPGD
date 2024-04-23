# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import pickle

import ecole
from pyscipopt import Eventhdlr, SCIP_EVENTTYPE, SCIP_STATUS, SCIP_PARAMSETTING, SCIP_STAGE
from pyscipopt import Model
import numpy as np


def test(datatype):
    # 替换为您的.sol文件路径
    sol_file_path = f'dataset/train/{datatype}/BG/{datatype}_0.lp.pkl'

    # 使用pickle加载文件内容
    with open(sol_file_path, 'rb') as f:
        sol_data = pickle.load(f)

    # 打印内容，或者进行进一步的处理
    print(sol_data["variable_features"])
    print("done")


def mip_example():
    # 创建模型
    print("Start1")
    ptype = "CA"
    m = Model("Model")
    m.readProblem(f"./instance/train/{ptype}/{ptype}_0.lp")
    print("Start2")
    m.optimize()
    sol = m.getBestSol()

    print("End")


def ecole_example():
    model = ecole.scip.Model.from_file('./instance/train/CA/CA_0.lp')
    obs = ecole.observation.MilpBipartite().extract(model, True)
    constraint_features = obs.constraint_features
    edge_indices = np.array(obs.edge_features.indices, dtype=int)
    edge_features = obs.edge_features.values.reshape((-1, 1))
    variable_features = obs.variable_features
    graph = [constraint_features, edge_indices, edge_features, variable_features]
    return graph



if __name__ == "__main__":
    print("End")
    # test("IS")
