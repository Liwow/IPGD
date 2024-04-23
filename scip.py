import os.path
import pickle
from datetime import datetime
from multiprocessing import Process, Queue
from pyscipopt import Model, quicksum, multidict
import numpy as np
import argparse
from utils import get_a_new2
from contextlib import redirect_stdout


def solve_scip(filepath, log_dir, settings):
    m = Model("Model")
    m.readProblem(filepath)

    # 设置参数
    m.setParam('limits/maxorigsol', settings['maxsol'])
    m.setParam('limits/maxsol', settings['maxsol'])
    m.setParam('limits/time', settings['maxtime'])
    m.setParam('parallel/maxnthreads', settings['threads'])
    # m.setIntParam('emphasis/memory', 1)  # 减少内存使用
    m.hideOutput()  # SCIP的等价于Gurobi的LogToConsole参数设置

    log_path = os.path.join(log_dir, os.path.basename(filepath) + '.log')
    with open(log_path, 'w') as log_file:
        with redirect_stdout(log_file):
            m.optimize()

    if args.status == 'train':
        # 提取解及其目标值
        sols = []
        objs = []
        solc = m.getNSols()

        for i in range(solc):
            sol = m.getSols()[i]
            sols.append([m.getSolVal(sol, var) for var in m.getVars()])
            objs.append(m.getSolObjVal(sol))

        sols = np.array(sols, dtype=np.float32)
        objs = np.array(objs, dtype=np.float32)


        sol_data = {
            'var_names': [var.name for var in m.getVars()],
            'sols': sols,
            'objs': objs,
        }

    else:
        best_sol = m.getBestSol()
        best_obj = m.getSolObjVal(best_sol)
        sol = []
        for var in m.getVars():
            sol.append(m.getSolVal(best_sol, var))
        sol_data = {'var_names': [var.name for var in m.getVars()], 'sols': sol, 'objs': best_obj}

    return sol_data


def collect(ins_dir, q, sol_dir, log_dir, bg_dir, settings):
    while True:
        filename = q.get()
        if not filename:
            print("filename is None")
            break
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"开始处理: {filename} 时间: {current_time}")
        filepath = os.path.join(ins_dir, filename)
        # get bipartite graph , binary variables' indices
        A2, v_map2, v_nodes2, c_nodes2, b_vars2, n_int_vars, int_indices = get_a_new2(filepath)
        BG_data = [A2, v_map2, v_nodes2, c_nodes2, b_vars2, n_int_vars, int_indices]
        # solver = Solver()
        # solver.load_model(filepath)·
        # features = solver.extract_lp_features_at_root()
        # sol_data = solve_scip(filepath, log_dir, settings)
        # # save data
        # pickle.dump(sol_data, open(os.path.join(sol_dir, filename + '.sol'), 'wb'))
        pickle.dump(BG_data, open(os.path.join(bg_dir, filename + '.bg'), 'wb'))


if __name__ == '__main__':
    sizes = ["IS", "CA", "SC"]
    # sizes = ['CF']

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataDir', type=str, default='./')
    parser.add_argument('--nWorkers', type=int, default=20)
    parser.add_argument('--maxTime', type=int, default=1000)
    parser.add_argument('--maxStoredSol', type=int, default=500)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--status', type=str, default='test')
    args = parser.parse_args()

    for size in sizes:

        dataDir = args.dataDir
        status = args.status
        INS_DIR = os.path.join(dataDir, f'instance/{status}/{size}')

        if not os.path.isdir(f'./dataset/{status}/{size}'):
            os.mkdir(f'./dataset/{status}/{size}')
        if not os.path.isdir(f'./dataset/{status}/{size}/solution'):
            os.mkdir(f'./dataset/{status}/{size}/solution')
        if not os.path.isdir(f'./dataset/{status}/{size}/NBP'):
            os.mkdir(f'./dataset/{status}/{size}/NBP')
        if not os.path.isdir(f'./dataset/{status}/{size}/logs'):
            os.mkdir(f'./dataset/{status}/{size}/logs')
        if not os.path.isdir(f'./dataset/{status}/{size}/BG'):
            os.mkdir(f'./dataset/{status}/{size}/BG')

        SOL_DIR = f'./dataset/{status}/{size}/solution'
        LOG_DIR = f'./dataset/{status}/{size}/logs'
        BG_DIR = f'./dataset/{status}/{size}/BG'
        os.makedirs(SOL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(BG_DIR, exist_ok=True)

        N_WORKERS = args.nWorkers

        # gurobi settings
        SETTINGS = {
            'maxtime': args.maxTime,
            'mode': 2,
            'maxsol': args.maxStoredSol,
            'threads': args.threads,
        }

        filenames = os.listdir(INS_DIR)

        q = Queue()
        # add ins
        for filename in filenames:
            if not os.path.exists(os.path.join(BG_DIR, filename + '.bg')):
                q.put(filename)
        # add stop signal
        for i in range(N_WORKERS):
            q.put(None)

        ps = []
        for i in range(N_WORKERS):
            p = Process(target=collect, args=(INS_DIR, q, SOL_DIR, LOG_DIR, BG_DIR, SETTINGS))
            p.start()
            ps.append(p)
        for p in ps:
            p.join()

        print('done')
