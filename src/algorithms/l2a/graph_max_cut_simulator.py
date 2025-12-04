import os
import sys
import time
import torch as th
from typing import List, Tuple

from graph_utils import load_graph_list, GraphList
from graph_utils import build_adjacency_bool, build_adjacency_indies, obtain_num_nodes,build_adjacency_matrix
from graph_utils import update_xs_by_vs, gpu_info_str, evolutionary_replacement

TEN = th.Tensor


class SimulatorGraphMaxCut:
    def __init__(self, sim_name: str = 'max_cut', graph_list: GraphList = (),
                 device=th.device('cpu'), if_bidirectional: bool = False):
        self.device = device
        self.sim_name = sim_name
        self.int_type = int_type = th.long
        self.if_maximize = True
        self.if_bidirectional = if_bidirectional

        '''load graph'''
        graph_list: GraphList = graph_list if graph_list else load_graph_list(graph_name=sim_name)

        '''建立邻接矩阵'''
        self.adjacency_matrix = build_adjacency_matrix(graph_list=graph_list, if_bidirectional=True).to(device)
        # self.adjacency_bool = build_adjacency_bool(graph_list=graph_list, if_bidirectional=True).to(device)
        # mask = self.adjacency_matrix >= 0.0
        self.edge_i, self.edge_j = self.adjacency_matrix.nonzero(as_tuple=True)
        self.edge_wts = self.adjacency_matrix[self.edge_i, self.edge_j]

        '''建立邻接索引'''
        n0_to_n1s, n0_to_dts = build_adjacency_indies(graph_list=graph_list, if_bidirectional=if_bidirectional)
        n0_to_n1s = [t.to(int_type).to(device) for t in n0_to_n1s]
        self.num_nodes = obtain_num_nodes(graph_list)
        self.num_edges = len(graph_list)
        self.adjacency_indies = n0_to_n1s

        '''基于邻接索引，建立基于边edge的索引张量：(n0_ids, n1_ids)是所有边(第0个, 第1个)端点的索引'''
        n0_to_n0s = [(th.zeros_like(n1s) + i) for i, n1s in enumerate(n0_to_n1s)]
        self.n0_ids = th.hstack(n0_to_n0s)[None, :]
        self.n1_ids = th.hstack(n0_to_n1s)[None, :]
        len_sim_ids = self.num_edges * (2 if if_bidirectional else 1)
        self.sim_ids = th.zeros(len_sim_ids, dtype=int_type, device=device)[None, :]
        self.n0_num_n1 = th.tensor([n1s.shape[0] for n1s in n0_to_n1s], device=device)[None, :]

    # def calculate_obj_values(self, xs: TEN, if_sum: bool = True) -> TEN:
    #     num_sims = xs.shape[0]  # 并行维度，环境数量。xs, vs第一个维度， dim0 , 就是环境数量
    #     if num_sims != self.sim_ids.shape[0]:
    #         self.n0_ids = self.n0_ids[0].repeat(num_sims, 1)
    #         self.n1_ids = self.n1_ids[0].repeat(num_sims, 1)
    #         self.sim_ids = self.sim_ids[0:1] + th.arange(num_sims, dtype=self.int_type, device=self.device)[:, None]

    #     values = xs[self.sim_ids, self.n0_ids] ^ xs[self.sim_ids, self.n1_ids]
    #     if if_sum:
    #         values = values.sum(1)
    #     if self.if_bidirectional:
    #         values = values // 2
    #     return values
    
    def calculate_obj_values(self, xs: TEN, if_sum: bool = True) -> TEN:
        """
        计算 Ising 模型哈密顿量 H = -∑_ij J_ij s_i s_j
        xs: [num_sims, N] （0/1 表示两个自旋态）
        返回：如果 if_sum=True，输出 [num_sims] 的能量向量；
            否则输出 [num_sims, E] 每条边的能量项（很少用）。
        """
        # 1) 把 0/1 映射到 ±1
        spins = (xs * 2 - 1).float()       # [num_sims, N]

        # 2) 取出每条边两端的自旋
        s_i = spins[:, self.edge_i]        # [num_sims, E]
        s_j = spins[:, self.edge_j]        # [num_sims, E]

        # 3) 对于 Ising：每条边的哈密顿量分量 H_ij = - J_ij * s_i * s_j
        terms =  self.edge_wts * (s_i * s_j)  # [num_sims, E]

        if not if_sum:
            # 返回所有边的分量
            return terms

        # 4) 如果 sum 模式，累加得到每个模拟的总能量
        H = terms.sum(dim=1)                # [num_sims]

        # 5) 如果 adjacency_matrix 是对称地存了两次，就除以 2
        if self.if_bidirectional:
            # print("here")
            H = H * 0.5

        return H


    # def calculate_obj_values_for_loop(self, xs: TEN, if_sum: bool = True, num_considered_nodes: int = 0) -> TEN:  # 代码简洁，但是计算效率低
    #     num_sims, num_nodes = xs.shape
    #     values = th.zeros((num_sims, num_nodes), dtype=self.int_type, device=self.device)
    #     rand_indices = th.randperm(num_nodes)[:num_considered_nodes]
    #     for node0 in rand_indices:
    #         node1s = self.adjacency_indies[node0]
    #         if node1s.shape[0] > 0:
    #             values[:, node0] = (xs[:, node0, None] ^ xs[:, node1s]).sum(dim=1)

    #     if if_sum:
    #         values = values.sum(dim=1)
    #     if self.if_bidirectional:
    #         values = values.float() / 2
    #     return values
    def calculate_obj_values_for_loop(
    self,
    xs: TEN,
    if_sum: bool = True,
    num_considered_nodes: int = 0
) -> TEN:
        """
        返回每条 sim、每个节点的「切边」权重和：
        - if_sum=False: 形状 [num_sims, num_nodes]
        - if_sum=True: 形状 [num_sims] 的总切边权重（所有节点求和）
        """
        num_sims, N = xs.shape
        # 0/1 → ±1
        spins = (xs * 2 - 1).float()                       # [num_sims, N]
        # 准备输出
        per_node = th.zeros((num_sims, N), device=xs.device)

        # 哪些节点要考虑？
        if num_considered_nodes <= 0 or num_considered_nodes > N:
            nodes = range(N)
        else:
            nodes = range(num_considered_nodes)

        for u in nodes:
            row    = self.adjacency_matrix[u]              # [N], 浮点：-1 表示无边
            neighs = (row >= 0).nonzero(as_tuple=True)[0]  # 所有邻居
            if neighs.numel() == 0:
                continue
            wts    = row[neighs]                           # [deg(u),]
            s_u    = spins[:, u:u+1]                       # [num_sims,1]
            s_vs   = spins[:, neighs]                      # [num_sims,deg(u)]
            # 对于 Max‐Cut/Ising 切边：cut = (s_u * s_vs < 0)
            # 取绝对值等价于 ½(1 - s_u*s_vs) 但这里直接用 (1 - sign)/2 不方便，
            # 我们用 XOR→float 就能得出 0/1，再乘权重：
            cuts   = (xs[:, u:u+1] ^ xs[:, neighs]).float()  # [num_sims,deg(u)]
            per_node[:, u] = (cuts * wts).sum(dim=1)         # [num_sims]

        if if_sum:
            totals = per_node.sum(dim=1)  # [num_sims]
            if self.if_bidirectional:
                totals = totals * 0.5
            return totals
        else:
            if self.if_bidirectional:
                return per_node * 0.5
            return per_node


    def generate_xs_randomly(self, num_sims):
        xs = th.randint(0, 2, size=(num_sims, self.num_nodes), dtype=th.bool, device=self.device)
        xs[:, 0] = 0
        return xs

    # def local_search_inplace(self, good_xs: TEN, good_vs: TEN,
    #                          num_iters: int = 8, num_spin: int = 8, noise_std: float = 0.3, num_considered_nodes: int = 0):

    #     vs_raw = self.calculate_obj_values_for_loop(good_xs, if_sum=False, num_considered_nodes=num_considered_nodes)
    #     good_vs = vs_raw.sum(dim=1).long() if good_vs.shape == () else good_vs.long()
    #     ws = self.n0_num_n1 - (2 if self.if_bidirectional else 1) * vs_raw
    #     ws_std = ws.max(dim=0, keepdim=True)[0] - ws.min(dim=0, keepdim=True)[0]
    #     rd_std = ws_std.float() * noise_std
    #     spin_rand = ws + th.randn_like(ws, dtype=th.float32) * rd_std
    #     thresh = th.kthvalue(spin_rand[:, :num_considered_nodes], k=min(self.num_nodes - num_spin, num_considered_nodes), dim=1)[0][:, None]

    #     for _ in range(num_iters):
    #         '''flip randomly with ws(weights)'''
    #         spin_rand = ws + th.randn_like(ws, dtype=th.float32) * rd_std
    #         spin_mask = spin_rand[:, :num_considered_nodes].gt(thresh)

    #         xs = good_xs.clone()
    #         xs[:, :num_considered_nodes][spin_mask] = th.logical_not(xs[:, :num_considered_nodes][spin_mask])
    #         vs = self.calculate_obj_values(xs)

    #         update_xs_by_vs(good_xs, good_vs, xs, vs, if_maximize=self.if_maximize)

    #     '''addition'''
    #     for i in range(num_considered_nodes):
    #         xs1 = good_xs.clone()
    #         xs1[:, i] = th.logical_not(xs1[:, i])
    #         vs1 = self.calculate_obj_values(xs1)

    #         update_xs_by_vs(good_xs, good_vs, xs1, vs1, if_maximize=self.if_maximize)
    #     return good_xs, good_vs
    def local_search_inplace(
    self,
    good_xs: TEN,
    good_vs: TEN,
    num_iters: int = 8,
    num_spin: int = 8,
    noise_std: float = 0.3,
    num_considered_nodes: int = 0
) -> Tuple[TEN, TEN]:
        """
        基于「节点局部切边权重 + 噪声」做 num_iters 轮翻转，
        然后再对每个节点尝试一次单点翻转。
        """
        if good_vs.dim() == 0:
            good_vs = self.calculate_obj_values(good_xs)
        xs = good_xs.clone()
        vs = good_vs.clone()
        num_sims, N = xs.shape

        # 确定哪些节点被考虑
        if num_considered_nodes <= 0 or num_considered_nodes > N:
            num_considered_nodes = N

        for _ in range(num_iters):
            # 1) 计算每条 sim、每个节点的局部切边权重
            local_w = self.calculate_obj_values_for_loop(
                xs, if_sum=False, num_considered_nodes=num_considered_nodes
            )  # [num_sims, N]

            # 2) 加点噪声，帮助逃离局部极值
            std_per_sim = local_w.std(dim=1, keepdim=True)            # [num_sims,1]
            noise = th.randn_like(local_w) * (std_per_sim * noise_std)
            scores = local_w + noise                                  # [num_sims, N]

            # 3) 对每条 sim，在前 num_considered_nodes 中挑 top-k 要翻转
            topk_idx = scores[:, :num_considered_nodes] \
                        .topk(k=min(num_spin, num_considered_nodes), dim=1) \
                        .indices   # [num_sims, num_spin]

            # 4) 构造翻转掩码
            flip = th.zeros_like(xs, dtype=th.bool)             # [num_sims, N]
            batch_idx = th.arange(num_sims, device=xs.device)[:,None]
            flip[batch_idx, topk_idx] = True

            # 5) 翻转并打分
            xs_new = xs.clone()
            xs_new[flip] = ~xs_new[flip]
            vs_new = self.calculate_obj_values(xs_new)

            # 6) 接受更优的解
            update_xs_by_vs(xs, vs, xs_new, vs_new, if_maximize=self.if_maximize)

        # 最后做一次「每个节点单点翻转」的补充
        for u in range(num_considered_nodes):
            xs1 = xs.clone()
            xs1[:, u] = ~xs1[:, u]
            vs1 = self.calculate_obj_values(xs1)
            update_xs_by_vs(xs, vs, xs1, vs1, if_maximize=self.if_maximize)

        return xs, vs



'''check'''


def find_best_num_sims():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    calculate_obj_func = 'calculate_obj_values'
    graph_name = 'gset_14'
    num_sims = 2 ** 16
    num_iter = 2 ** 6
    # calculate_obj_func = 'calculate_obj_values_for_loop'
    # graph_name = 'gset_14'
    # num_sims = 2 ** 13
    # num_iter = 2 ** 9

    if os.name == 'nt':
        graph_name = 'powerlaw_64'
        num_sims = 2 ** 4
        num_iter = 2 ** 3

    graph = load_graph_list(graph_name=graph_name)
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    simulator = SimulatorGraphMaxCut(sim_name=graph_name, graph_list=graph, device=device, if_bidirectional=False)

    print('find the best num_sims')
    from math import ceil
    for j in (1, 1, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        _num_sims = int(num_sims * j)
        _num_iter = ceil(num_iter * num_sims / _num_sims)

        timer = time.time()
        for i in range(_num_iter):
            xs = simulator.generate_xs_randomly(num_sims=_num_sims)
            vs = getattr(simulator, calculate_obj_func)(xs=xs)
            assert isinstance(vs, TEN)
            # print(f"| {i}  max_obj_value {vs.max().item()}")
        print(f"_num_iter {_num_iter:8}  "
              f"_num_sims {_num_sims:8}  "
              f"UsedTime {time.time() - timer:9.3f}  "
              f"GPU {gpu_info_str(device)}")


def check_simulator():
    gpu_id = -1
    num_sims = 16
    num_nodes = 24
    graph_name = f'powerlaw_{num_nodes}'

    graph = load_graph_list(graph_name=graph_name)
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    simulator = SimulatorGraphMaxCut(sim_name=graph_name, graph_list=graph, device=device)

    for i in range(8):
        xs = simulator.generate_xs_randomly(num_sims=num_sims)
        obj = simulator.calculate_obj_values(xs=xs)
        print(f"| {i}  max_obj_value {obj.max().item()}")
    pass


def check_local_search():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    graph_type = 'gset_14'
    graph_list = load_graph_list(graph_name=graph_type)
    num_nodes = obtain_num_nodes(graph_list)

    show_gap = 4

    num_sims = 2 ** 8
    num_iters = 2 ** 8
    reset_gap = 2 ** 6
    save_dir = f"./{graph_type}_{num_nodes}"

    if os.name == 'nt':
        num_sims = 2 ** 2
        num_iters = 2 ** 5

    '''simulator'''
    sim = SimulatorGraphMaxCut(graph_list=graph_list, device=device, if_bidirectional=True)
    if_maximize = sim.if_maximize

    '''evaluator'''
    good_xs = sim.generate_xs_randomly(num_sims=num_sims)
    good_vs = sim.calculate_obj_values(xs=good_xs)
    from evaluator import Evaluator
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, if_maximize=if_maximize,
                          x=good_xs[0], v=good_vs[0].item(), )

    for i in range(num_iters):
        evolutionary_replacement(good_xs, good_vs, low_k=2, if_maximize=if_maximize)

        for _ in range(4):
            sim.local_search_inplace(good_xs, good_vs)

        if_show_x = evaluator.record2(i=i, vs=good_vs, xs=good_xs)
        if (i + 1) % show_gap == 0 or if_show_x:
            show_str = f"| cut_value {good_vs.float().mean():8.2f} < {good_vs.max():6}"
            evaluator.logging_print(show_str=show_str, if_show_x=if_show_x)
            sys.stdout.flush()

        if (i + 1) % reset_gap == 0:
            print(f"| reset {gpu_info_str(device=device)} "
                  f"| up_rate {evaluator.best_v / evaluator.first_v - 1.:8.5f}")
            sys.stdout.flush()

            good_xs = sim.generate_xs_randomly(num_sims=num_sims)
            good_vs = sim.calculate_obj_values(xs=good_xs)

    print(f"\nbest_x.shape {evaluator.best_x.shape}"
          f"\nbest_v {evaluator.best_v}"
          f"\nbest_x_str {evaluator.best_x_str}")


if __name__ == '__main__':
    check_simulator()
    # check_local_search()
