from demo_graph_max_cut_single_instant_2 import solve_single_graph_problem_using_trs
from graph_max_cut_simulator import *
from graph_max_cut_trs import *

import os
from natsort import natsorted
import json
import argparse
import random
import numpy as np
import torch as th
from math import sqrt
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from config import ConfigGraph, ConfigPolicy
from evaluator import Evaluator
from graph_max_cut_trs import TrsCell, Buffer, convert_solution_to_prob, sub_set_sampling, get_advantages
from network import GraphTRS, create_mask
from graph_utils import load_graph_list, gpu_info_str, GraphTypes
from graph_utils import update_xs_by_vs, pick_xs_by_vs
from graph_utils import GraphList, build_adjacency_bool
from graph_embedding_pretrain import train_graph_net_in_a_single_graph, train_graph_net_in_graph_distribution


# def run_inference_on_graph(
#     graph_path: str,
#     model_path: str,
#     graph_type: str,
#     num_nodes: int,
#     gpu_id: int = 0
# ):

#     device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

#     graph_list = load_graph_list(graph_path)
#     args_graph = ConfigGraph(graph_list=graph_list, graph_type=graph_type)
#     args_policy = ConfigPolicy(graph_list=graph_list, graph_type=graph_type)

#     net = TrsCell(embed_dim=args_graph.embed_dim, num_heads=args_graph.num_heads, num_layers=1).to(device)
#     assert os.path.exists(model_path), f"exists?: {model_path}"
#     net.load_state_dict(th.load(model_path, map_location=device))
#     print(f"Found in：{model_path}")

#     cl_stages = [(4, 2, 6, 1.0, 0.1)]
#     graph_name = os.path.basename(graph_path).replace('.txt', '')

#     scores, states = solve_single_graph_problem_using_trs(
#         graph_list=graph_list,
#         args_graph=args_graph,
#         args_policy=args_policy,
#         cl_stages=cl_stages,
#         gpu_id=gpu_id,
#         is_CL=False,
#         exp_name='EvalOnly',
#         graph_name=graph_name,
#         net=net,
#         inference_only=True
#     )

#     final_scores = scores[-1]
#     best_score = max([max(s) for s in final_scores])
#     print(f"Best inference score= {best_score}")
#     return scores, states, best_score

def single_l2a(graph_type):
    graph_type = graph_type
    gpu_id: int = 0

    graph_list = load_graph_list(graph_name=graph_type)
    args_graph = ConfigGraph(graph_list=graph_list, graph_type=graph_type)
    args_graph.batch_size = 2 ** 6
    args_graph.num_buffers = 10
    args_graph.buffer_repeats = 32
    args_graph.show_gap = 2 ** 0

    args_policy = ConfigPolicy(graph_list=graph_list, graph_type=graph_type)
    args_policy.num_sims = 2 ** 6

    cl_stages = [(4, 6, 6, 0.55, 1.0), (4, 4, 6, 0.83, 0.5), (4, 2, 6, 1.0, 0.1)]
    is_CL = True
    exp_name = "PG_CL"
    graph_name = graph_type.split("/")[-1].split(".")[0]

    scores, _ = solve_single_graph_problem_using_trs(
        graph_list,
        args_graph,
        args_policy,
        cl_stages=cl_stages,
        gpu_id=gpu_id,
        is_CL=is_CL,
        exp_name=exp_name,
        graph_name=graph_name,
    )

    final_scores = scores[-1]
    best_score = max([max(s) for s in final_scores])
    print(f"Best inference score= {best_score}")

    return best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_instance", type=str,
                        help="input the data file for the problem instance")

    args = parser.parse_args()
    data_dir = args.problem_instance
    input_files = [ f for f in os.listdir(data_dir) ]
    input_files = natsorted(input_files)

    results_dir = f"results/{'/'.join(data_dir.split('/')[1:])}"
    os.makedirs(results_dir, exist_ok=True)

    i = 0
    for file in input_files:
        if os.path.isdir(f'{data_dir}/{file}'): continue
        i += 1
        if i < 15: continue
        if i > 25: break
        print(file)

        best_energies = single_l2a(f"{data_dir}/{file}")
        best_energies *= -1

        with open(f"{results_dir}/L2A.txt", "a") as f:
            f.write(f"{best_energies}\n")
