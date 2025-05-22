from __future__ import print_function
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# print("CUDA_LAUNCH_BLOCKING:", os.getenv("CUDA_LAUNCH_BLOCKING"))
# print("TORCH_USE_CUDA_DSA:", os.getenv("TORCH_USE_CUDA_DSA"))
import argparse
import yaml
import json
from datetime import datetime
now = datetime.now()
timestamp = str(now.year)[-2:] + "_" + str(now.month).zfill(2) + "_" + str(now.day).zfill(2) + "_" + \
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

######################## PLEASE CHECK THIS FIELD AND LINE 319 PROJECT NAME AND ENTITY INSIDE ##########################
os.environ["WANDB_API_KEY"] = "ABCDEFG"
WANDB_PROJECT_NAME = "MolStitch"
WANDB_ENTITY_NAME = "YOURNAME"
#######################################################################################################################

import sys
sys.path.append(os.path.realpath(__file__))
from tdc import Oracle
from time import time 
import numpy as np 
from evaluators.hypervolume import get_hypervolume, get_pareto_fronts
from main.optimizer import chebyshev_scalarization_batch
from evaluators.dock.qvina2 import QuickVina2

''' 
name example 
"qed:1+sa:1+jnk3:1+gsk3b:1"
'''


class MultiOracle(Oracle):
    def __init__(self, name, cheby=None, target_smiles=None, num_max_call=None, **kwargs):
        name_split = name.split('+')
        self.name_list = [i.split(':')[0] for i in name_split]
        self.weight_list = [float(i.split(':')[1]) for i in name_split]
        self.weight_list = [i/sum(self.weight_list) for i in self.weight_list]
        self.docking_list = ["parp1", "jak2", "braf", "fa7", "5ht1b"]
        self.oracle_list = []
        for i in self.name_list:
            if i in self.docking_list:
                self.oracle_list.append(QuickVina2(i))
            else:
                self.oracle_list.append(Oracle(i, target_smiles, num_max_call, **kwargs))
        self.name = name
        self.temp_oracle_score = {}
        self.pareto_rewards = np.array([])
        self.pareto_smiles = np.array([])
        self.cheby = cheby

    def __call__(self, return_all=False, *args, **kwargs):
        temp_list = []
        temp_hyper = []
        temp_smi = []

        smi = args[0]

        for oracle, weight in zip(self.oracle_list, self.weight_list):
            name = oracle.name
            score = oracle(*args, **kwargs)

            if name == "sa":
                score = (10 - score) / 9
            if name in self.docking_list:
                score = (-score / 20).item()
                if score < 0:
                    score = 0
            temp_list.append(score * weight)
            self.temp_oracle_score[name] = score
            temp_hyper.append(score)

        temp_smi.append(*args)
        temp_hyper = np.array(temp_hyper)
        temp_smi = np.array(temp_smi)
        temp_smi = np.expand_dims(temp_smi, 0)
        temp_pare = np.expand_dims(temp_hyper, 0)

        if self.pareto_rewards.ndim == 2:
            temp_pare = np.append(temp_pare, self.pareto_rewards, axis=0)
            temp_smi = np.append(temp_smi, self.pareto_smiles, axis=0)
        candidates, pareto_rewards = get_pareto_fronts(temp_smi, temp_pare)
        self.pareto_rewards = pareto_rewards
        self.pareto_smiles = candidates
        if self.cheby:
            avg_score = chebyshev_scalarization_batch(temp_hyper, weights=self.weight_list)
        else:
            avg_score = np.sum(temp_list)
        if return_all:
            return avg_score, temp_hyper
        else:
            return avg_score


def main():
    start_time = time() 
    parser = argparse.ArgumentParser()
    parser.add_argument('method', default='graph_ga')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=501)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--offline_batch_size', type=int, default=5000)
    # parser.add_argument('--buffer_size', type=int, default=100)
    parser.add_argument('--freq_log', type=int, default=200)
    parser.add_argument('--n_runs', type=int, default=5)
    # parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seed', type=int, nargs="+", default=[0])
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])  #
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    parser.add_argument('--wandb', type=str, default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument('--load_pretrained', default=None)
    parser.add_argument('--do_save', default=None)
    parser.add_argument('--timestamp', default=timestamp)
    parser.add_argument('--device', type=str, default=0)
    parser.add_argument('--cheby', default=None)
    parser.add_argument('--dynamic_name', type=str, default="")
    parser.add_argument('--div_filter', type=str, default="NoFilter", choices=["IdenticalMurckoScaffold", "NoFilter", "ScaffoldSimilarity", "IdenticalTopologicalScaffold"])
    parser.add_argument('--update_order', type=str, default=None)
    parser.add_argument('--update_num', type=int, default=2)
    parser.add_argument('--replay', type=int, default=8)
    parser.add_argument('--stitch_round', type=int, default=16)
    parser.add_argument('--self_aug_round', type=int, default=8)
    parser.add_argument('--ga_mode', type=str, default="")
    parser.add_argument('--inval', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--kl_loss', type=float, default=0)
    parser.add_argument('--min_atom_diff', type=int, default=0)
    parser.add_argument('--num_proxy', type=int, default=4)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.device}"
    os.environ["WANDB_MODE"] = args.wandb

    if not args.log_code:
        os.environ["WANDB_DISABLE_CODE"] = "false"

    args.method = args.method.lower() 

    path_main = os.path.dirname(os.path.realpath(__file__))
    path_main = os.path.join(path_main, "main", args.method)

    sys.path.append(path_main)
    
    print(args.method)
    # Add method name here when adding new ones
    if args.method == 'screening':
        from main.screening.run import Exhaustive_Optimizer as Optimizer 
    elif args.method == 'molpal':
        from main.molpal.run import MolPAL_Optimizer as Optimizer
    elif args.method == 'graph_ga':
        from main.graph_ga.run import GB_GA_Optimizer as Optimizer
    elif args.method == 'smiles_ga':
        from main.smiles_ga.run import SMILES_GA_Optimizer as Optimizer
    elif args.method == "selfies_ga":
        from main.selfies_ga.run import SELFIES_GA_Optimizer as Optimizer
    elif args.method == "synnet":
        from main.synnet.run import SynNet_Optimizer as Optimizer
    elif args.method == 'hebo':
        from main.hebo.run import HEBO_Optimizer as Optimizer 
    elif args.method == 'graph_mcts':
        from main.graph_mcts.run import Graph_MCTS_Optimizer as Optimizer
    elif args.method == 'smiles_ahc':
        from main.smiles_ahc.run import AHC_Optimizer as Optimizer
    elif args.method == 'smiles_aug_mem':
        from main.smiles_aug_mem.run import AugmentedMemory_Optimizer as Optimizer
    ################################################################################
    ################################################################################
    elif args.method == 'saturn' and args.ga_mode == "ranked":
        from main.saturn.run_ranked import Mamba_Optimizer as Optimizer
    elif args.method == 'saturn' and args.ga_mode == "ranked_mamba":
        from main.saturn.run_ranked_mamba_based import Mamba_Optimizer as Optimizer
    elif args.method == 'saturn' and args.ga_mode == "pref":
        from main.saturn.run_pref import Mamba_Optimizer as Optimizer
    elif args.method == 'saturn' and args.ga_mode == "pref_parp":
        from main.saturn.run_pref_parp import Mamba_Optimizer as Optimizer
    elif args.method == 'saturn':
        from main.saturn.run import Mamba_Optimizer as Optimizer
    elif args.method == 'augmem':
        from main.augmem.run import AugMem_Optimizer as Optimizer
    elif args.method == "offline_cluster" and args.ga_mode == "gfn":
        from main.offline_cluster.run_gfn_proxy_pref import REINVENTGame_Optimizer as Optimizer
    ################################################################################
    ################################################################################
    elif args.method == 'smiles_bar':
        from main.smiles_bar.run import BAR_Optimizer as Optimizer 
    elif args.method == "smiles_lstm_hc":
        from main.smiles_lstm_hc.run import SMILES_LSTM_HC_Optimizer as Optimizer
    elif args.method == 'selfies_lstm_hc':
        from main.selfies_lstm_hc.run import SELFIES_LSTM_HC_Optimizer as Optimizer
    elif args.method == 'dog_gen':
        from main.dog_gen.run import DoG_Gen_Optimizer as Optimizer
    elif args.method == 'gegl':
        from main.gegl.run import GEGL_Optimizer as Optimizer 
    elif args.method == 'boss':
        from main.boss.run import BOSS_Optimizer as Optimizer
    elif args.method == 'chembo':
        from main.chembo.run import ChemBOoptimizer as Optimizer 
    elif args.method == 'gpbo':
        from main.gpbo.run import GPBO_Optimizer as Optimizer
    elif args.method == 'reinvent_gp':
        from main.reinvent_gp.run_rein import GPBO_Optimizer as Optimizer
    elif args.method == 'stoned': 
        from main.stoned.run import Stoned_Optimizer as Optimizer
    elif args.method == "selfies_vae":
        from main.selfies_vae.run import SELFIES_VAEBO_Optimizer as Optimizer
    elif args.method == "smiles_vae":
        from main.smiles_vae.run import SMILES_VAEBO_Optimizer as Optimizer
    elif args.method == 'jt_vae':
        from main.jt_vae.run import JTVAE_BO_Optimizer as Optimizer
    elif args.method == 'dog_ae':
        from main.dog_ae.run import DoG_AE_Optimizer as Optimizer
    elif args.method == 'pasithea':
        from main.pasithea.run import Pasithea_Optimizer as Optimizer
    elif args.method == 'dst':
        from main.dst.run import DST_Optimizer as Optimizer        
    elif args.method == 'molgan':
        from main.molgan.run import MolGAN_Optimizer as Optimizer
    elif args.method == 'mars':
        from main.mars.run import MARS_Optimizer as Optimizer
    elif args.method == 'mimosa':
        from main.mimosa.run import MIMOSA_Optimizer as Optimizer
    elif args.method == 'gflownet':
        from main.gflownet.run import GFlowNet_Optimizer as Optimizer
    elif args.method == 'gflownet_al':
        from main.gflownet_al.run import GFlowNet_AL_Optimizer as Optimizer
    elif args.method == 'moldqn':
        from main.moldqn.run import MolDQN_Optimizer as Optimizer
    elif args.method == 'reinvent':
        from main.reinvent.run import REINVENT_Optimizer as Optimizer
    elif args.method == 'reinvent_selfies':
        from main.reinvent_selfies.run import REINVENT_SELFIES_Optimizer as Optimizer
    elif args.method == 'graphinvent':
        from main.graphinvent.run import GraphInvent_Optimizer as Optimizer
    elif args.method == "rationale_rl":
        from main.rationale_rl.run import Rationale_RL_Optimizer as Optimizer
    elif args.method == "gflownet_game":
        from main.gflownet_game.run import GFlowNetGame_Optimizer as Optimizer
    elif args.method == "genetic_gfn":
        from main.genetic_gfn.run import Genetic_GFN_Optimizer as Optimizer
    elif args.method == "genetic_gfn_al":
        from main.genetic_gfn_al.run import Genetic_GFN_AL_Optimizer as Optimizer
    elif args.method == "genetic_gfn_selfies":
        from main.genetic_gfn_selfies.run import Genetic_GFN_SELFIES_Optimizer as Optimizer
    elif args.method == "reinvent_game":
        from main.reinvent_game.run_1 import REINVENTGame_Optimizer as Optimizer
    elif args.method == "reinvent_game_stitch":
        from main.reinvent_game_stitch.run_1 import REINVENTGame_Optimizer as Optimizer
    elif args.method == "reinvent_ma":
        from main.reinvent_MA.run_1 import REINVENTGame_Optimizer as Optimizer
    elif args.method == "reinvent_bo":
        from main.reinvent_bo.run_bo import REINVENTGame_Optimizer as Optimizer
    elif args.method == "reinvent_3":
        from main.reinvent_3.run import REINVENT3_Optimizer as Optimizer
    elif args.method == "reinvent_offline":
        from main.reinvent_offline.run_1 import REINVENTGame_Optimizer as Optimizer
    elif args.method == "offline_ga":
        from main.offline_ga.run_1 import REINVENTGame_Optimizer as Optimizer
    elif args.method == "offline_augmem":
        from main.offline_augmem.run import AugmentedMemory_Optimizer as Optimizer
    elif args.method == "reinvent_cl":
        from main.reinvent_cl.run_CL import REINVENTGame_Optimizer as Optimizer
    elif args.method == "reinvent_dynamic":
        from main.reinvent_dynamic.run_DN import REINVENTDynamic_Optimizer as Optimizer
    elif args.method == "reinvent_lambo":
        from main.reinvent_lambo.run_1 import REINVENTLamBO_Optimizer as Optimizer
    elif args.method == "lambo2":
        from main.lambo2.run_1 import LamBO_Optimizer as Optimizer
    elif args.method == "lambo_selfies":
        from main.lambo_selfies.run_1 import LamBO_Optimizer as Optimizer
    elif args.method == "reinvent_game_dynamic":
        from main.reinvent_game_dynamic.run_1 import REINVENTGame_Optimizer as Optimizer
    elif args.method == "graph_ga_dynamic":
        from main.graph_ga_dynamic.run import GB_GA_Optimizer as Optimizer
    elif args.method == "smiles_aug_mem_dynamic":
        from main.smiles_aug_mem_dynamic.run_AugDN import AugmentedMemory_Optimizer as Optimizer
    else:
        raise ValueError("Unrecognized method name.")

    if args.output_dir is None:
        args.output_dir = os.path.join(path_main, "results")
        args.memory_out_dir = os.path.join(path_main, "memory")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.pickle_directory is None:
        args.pickle_directory = path_main

    if args.task != "tune":
    
        for oracle_name in args.oracles:

            print(f'Optimizing oracle function: {oracle_name}')

            try:
                config_default = yaml.safe_load(open(args.config_default))
            except:
                config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))
            if len(args.oracles[0].split('+')) == 1:
                oracle = Oracle(name=oracle_name)
            else:
                oracle = MultiOracle(name=oracle_name, cheby=args.cheby)
            optimizer = Optimizer(args=args)

            if args.task == "simple":
                # optimizer.optimize(oracle=oracle, config=config_default, seed=args.seed) 
                for seed in args.seed:
                    print('seed', seed)
                    optimizer.optimize(oracle=oracle, config=config_default, entity=WANDB_ENTITY_NAME, seed=seed, project=WANDB_PROJECT_NAME)
            elif args.task == "production":
                optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)
            else:
                raise ValueError('Unrecognized task name, task should be in one of simple, tune and production.')

    elif args.task == "tune":

        print(f'Tuning hyper-parameters on tasks: {args.oracles}')

        try:
            config_default = yaml.safe_load(open(args.config_default))
        except:
            config_default = yaml.safe_load(open(os.path.join(path_main, args.config_default)))

        try:
            config_tune = yaml.safe_load(open(args.config_tune))
        except:
            config_tune = yaml.safe_load(open(os.path.join(path_main, args.config_tune)))

        oracles = [MultiOracle(name=oracle_name) for oracle_name in args.oracles]
        optimizer = Optimizer(args=args)
        
        optimizer.hparam_tune(oracles=oracles, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)

    else:
        raise ValueError('Unrecognized task name, task should be in one of simple, tune and production.')
    end_time = time()
    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))
    print("timestamp: ", args.timestamp)
    # print('If the program does not exit, press control+c.')


if __name__ == "__main__":
    main()

