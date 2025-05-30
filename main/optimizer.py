import os
import yaml
import random
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import tdc
from tdc.generation import MolGen
import wandb
from main.utils.chem import *
from evaluators.hypervolume import get_hypervolume, get_hypervolume_pygmo, get_pareto_fronts
import subprocess

class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls


class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log
        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name='SA')
        self.diversity_evaluator = tdc.Evaluator(name='Diversity')

        # self.novelty_scorer = tdc.Evaluator(name='novelty')
        # self.uniqueness_scorer = tdc.Evaluator(name='uniqueness')
        # self.fcd_scorer = tdc.Oracle(name='fcd_distance')
        # self.validity_scorer = tdc.Evaluator(name='validity')
        self.last_log = 0

    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[
                              :self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)
        if len(self.evaluator.name.split('+')) > 1:
            if len(self.evaluator.name.split('+')) < 4:
                HV, R2 = get_hypervolume(None, self.evaluator.pareto_rewards, len(self.evaluator.name.split('+')))
            else:  # higher than 4 objective needs bunch of calculations
                HV, R2 = get_hypervolume_pygmo(None, self.evaluator.pareto_rewards, len(self.evaluator.name.split('+')))
                # HV, R2 = get_hypervolume_docker(None, self.evaluator.pareto_rewards, len(self.evaluator.name.split('+')))
            # elif n_calls == self.max_oracle_calls:
            #     HV, R2 = get_hypervolume_docker(None, self.evaluator.pareto_rewards, len(self.evaluator.name.split('+')))
            # else:
            #     HV, R2 = 0, 0
        else:
            HV, R2 = 0, 0
        # novelty_top100 = self.novelty_scorer(smis)
        # uniqueness_top100 = self.uniqueness_scorer(smis)
        # fcd_top100 = self.fcd_scorer(smis)
        # validity_top100 = self.validity_scorer(smis)

        auc_top1 = top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls)
        auc_top10 = top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls)
        auc_top100 = top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls)
        print(f'{n_calls}/{self.max_oracle_calls} | '
              f'avg_top1: {avg_top1:.3f} | '
              f'avg_top10: {avg_top10:.3f} | '
              f'avg_top100: {avg_top100:.3f} | '
              # f'auc_top1: {auc_top1:.3f} | '
              # f'auc_top10: {auc_top10:.3f} | '
              # f'auc_top100: {auc_top100:.3f} | '
              f'HV: {HV:.3f} | '
              f'R2: {R2:.3f} | '
              f'avg_sa: {avg_sa:.3f} | '
              # f'avg_nov: {novelty_top100:.3f} | '
              # f'avg_uniq: {uniqueness_top100:.3f} | '
              # f'avg_fcd: {fcd_top100:.3f} | '
              # f'avg_vali: {validity_top100:.3f} | '
              f'div: {diversity_top100:.3f}'
              )

        # try:
        wandb.log({
            "avg_top1": avg_top1,
            "avg_top10": avg_top10,
            "avg_top100": avg_top100,
            "auc_top1": auc_top1,
            "auc_top10": auc_top10,
            "auc_top100": auc_top100,
            "avg_sa": avg_sa,
            "diversity_top100": diversity_top100,
            "HV": HV,
            "R2": R2,
            # f'avg_nov': novelty_top100,
            # f'avg_uniq': uniqueness_top100,
            # f'avg_fcd': fcd_top100,
            # f'avg_vali': validity_top100,
            "n_oracle": n_calls,

            # "best_mol": wandb.Image(Draw.MolsToGridImage([Chem.MolFromSmiles(item[0]) for item in temp_top10],
            #           molsPerRow=5, subImgSize=(200,200), legends=[f"f = {item[1][0]:.3f}, #oracle = {item[1][1]}" for item in temp_top10]))
        })

    def __len__(self):
        return len(self.mol_buffer)

    def score_smi(self, smi, return_all=False):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        pass_flag = True
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass_flag = False
                pass
            else:
                if return_all:
                    avg_score, all_score = self.evaluator(return_all, smi)
                    avg_score = float(avg_score)
                else:
                    avg_score = float(self.evaluator(return_all, smi))
                if len(self.evaluator.name.split('+')) > 1:
                    temp_list = [avg_score, len(self.mol_buffer) + 1]
                    for key, value in zip(self.evaluator.temp_oracle_score.keys(),
                                          self.evaluator.temp_oracle_score.values()):
                        temp_list.append(str(key) + ": " + str(value))

                    self.mol_buffer[smi] = temp_list
                else:
                    self.mol_buffer[smi] = [avg_score, len(self.mol_buffer) + 1, str(self.name) + ": " + str(avg_score)]
            if return_all and pass_flag:
                return self.mol_buffer[smi][0], all_score
            elif return_all and not pass_flag:
                return 0
            else:
                return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst, return_all=False):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_list = []
            score_all_list = []
            for smi in smiles_lst:
                if return_all:
                    score = self.score_smi(smi, return_all=return_all)
                    if score == 0:
                        all_score = np.array([0] * len(self.evaluator.name.split('+')))
                    else:
                        score, all_score = score
                    score_list.append(score)
                    score_all_list.append(all_score)
                else:
                    score_list.append(self.score_smi(smi, return_all=return_all))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        else:  ### a string of SMILES 
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        if return_all:
            return score_list, score_all_list
        else:
            return score_list

    def input_offline_data(self, smiles_dict, return_all=True):
        smi_list = []
        score_avg_list = []
        score_all_list = []
        for smi in smiles_dict.keys():
            smi_list.append(smi)
            temp_avg_score = 0
            temp_all_score_list = []
            temp_mol_buffer = []
            for name, weight in zip(self.evaluator.name_list, self.evaluator.weight_list):

                obj_score = smiles_dict[smi][name]
                temp_all_score_list.append(obj_score)
                temp_avg_score += obj_score * weight
                temp_mol_buffer.append(str(name) + ": " + str(obj_score))

            score_avg_list.append(temp_avg_score)
            score_all_list.append(temp_all_score_list)
            molbuffer = [temp_avg_score, len(self.mol_buffer) + 1] + temp_mol_buffer
            self.mol_buffer[smi] = molbuffer


        temp_smi = np.array(smi_list)
        score = np.array(score_avg_list)
        all_score = np.array(score_all_list)
        candidates, pareto_rewards = get_pareto_fronts(temp_smi, all_score)
        self.evaluator.pareto_rewards = pareto_rewards
        self.evaluator.pareto_smiles = np.expand_dims(candidates, 1)

        self.sort_buffer()
        self.log_intermediate()
        self.last_log = len(self.mol_buffer)
        self.save_result(self.task_label)
        if return_all:
            return smi_list, score_avg_list, score_all_list
        else:
            return smi_list, score_avg_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    def __init__(self, args=None):
        self.model_name = "Default"
        self.args = args
        self.n_jobs = args.n_jobs
        # self.pool = joblib.Parallel(n_jobs=self.n_jobs)
        self.smi_file = args.smi_file
        self.oracle = Oracle(args=self.args)
        if self.smi_file is not None:
            self.all_smiles = torch.load(self.smi_file).tolist()
        else:
            data = MolGen(name='ZINC')
            self.all_smiles = data.get_data()['smiles'].tolist()

        self.sa_scorer = tdc.Oracle(name='SA')
        self.diversity_evaluator = tdc.Evaluator(name='Diversity')
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters=['PAINS', 'SureChEMBL', 'Glaxo'],
                                                             property_filters_flag=False)

    # def load_smiles_from_file(self, file_name):
    #     with open(file_name) as f:
    #         return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()
        for mol in mol_list:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    print('bad smiles')
        return new_mol_list

    def sort_buffer(self):
        self.oracle.sort_buffer()

    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)

    def log_result(self):

        print(f"Logging final results...")

        # import ipdb; ipdb.set_trace()

        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]

        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))

        # Currently logging the top-100 moelcules, will update to PDD selection later
        data = [[i + 1, results_all_level[-1][i][1][0], results_all_level[-1][i][1][1], \
                 wandb.Image(Draw.MolToImage(Chem.MolFromSmiles(results_all_level[-1][i][0]))),
                 results_all_level[-1][i][0]] for i in range(100)]
        columns = ["Rank", "Score", "#Oracle", "Image", "SMILES"]
        wandb.log({"Top 100 Molecules": wandb.Table(data=data, columns=columns)})

        # Log batch metrics at various oracle calls
        data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "Diversity", "avg_SA", "%Pass", "Top-1 Pass"]
        wandb.log({"Batch metrics at various level": wandb.Table(data=data, columns=columns)})

    def save_result(self, suffix=None):
        print(f"Saving molecules...")
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [np.mean(scores),
                np.mean(scores[:10]),
                np.max(scores),
                self.diversity_evaluator(smis),
                np.mean(self.sa_scorer(smis)),
                float(len(smis_pass) / 100),
                top1_pass]

    def reset(self):
        del self.oracle
        self.oracle = Oracle(args=self.args)

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish

    def _optimize(self, oracle, config):
        raise NotImplementedError

    def hparam_tune(self, oracles, hparam_space, hparam_default, count=5, num_runs=3, project="tune"):
        seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        seeds = seeds[:num_runs]
        hparam_space["name"] = hparam_space["name"]

        def _func():
            with wandb.init(config=hparam_default, allow_val_change=True, project="MOMAGame") as run:
                avg_auc = 0
                for oracle in oracles:
                    auc_top10s = []
                    for seed in seeds:
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        random.seed(seed)
                        config = wandb.config
                        self._optimize(oracle, config)
                        auc_top10s.append(top_auc(self.oracle.mol_buffer, 10, True, self.oracle.freq_log,
                                                  self.oracle.max_oracle_calls))
                        self.reset()
                    avg_auc += np.mean(auc_top10s)
                wandb.log({"avg_auc": avg_auc})

        sweep_id = wandb.sweep(hparam_space)
        # wandb.agent(sweep_id, function=_func, count=count, project=self.model_name + "_" + oracle.name)
        wandb.agent(sweep_id, function=_func, count=count, entity="younghan")

    def optimize(self, oracle, config, entity, seed=0,  project="test"):
        run = wandb.init(project=project, config=config, reinit=True, entity=entity)
        wandb.config.update(self.args)
        wandb.run.name = self.model_name + "_" + oracle.name + "_" + wandb.run.id
        wandb.run.log_code(f"./main/{self.args.method}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed
        self.oracle.task_label = self.model_name + "_" + oracle.name + "_" + str(seed)
        self._optimize(oracle, config)
        if self.args.log_results:
            self.log_result()
        self.save_result(f"{self.model_name}_{oracle.name}_{self.args.timestamp}_{str(seed)}")
        # self.reset()
        run.finish()
        self.reset()

    def production(self, oracle, config, num_runs=5, project="production"):
        seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        # seeds = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        if num_runs > len(seeds):
            raise ValueError(f"Current implementation only allows at most {len(seeds)} runs.")
        seeds = seeds[:num_runs]
        for seed in seeds:
            self.optimize(oracle, config, seed, project)
            self.reset()


def chebyshev_scalarization_batch(f_values, weights=None, reference_point=None):
    """
    Computes the Chebyshev scalarization for objective function values using NumPy,
    with the goal of maximizing the function values. Handles both single input and batch input.

    :param f_values: A NumPy array of objective function values. Can be 1D (single input) or 2D (batch input).
    :param weights: A NumPy array of weights for each objective function.
    :param reference_point: A NumPy array representing the reference point, typically the ideal maximum values.

    :return: A NumPy array of Chebyshev scalarization values.
    """
    # Check if the input is a single data point (1D array)
    if reference_point is None:
        reference_point = np.array([1, 1, 1, 1])
    if f_values.ndim == 1:
        # Convert the input to a 2D array with a single row
        f_values = np.array([f_values])

    # Calculate the weighted absolute differences
    weighted_diffs = weights * np.abs(f_values - reference_point)

    # Return the maximum of these weighted differences for each data point
    return np.max(weighted_diffs, axis=1)

