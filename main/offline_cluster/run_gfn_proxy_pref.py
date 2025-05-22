import os
import sys
import numpy as np
from dacite import from_dict
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
sys.path.append(path_here + "/diversity_filters")
from diversity_filters.filters import DiversityFilter
from diversity_filters.component_summary import DiversityFilterParameters, FinalSummary
from diversity_filters.conversions import Conversions
from main.optimizer import BaseOptimizer, Objdict
from utils import Variable, seq_to_smiles, unique, get_unique_list_indices, get_randomized_smiles_without_prior, \
    extract_scores, generate_non_overlapping_indices
from model import RNN, ProxyOracle, RankModel

from data_structs import Vocabulary, Experience, MolData
import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
import main.graph_ga.crossover as co, main.graph_ga.mutate as mu

import random
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

MINIMUM = 1e-10



def smiles_f1_f2_encoded(smiles, scores, voc: Vocabulary, stitch_round=False):
    mating_mols = Conversions.smiles_to_mols(smiles)
    tokens = []
    tokens1 = []
    tokens2 = []
    scores_list = []
    for index, mol in enumerate(mating_mols):
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except ValueError:
            pass
        non = co.crossover_return_fragment(mol, mol, return_all=True)
        if non is None:
            continue
        frag1s, frag2s, new_mols = non
        if stitch_round is True:
            random_index = np.random.randint(0, len(new_mols))
            new_mol, idx = new_mols[random_index], random_index
        else:
            closest_offspring = co.return_max_sim_offspring(mol, new_mols)
            new_mol, idx, _, _, _, _ = closest_offspring

        smi = Chem.MolToSmiles(new_mol, canonical=False)
        f1 = Chem.MolToSmiles(co.remove_special_atoms(frag1s[idx]), canonical=False)
        f2 = Chem.MolToSmiles(co.remove_special_atoms(frag2s[idx]), canonical=False)
        tokens.append(voc.tokenize(smi))
        tokens1.append(voc.tokenize(f1))
        tokens2.append(voc.tokenize(f2))
        scores_list.append(scores[index])
    encs, encs1, encs2, sc_list = [], [], [], []
    for tok, f1, f2, sc in zip(tokens, tokens1, tokens2, scores_list):
        try:
            enc = Variable(voc.encode(tok))
            enc1 = Variable(voc.encode(f1))
            enc2 = Variable(voc.encode(f2))
            encs.append(enc)
            encs1.append(enc1)
            encs2.append(enc2)
            sc_list.append(sc)
        except:
            continue
    encoded = MolData.collate_fn(encs).long()
    encoded1 = MolData.collate_fn(encs1).long()
    encoded2 = MolData.collate_fn(encs2).long()
    return encoded, encoded1, encoded2, np.array(sc_list)


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float = 0.1,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


class REINVENTGame_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "GFNProxyPref"


    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        self.obj_dim = len(oracle.name.split('+'))
        config = Objdict(config)

        # pool = Parallel(n_jobs=self.n_jobs)
        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from = os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from = restore_prior_from
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        offline_dataset = torch.load(os.path.join(path_here, f"data/offline_dataset/MolStitch_offline_dataset{self.args.seed}.pt"))

        Prior = RNN(voc)
        Agent = RNN(voc)
        Prior_stitch = RNN(voc)
        Agent_stitch = RNN(voc)

        # for score-based proxy model
        # hidden_dim = 512  # Example hidden layer size
        # output_dim = 1  # Output dimension (single value)
        # feature_dim = 128  # Feature dimension (fixed)
        # proxy_oracle = RankModel(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim).cuda()
        # proxy_oracle = RNN(voc)

        num_proxy = self.args.num_proxy
        proxy_oracle_list = [RNN(voc) for _ in range(num_proxy)]
        p_alpha = torch.tensor([3.0] * self.obj_dim)
        dirichlet_dist = torch.distributions.Dirichlet(p_alpha)
        if num_proxy == 1:
            proxy_alpha_list = [np.array([1.0] * self.obj_dim)]
        else:
            proxy_alpha_list = [dirichlet_dist.sample().cpu().numpy() for _ in range(num_proxy)]
        # proxy_optimizer = torch.optim.Adam(proxy_oracle.rnn.parameters(), lr=config['learning_rate'], weight_decay=1e-04)
        proxy_optimizer_list = []
        for proxy_oracle in proxy_oracle_list:
            optimizer = torch.optim.Adam(proxy_oracle.rnn.parameters(), lr=config['learning_rate'], weight_decay=1e-04)
            proxy_optimizer_list.append(optimizer)
        diversity_config = from_dict(data_class=DiversityFilterParameters, data=config.diversity_filter)
        diversity_config.name = self.args.div_filter
        diversity_filter = DiversityFilter(diversity_config)
        weight_list = np.array(self.oracle.evaluator.weight_list)

        exp_dir = f'{config.save_path}'
        if self.args.do_save:
            do_save = True
            os.makedirs(exp_dir, exist_ok=True)
        else:
            do_save = False

        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt')), strict=False)
            Prior_stitch.rnn.load_state_dict(
                torch.load(os.path.join(path_here, 'data/prior_co/Prior_CO_9_1500_ 9_94.ckpt')), strict=False)
            Agent.rnn.load_state_dict(torch.load(restore_agent_from), strict=False)
            Agent_stitch.rnn.load_state_dict(
                torch.load(os.path.join(path_here, 'data/prior_co/Prior_CO_9_1500_ 9_94.ckpt')), strict=False)

        else:
            Prior.rnn.load_state_dict(
                torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage), strict=False)
            Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage), strict=False)

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False
        for param in Prior_stitch.rnn.parameters():
            param.requires_grad = False
        # We extremely reduced log Z and its learning rate from original code for stability
        log_z = torch.nn.Parameter(torch.tensor([0.001]).cuda())
        optimizer = torch.optim.Adam([{'params': Agent.rnn.parameters(),
                                        'lr': config['learning_rate']},
                                    {'params': log_z,
                                        'lr': config['learning_rate']}])

        # optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])
        optimizer_stitch = torch.optim.Adam(Agent_stitch.rnn.parameters(), lr=config['learning_rate'])

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefore not as theoretically sound as it is for value based RL, but it seems to work well.
        experience = Experience(voc, max_size=1000)
        experience.final_weight = weight_list
        experience_proxy = Experience(voc, max_size=1000)
        experience_proxy.final_weight = weight_list

        # can be loaded from csv smi file

        print("Model initialized, starting training...")

        step = 0

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()

            # Sample from Agent

            offline_smiles_batch = []
            offline_canon_smiles = []
            offline_scaffold = []
            offline_invalid_smiles = []
            unique_nums = 0
            valid_nums = 0
            # We use offline dataset in the first step
            if step == 0:
                offline_smiles_batch, score, all_score = self.oracle.input_offline_data(offline_dataset)
            else:
                with torch.inference_mode():
                    while len(offline_smiles_batch) < self.args.offline_batch_size*2:
                        seqs, agent_likelihood, _ = Agent.sample(config['batch_size'])
                        # Remove duplicates, ie only consider unique seqs
                        unique_idxs = unique(seqs)
                        seqs = seqs[unique_idxs]
                        # agent_likelihood = agent_likelihood[unique_idxs]
                        uni_smiles = seq_to_smiles(seqs, voc)
                        molecules, valid_indices = Conversions.smiles_to_mols_and_indices(uni_smiles)

                        unique_nums += len(seqs)
                        valid_nums += len(valid_indices)
                        invalid_indices = [i for i in range(len(seqs)) if i not in valid_indices]
                        invalid_smiles = [uni_smiles[i] for i in invalid_indices]
                        offline_invalid_smiles.extend(invalid_smiles)

                        canon_smiles = [Chem.MolToSmiles(m, canonical=True) for m in molecules]
                        uni_smiles = [uni_smiles[i] for i in valid_indices]
                        seqs, agent_likelihood = seqs[valid_indices], agent_likelihood[valid_indices]
                        unique_list_idx = get_unique_list_indices(canon_smiles)
                        canon_smiles = [canon_smiles[i] for i in unique_list_idx]
                        uni_smiles = [uni_smiles[i] for i in unique_list_idx]
                        seqs, agent_likelihood = seqs[unique_list_idx], agent_likelihood[unique_list_idx]
                        not_in_buffer_indices = [i for i, smile in enumerate(canon_smiles) if
                                                 smile not in self.oracle.mol_buffer]
                        smiles = [uni_smiles[i] for i in not_in_buffer_indices]
                        canon_smiles = [canon_smiles[i] for i in not_in_buffer_indices]
                        if diversity_config.name != 'NoFilter':
                            smiles, scaffold_list, survive_idx = diversity_filter.filter_by_scaffold(smiles)
                            seqs, agent_likelihood = seqs[survive_idx], agent_likelihood[survive_idx]
                            canon_smiles = [canon_smiles[i] for i in survive_idx]
                            offline_scaffold.extend(scaffold_list)

                        offline_smiles_batch.extend(smiles)
                        offline_canon_smiles.extend(canon_smiles)

                oracle_indexes = []
                score = []
                all_score = []
                print("model validity: ", round(valid_nums/unique_nums * 100, 2))
                for i, smi in enumerate(offline_canon_smiles):
                    prev_len = len(self.oracle)
                    score_single, all_score_single = self.oracle([smi], return_all=True)
                    if len(self.oracle) > prev_len:
                        oracle_indexes.append(i)
                        score.append(score_single[0])
                        all_score.append(all_score_single[0])
                    if len(self.oracle) == (step+1) * self.args.offline_batch_size:
                        break
                offline_smiles_batch = [offline_smiles_batch[i] for i in oracle_indexes]
                offline_canon_smiles = [offline_canon_smiles[i] for i in oracle_indexes]

            non_zero_indices = [index for index, value in enumerate(score) if value > 1e-6]
            log = ""
            for i, name in enumerate(self.oracle.evaluator.name_list):
                if len(non_zero_indices) != 0:
                    log += f"{name}: {np.mean([all_score[j] for j in non_zero_indices], axis=0)[i].item():3f} "
            print(log)

            if diversity_config.name != 'NoFilter':
                offline_scaffold = offline_scaffold[:self.args.offline_batch_size]
                diversity_filter.add_with_filtered(FinalSummary(score, offline_smiles_batch, None), offline_scaffold,
                                                   step)
            all_score = np.array(all_score)
            score = np.array(score)

            if self.finish:
                print('max oracle hit')
                break

                # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()

            # Batch update Generative model in here.

            indices = np.random.permutation(len(offline_smiles_batch))
            shuffled_smiles = [offline_smiles_batch[i] for i in indices]
            shuffled_score = score[indices]
            shuffled_all_score = all_score[indices]
            batch_size = 500
            batch_size += 1
            smiles_batches = [shuffled_smiles[i:i + batch_size] for i in range(0, len(shuffled_smiles), batch_size)]
            score_batches = [shuffled_score[i:i + batch_size] for i in range(0, len(shuffled_score), batch_size)]
            all_score_batches = [shuffled_all_score[i:i + batch_size] for i in
                                 range(0, len(shuffled_all_score), batch_size)]
            # update all

            for _ in range(2):
                for smiles_batch, score_batch, all_score_batch in zip(smiles_batches, score_batches, all_score_batches):
                    offline_seqs = []
                    offline_smiles_tokenized = [experience.voc.tokenize(smile) for smile in smiles_batch]
                    for ind, tokenized_i in enumerate(offline_smiles_tokenized):
                        enc = Variable(experience.voc.encode(tokenized_i))
                        offline_seqs.append(enc)
                    offline_seqs = MolData.collate_fn(offline_seqs)
                    offline_prior_likelihood, _ = Prior.likelihood(Variable(offline_seqs.long()))
                    offline_agent_likelihood, _ = Agent.likelihood(Variable(offline_seqs.long()))

                    reward = torch.tensor(score_batch).cuda()

                    # Modified from original GeneticGFN.
                    exp_forward_flow = offline_agent_likelihood + log_z - offline_prior_likelihood
                    exp_backward_flow = reward * 500

                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    # KL penalty. Original REINVENT's (agent - prior) actually behaves like kl penalty
                    # loss_p = (offline_agent_likelihood - offline_prior_likelihood).mean()
                    # loss += 0.1 * loss_p


                    optimizer.zero_grad()
                    loss.backward()
                    # grad_norms = torch.nn.utils.clip_grad_norm_(Agent.rnn.parameters(), 1.0)
                    optimizer.step()
                    # You can reduce the likelihood of invalid sequences generated by Generator
                    if self.args.inval != 0:
                        inval_seqs = []
                        indices = np.random.permutation(len(offline_invalid_smiles))
                        inval_smis = [offline_invalid_smiles[i] for i in indices]
                        inval_smis = inval_smis[:self.args.inval]
                        offline_inval_tokenized = [experience.voc.tokenize(smile) for smile in inval_smis]
                        for ind, tokenized_i in enumerate(offline_inval_tokenized):
                            try:
                                enc = Variable(experience.voc.encode(tokenized_i))
                                inval_seqs.append(enc)
                            except:
                                continue
                        inval_seqs = MolData.collate_fn(inval_seqs)
                        inval_prior_likelihood, _ = Prior.likelihood(Variable(inval_seqs.long()))
                        inval_agent_likelihood, _ = Agent.likelihood(Variable(inval_seqs.long()))
                        inval_loss = torch.pow((inval_prior_likelihood.float() - inval_agent_likelihood), 2)
                        loss = inval_loss
                        loss = loss.mean()

                        # Calculate gradients and make an update to the network weights
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Then add new experience
                    offline_prior_likelihood = offline_prior_likelihood.data.cpu().numpy()
                    new_experience = zip(smiles_batch, all_score_batch, offline_prior_likelihood)
                    experience.add_experience(new_experience)
                    experience_proxy.add_experience(zip(smiles_batch, all_score_batch, offline_prior_likelihood))


            ########################## Need to shuffle offline_seqs_batches
            for _ in range(20):
                for proxy_oracle, proxy_optimizer, p_pref in zip(proxy_oracle_list, proxy_optimizer_list, proxy_alpha_list):
                    # each proxy has its own fixed score preferences.
                    offline_seqs_all, shuffled_score, _ = experience_proxy.sample_pref(200, pref=np.array([1.0]*self.obj_dim), score_pref=p_pref)
                    offline_seqs_batches = [offline_seqs_all[i:i + batch_size] for i in range(0, len(offline_seqs_all), batch_size)]
                    score_batches = [shuffled_score[i:i + batch_size] for i in range(0, len(shuffled_score), batch_size)]

                    for offline_seqs, score_batch in zip(offline_seqs_batches, score_batches):
                        indices1, indices2 = generate_non_overlapping_indices(len(offline_seqs), device=offline_seqs.device)

                        proxy_optimizer.zero_grad()
                        predicted_score, _ = proxy_oracle.likelihood(offline_seqs.long())
                        predicted_score1 = predicted_score[indices1]
                        predicted_score2 = predicted_score[indices2]

                        p = torch.sigmoid(predicted_score1 - predicted_score2)

                        true_score1 = torch.tensor(score_batch).float().cuda()[indices1]
                        true_score2 = torch.tensor(score_batch).float().cuda()[indices2]
                        rank = (true_score1 > true_score2).float()  # Convert to float tensor for BCELoss

                        loss = torch.nn.BCELoss()(p, rank)
                        loss.backward()
                        proxy_optimizer.step()

            for _ in range(self.args.replay):
                if config['experience_replay'] and len(experience) > config['experience_replay']:  # experience replay 24
                    # exp_seqs, exp_score, exp_prior_likelihood = experience.sample(config['experience_replay'])
                    exp_seqs, exp_score = experience.rank_based_sample(100, rank_coefficient=0.01)

                    exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
                    exp_prior_likelihood, exp_entropy = Prior.likelihood(exp_seqs.long())
                    reward = torch.tensor(exp_score).cuda()

                    exp_forward_flow = exp_agent_likelihood + log_z - exp_prior_likelihood
                    exp_backward_flow = reward * 500

                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            ###### Pareto Experience Replay ######
            # if config['experience_replay'] and len(experience.pareto_memory) > config['experience_replay']:  # experience replay 24
            #     exp_seqs, exp_score, exp_prior_likelihood = experience.pareto_sample(config['experience_replay'] // 2)
            #     exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            #     exp_augmented_likelihood = exp_prior_likelihood + config['sigma'] * exp_score
            #     exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            #     loss = torch.cat((loss, exp_loss), 0)
            #     offline_agent_likelihood = torch.cat((offline_agent_likelihood, exp_agent_likelihood), 0)
            alpha = torch.tensor([1.0] * self.obj_dim)
            dirichlet_dist = torch.distributions.Dirichlet(alpha)
            self_augmentation_round = self.args.self_aug_round
            # if step > 1:
            #     self_augmentation_round = 0
            for _ in range(self_augmentation_round):
                self.oracle.sort_buffer()
                pop_smis, pop_scores = tuple(
                    map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))
                pop_smis, pop_scores = pop_smis[:config['population_size']], pop_scores[:config['population_size']]
                all_score_list = extract_scores([elem[2:] for (smi, elem) in self.oracle.mol_buffer.items()])
                all_score_tensor = torch.tensor(all_score_list[:config['population_size']])
                pref1 = dirichlet_dist.sample()
                pref_score1 = (torch.matmul(all_score_tensor, pref1) + MINIMUM).tolist()
                sum_pref1 = sum(pref_score1)
                population_probs1 = [p / sum_pref1 for p in pref_score1]

                population_scores = [s + MINIMUM for s in pop_scores]
                sum_scores = sum(population_scores)
                population_probs = [p / sum_scores for p in population_scores]
                # mating_index = np.random.choice(list(range(len(pop_smis))), p=population_probs, size=50, replace=False)
                mating_index = np.random.choice(list(range(len(pop_smis))), p=population_probs1, size=50, replace=False)
                pop_smis = [pop_smis[idx] for idx in mating_index]

                mating_scores = [pop_scores[idx] for idx in mating_index]

                encoded, encoded_f1, encoded_f2, aug_scores = smiles_f1_f2_encoded(pop_smis, mating_scores, voc, stitch_round=False)

                with torch.inference_mode():
                    p_h1 = Prior_stitch.likelihood_h_out(encoded_f1)
                    p_h2 = Prior_stitch.likelihood_h_out(encoded_f2)
                    prior_likelihood, _ = Prior_stitch.likelihood_given_h(encoded, (p_h1 + p_h2) / 2)

                a_h1 = Agent_stitch.likelihood_h_out(encoded_f1)
                a_h2 = Agent_stitch.likelihood_h_out(encoded_f2)
                agent_likelihood, _ = Agent_stitch.likelihood_given_h(encoded, (a_h1 + a_h2) / 2)

                augmented_likelihood = prior_likelihood.float() + config['sigma'] * Variable(aug_scores).float()
                loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
                loss = loss.mean()
                loss_p = - (1 / agent_likelihood).mean()
                loss += 5 * 1e3 * loss_p

                loss /= 2

                # Calculate gradients and make an update to the network weights
                optimizer_stitch.zero_grad()
                loss.backward()
                optimizer_stitch.step()


            ############################################################
            stitch_round = self.args.stitch_round
            # if step > 1:
            #     stitch_round = 0
            mating_pool_size = 250
            sampling_size = 250
            unique_nums = 0
            valid_nums = 0
            alpha = torch.tensor([1.0] * self.obj_dim)
            dirichlet_dist = torch.distributions.Dirichlet(alpha)
            for _ in range(stitch_round):
                self.oracle.sort_buffer()
                pop_smis, pop_scores = tuple(
                    map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))
                all_score_list = extract_scores([elem[2:] for (smi, elem) in self.oracle.mol_buffer.items()])
                pop_smis, pop_scores = pop_smis[:config['population_size']], pop_scores[:config['population_size']]
                all_score_tensor = torch.tensor(all_score_list[:config['population_size']])
                pref1 = dirichlet_dist.sample()
                pref2 = dirichlet_dist.sample()
                pref_score1 = (torch.matmul(all_score_tensor, pref1) + MINIMUM).tolist()
                pref_score2 = (torch.matmul(all_score_tensor, pref2) + MINIMUM).tolist()
                sum_pref1 = sum(pref_score1)
                sum_pref2 = sum(pref_score2)
                population_probs1 = [p / sum_pref1 for p in pref_score1]
                population_probs2 = [p / sum_pref2 for p in pref_score2]

                # pop_smis, pop_scores = pop_smis, pop_scores
                # population_scores = [s + MINIMUM for s in pop_scores]
                # sum_scores = sum(population_scores)
                # population_probs = [p / sum_scores for p in population_scores]
                mating_index1 = np.random.choice(list(range(len(pop_smis))), p=population_probs1, size=mating_pool_size, replace=False)
                mating_index2 = np.random.choice(list(range(len(pop_smis))), p=population_probs2, size=mating_pool_size, replace=False)
                pop_smis1 = [pop_smis[idx] for idx in mating_index1]
                pop_smis2 = [pop_smis[idx] for idx in mating_index2]
                mating_scores1 = [pop_scores[idx] for idx in mating_index1]
                mating_scores2 = [pop_scores[idx] for idx in mating_index2]
                _, encoded_f1, _, aug_scores1 = smiles_f1_f2_encoded(pop_smis1, mating_scores1, voc, stitch_round=False)
                _, _, encoded_f2, aug_scores2 = smiles_f1_f2_encoded(pop_smis2, mating_scores2, voc, stitch_round=False)

                population_scores1 = [s + MINIMUM for s in aug_scores1]
                population_scores2 = [s + MINIMUM for s in aug_scores2]
                population_probs1 = [p / sum(population_scores1) for p in population_scores1]
                population_probs2 = [p / sum(population_scores2) for p in population_scores2]
                mating_index1 = np.random.choice(list(range(len(encoded_f1))), p=population_probs1, size=sampling_size, replace=True)
                mating_index2 = np.random.choice(list(range(len(encoded_f2))), p=population_probs2, size=sampling_size, replace=True)

                with torch.inference_mode():
                    a_h1 = Agent_stitch.likelihood_h_out(encoded_f1[mating_index1])
                    a_h2 = Agent_stitch.likelihood_h_out(encoded_f2[mating_index2])

                    seqs, agent_likelihood, _ = Agent_stitch.sample_from_h(sampling_size, (a_h1 + a_h2) / 2)

                seqs, agent_likelihood = seqs.clone(), agent_likelihood.clone()

                uni_smiles = seq_to_smiles(seqs, voc)
                molecules, valid_indices = Conversions.smiles_to_mols_and_indices(uni_smiles)
                unique_nums += len(uni_smiles)
                valid_nums += len(valid_indices)
                seqs, agent_likelihood_stitch = seqs[valid_indices], agent_likelihood[valid_indices]
                uni_smiles = [uni_smiles[i] for i in valid_indices]

                with torch.inference_mode():
                    greater_mask_list = []
                    lesser_mask_list = []
                    indices1, indices2 = generate_non_overlapping_indices(len(seqs), device=seqs.device)
                    for proxy_oracle in proxy_oracle_list:
                        predicted_score, _ = proxy_oracle.likelihood(seqs)
                        predicted_score1 = predicted_score[indices1]
                        predicted_score2 = predicted_score[indices2]
                        greater_mask = predicted_score1 > predicted_score2
                        lesser_mask = predicted_score1 <= predicted_score2
                        greater_mask_list.append(greater_mask)
                        lesser_mask_list.append(lesser_mask)
                    greater_mask = torch.stack(greater_mask_list).long().sum(0) > (num_proxy / 2)
                    lesser_mask = torch.stack(lesser_mask_list).long().sum(0) > (num_proxy / 2)
                    greater_indices = torch.cat([indices1[greater_mask], indices2[lesser_mask]], dim=0)
                    lesser_indices = torch.cat([indices1[lesser_mask], indices2[greater_mask]], dim=0)


                with torch.inference_mode():
                    prior_likelihood, _ = Prior.likelihood(seqs.clone())  # -45 mean
                agent_likelihood, _ = Agent.likelihood(seqs)  # -45 mean
                # agent_likelihood = - agent_likelihood
                # prior_likelihood = - prior_likelihood
                # weights = torch.sigmoid(winner_score - loser_score)

                # Compute the DPO loss
                # chosen_likeness = agent_likelihood[greater_indices.clone()] / agent_likelihood.mean()
                # rejected_likeness = agent_likelihood[lesser_indices.clone()] / agent_likelihood.mean()
                agent_chosen_likeness = agent_likelihood[greater_indices.clone()]
                prior_chosen_likeness = prior_likelihood[greater_indices.clone()]
                agent_rejected_likeness = agent_likelihood[lesser_indices.clone()]
                prior_rejected_likeness = prior_likelihood[lesser_indices.clone()]

                dpo_loss, chosen_reward, rejected_reward = preference_loss(agent_chosen_likeness + log_z,
                                                                           agent_rejected_likeness + log_z,
                                                                           prior_chosen_likeness,
                                                                           prior_rejected_likeness,
                                                                           beta=0.2,
                                                                           label_smoothing=0.2,
                                                                           ipo=True,
                                                                           reference_free=False)
                dpo_loss = dpo_loss.mean() * 0.5
                optimizer.zero_grad()
                dpo_loss.backward()
                optimizer.step()

                print("dpo_loss: ", dpo_loss.item(), "chosen_likelihood: ", chosen_reward.mean().item(), "rejected_likelihood: ", rejected_reward.mean().item())

                # augmented_likelihood = prior_likelihood + config['sigma'] * predicted_score
                # # augmented_likelihood = prior_likelihood + config['sigma'] * torch.ones_like(predicted_score, device=predicted_score.device)
                # loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
                # loss = loss.mean()
                # print(f"stitch_aug loss: {loss.item()}")
                # loss_p = - (1 / agent_likelihood).mean()
                # loss += 5 * 1e3 * loss_p

                # Calculate gradients and make an update to the network weights

            print("stitch validity: ", round(valid_nums / unique_nums * 100, 2))
            # sorted_data = sorted(avg_score, reverse=True)
            # print(f"average stitch_score: {sum(avg_score) / len(avg_score):.4f} top1: {sorted_data[0]} top10: {sum(sorted_data[:10]) / 10:.4f}")


            step += 1

        # if do_save:
        #     save_stuff()
        print(f"timestamp: {self.args.timestamp}")
        print('Done.')

        def numpy_to_list(data):
            if isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, np.float64):
                return data.item()
            elif isinstance(data, np.float32):
                return data.tolist()
            elif isinstance(data, dict):
                return {key: numpy_to_list(value) for key, value in data.items()}
            elif isinstance(data, tuple):
                return [numpy_to_list(x) for x in data]
            elif isinstance(data, list):
                return [numpy_to_list(x) for x in data]
            else:
                return data



        data = {"memory": experience.memory,
                "pareto_memory": experience.pareto_memory,
                "diversity_filter": diversity_filter.get_memory_as_dataframe().to_dict(orient='list')
                }

        converted_data = numpy_to_list(data)

        if not os.path.exists(self.args.memory_out_dir):
            os.mkdir(self.args.memory_out_dir)
        memory = os.path.join(self.args.memory_out_dir, self.args.timestamp + '.yaml')
        with open(memory, 'w') as f:
            yaml.dump(converted_data, f, sort_keys=False)


