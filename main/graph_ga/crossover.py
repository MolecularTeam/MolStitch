import random

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
rdBase.DisableLog('rdApp.error')
from rdkit.DataStructs import TanimotoSimilarity, DiceSimilarity, ConvertToNumpyArray


def cut(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
        return None

    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')))  # single bond not in ring

    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return None

    return None


def cut_ring(mol):

    for i in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')))
            bis = ((bis[0], bis[1]), (bis[2], bis[3]),)
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')))
            bis = ((bis[0], bis[1]), (bis[1], bis[2]),)

        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)])

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) == 2:
                return fragments
        except ValueError:
            return None

    return None


def ring_OK(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]')):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts('[R]=[R]=[R]'))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]'))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


# TODO: set from main? calculate for dataset?
average_size = 39.15
size_stdev = 3.50


def mol_ok(mol):
    try:
        Chem.SanitizeMol(mol)
        target_size = size_stdev * np.random.randn() + average_size  # parameters set in GA_mol
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except ValueError:
        return False


def crossover_ring(parent_A, parent_B, return_fragment=False, return_all=False):
    ring_smarts = Chem.MolFromSmarts('[R]')
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']

    for i in range(10):
        fragments_A = cut_ring(parent_A)
        fragments_B = cut_ring(parent_B)
        fragAs = []
        fragBs = []
        if fragments_A is None or fragments_B is None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            # new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_B:
                    new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])
                    fragAs.append(fa)
                    fragBs.append(fb)
        new_mols = []
        new_a = []
        new_b = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for idx, m in enumerate(new_mol_trial):
                m = m[0]
                if mol_ok(m):
                    mmm = list(rxn2.RunReactants((m,)))
                    new_mols += mmm
                    for _ in range(len(list(mmm))):
                        new_a.append(fragAs[idx])
                        new_b.append(fragBs[idx])
        new_a2 = []
        new_b2 = []
        new_mols2 = []
        for idx, m in enumerate(new_mols):
            m = m[0]
            if mol_ok(m) and ring_OK(m):
                new_mols2.append(m)
                new_a2.append(new_a[idx])
                new_b2.append(new_b[idx])

        if len(new_mols2) > 0:
            if return_fragment:
                if return_all:
                    return new_a2, new_b2, new_mols2
                random_idx = random.choice(range(len(new_mols2)))
                return new_a2[random_idx], new_b2[random_idx], new_mols2[random_idx]
            else:
                return random.choice(new_mols2)

    return None


def crossover_ring_return_all(parent_A, parent_B):
    ring_smarts = Chem.MolFromSmarts('[R]')
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']

    for i in range(10):
        fragAs = []
        fragBs = []
        fragments_A = cut_ring(parent_A)
        fragments_B = cut_ring(parent_B)

        if fragments_A is None or fragments_B is None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            # new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_B:
                    new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])
                    fragAs.append(fa)
                    fragBs.append(fb)
        new_mols = []
        new_a = []
        new_b = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for idx, m in enumerate(new_mol_trial):
                m = m[0]
                if mol_ok(m):
                    new_mols += list(rxn2.RunReactants((m,)))
                    new_a.append(fragAs[idx])
                    new_b.append(fragBs[idx])
        new_a2 = []
        new_b2 = []
        new_mols2 = []
        for idx, m in enumerate(new_mols):
            m = m[0]
            if mol_ok(m) and ring_OK(m):
                new_mols2.append(m)
                new_a2.append(new_a[idx])
                new_b2.append(new_b[idx])

        if len(new_mols2) > 0:

            return new_mols2

    return None


def crossover_ring_get_jointed_part(parent_A, parent_B):
    ring_smarts = Chem.MolFromSmarts('[R]')
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']

    for i in range(10):
        fragments_A = cut_ring(parent_A)
        fragments_B = cut_ring(parent_B)

        if fragments_A is None or fragments_B is None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            for fa in fragments_A:
                for fb in fragments_B:
                    # Add atom mapping
                    fa = Chem.RenumberAtoms(fa, list(range(fa.GetNumAtoms())))
                    for atom in fa.GetAtoms():
                        atom.SetAtomMapNum(atom.GetIdx() + 1)
                    fb = Chem.RenumberAtoms(fb, list(range(fb.GetNumAtoms())))
                    for atom in fb.GetAtoms():
                        atom.SetAtomMapNum(atom.GetIdx() + 1 + fa.GetNumAtoms())
                    try:
                        new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])
                    except:
                        continue

        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_ok(m):
                    new_mols += list(rxn2.RunReactants((m,)))

        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if mol_ok(m) and ring_OK(m):
                new_mols2.append(m)

        if len(new_mols2) > 0:
            final_mol = new_mols2[0]  # Return the first valid new molecule
            # Highlight the atoms with atom mapping to identify the combined parts
            combined_part = [atom.GetAtomMapNum() for atom in final_mol.GetAtoms() if atom.GetAtomMapNum() > 0]
            return final_mol, combined_part

    return None


def crossover_ring_self(parent_A):
    ring_smarts = Chem.MolFromSmarts('[R]')
    if not parent_A.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']

    for i in range(10):
        fragments_A = cut_ring(parent_A)

        if fragments_A is None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            # new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_A:
                    if fa != fb:
                        new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_ok(m):
                    new_mols += list(rxn2.RunReactants((m,)))

        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if mol_ok(m) and ring_OK(m):
                new_mols2.append(m)

        if len(new_mols2) > 0:

            return new_mols2

    return None


def crossover_ring_self_fragments(parent_A, min_atom_diff=False, return_all=False):
    ring_smarts = Chem.MolFromSmarts('[R]')
    if not parent_A.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']

    for i in range(10):
        fragments_A = cut_ring(parent_A)

        if fragments_A is None:
            return None

        new_mol_trial = []
        frags1 = []
        frags2 = []
        clean_frags1 = []
        clean_frags2 = []
        final_frags1 = []
        final_frags2 = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            # new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_A:
                    if fa != fb:
                        new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])
                        frags1.append(fa)
                        frags2.append(fb)

        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for idx, m in enumerate(new_mol_trial):
                m = m[0]
                if mol_ok(m):
                    r = list(rxn2.RunReactants((m,)))
                    num_reacts = len(r)
                    new_mols += r
                    for _ in range(num_reacts):
                        clean_frags1.append(frags1[idx])
                        clean_frags2.append(frags2[idx])
        new_mols2 = []
        for idx, m in enumerate(new_mols):
            m = m[0]
            if mol_ok(m) and ring_OK(m):
                new_mols2.append(m)
                final_frags1.append(clean_frags1[idx])
                final_frags2.append(clean_frags2[idx])
        if len(new_mols2) > 0:
            if min_atom_diff:
                best_pair = find_most_similar_fragments(new_mols2, final_frags1, final_frags2)
                return best_pair
            else:
                if return_all:
                    return [new_mols2, final_frags1, final_frags2]
                else:
                    return [new_mols2[0], final_frags1[0], final_frags2[0]]

    return None




def crossover_non_ring(parent_A, parent_B, return_fragment=False, return_all=False):

    for i in range(10):
        fragAs = []
        fragBs = []
        fragments_A = cut(parent_A)
        fragments_B = cut(parent_B)
        if fragments_A is None or fragments_B is None:
            return None
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        new_mol_trial = []
        for fa in fragments_A:
            for fb in fragments_B:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])
                fragAs.append(fa)
                fragBs.append(fb)

        new_mols = []
        new_fragAs = []
        new_fragBs = []
        for idx, mol in enumerate(new_mol_trial):
            mol = mol[0]
            if mol_ok(mol):
                new_mols.append(mol)
                new_fragAs.append(fragAs[idx])
                new_fragBs.append(fragBs[idx])

        if len(new_mols) > 0:
            if return_fragment:
                if return_all:
                    return new_fragAs, new_fragBs, new_mols
                random_idx = random.choice(range(len(new_mols)))
                return new_fragAs[random_idx], new_fragBs[random_idx], new_mols[random_idx]
            else:
                return random.choice(new_mols)

    return None


def crossover(parent_A, parent_B, return_fragment=False):
    parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)

    except ValueError:
        pass

    for i in range(10):
        if random.random() <= 0.1:
            # print 'non-ring crossover'
            new_mol = crossover_non_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol
        else:
            # print 'ring crossover'
            new_mol = crossover_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol

    return None


def crossover_return_fragment(parent_A, parent_B, return_all=False):
    parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)

    except ValueError:
        pass

    for i in range(10):
        if random.random() <= 0.1:
            # print 'non-ring crossover'
            non = crossover_non_ring(parent_A, parent_B, return_fragment=True, return_all=return_all)
            if non is not None:
                new_frag1, new_frag2, new_mol = non
                if return_all:
                    return non
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return non
        else:
            # print 'ring crossover'
            non = crossover_ring(parent_A, parent_B, return_fragment=True, return_all=return_all)
            if non is not None:
                new_frag1, new_frag2, new_mol = non
                if return_all:
                    return non
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return non

    return None






def remove_special_atoms(mol):
    em = Chem.EditableMol(mol)
    to_remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    for idx in reversed(to_remove):
        em.RemoveAtom(idx)
    return em.GetMol()


def find_most_similar_fragments(new_mol_list, frag1_list, frag2_list):
    min_diff = float('inf')
    best_pair = None

    for new_mol, frag1, frag2 in zip(new_mol_list, frag1_list, frag2_list):
        # 각 프래그먼트의 분자량 계산
        atoms1 = frag1.GetNumAtoms()
        atoms2 = frag2.GetNumAtoms()

        # 분자량 차이의 절대값 계산
        diff = abs(atoms1 - atoms2)

        # 가장 작은 차이를 가진 쌍 찾기
        if diff < min_diff:
            min_diff = diff
            best_pair = [new_mol, frag1, frag2]

    return best_pair


def return_max_sim_offspring(parent_mol, offspring_mols):
    # parent_smiles = "CCO"  # 예: Ethanol
    # offspring_smiles_list = ["CCCO", "CCN", "CCCC", "C1=CC=CC=C1", "CCO"]  # 예: 여러 분자들

    # RDKit Mo
    # parent_mol = Chem.MolFromSmiles(parent_smiles)
    parent_fp = AllChem.GetMorganFingerprintAsBitVect(parent_mol, radius=2, nBits=2048)

    # Convert parent_fp to NumPy array for Cosine Similarity
    parent_fp_np = np.zeros((1,))
    ConvertToNumpyArray(parent_fp, parent_fp_np)

    # Offspring SMILES Tanimoto, Dice, Cosine Similarity
    similarities = []

    for idx, offspring_mol in enumerate(offspring_mols):
        # offspring_mol = Chem.MolFromSmiles(offspring_smiles)
        offspring_fp = AllChem.GetMorganFingerprintAsBitVect(offspring_mol, radius=2, nBits=2048)

        # Convert offspring_fp to NumPy array for Cosine Similarity
        offspring_fp_np = np.zeros((1,))
        ConvertToNumpyArray(offspring_fp, offspring_fp_np)

        # Tanimoto and Dice Similarity
        tanimoto_sim = TanimotoSimilarity(parent_fp, offspring_fp)
        dice_sim = DiceSimilarity(parent_fp, offspring_fp)

        # Cosine Similarity calculation
        cosine_sim = np.dot(parent_fp_np, offspring_fp_np) / (
                    np.linalg.norm(parent_fp_np) * np.linalg.norm(offspring_fp_np))

        # Similarity 평균 계산
        avg_similarity = (tanimoto_sim + dice_sim + cosine_sim) / 3
        similarities.append((offspring_mol, idx, tanimoto_sim, dice_sim, cosine_sim, avg_similarity))

    # 가장 높은 평균 유사성을 가진 offspring 찾기
    closest_offspring = max(similarities, key=lambda x: x[5])

    return closest_offspring


