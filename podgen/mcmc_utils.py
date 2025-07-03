import numpy as np
import pandas as pd
import torch
import copy
import time
import glob
import os
from torch_geometric.data import Data
from torch.nn import functional as F
from p_tqdm import p_umap, p_map
import itertools
from pymatgen.core import Structure, Lattice


from CFtorch.common.data_utils import process_one
from CFtorch.common.group_utils import dof0_table, wmax_table, fc_mask_table, lattice_mask_table, mult_table, symops
from predictor.data_utils import build_crystal, build_crystal_graph
from scripts.eval_utils import project_xyz2
from scripts.eval_utils import top_p_sampling, sample_x, project_xyz, sample_normal, symmetrize_lattice

def get_temperature_list(max_t, min_t, num_t, type='line'):
    if type == 'line':
        anneal_T = np.linspace(max_t, min_t, num=num_t).tolist()
    elif type == 'cos':
        anneal_T = (min_t + (max_t - min_t) * (
                1 + np.cos(np.linspace(0, np.pi, num_t))) / 2).tolist()
    else:
        print('Temperature type must be either line or cos')
        raise ValueError(f"ValueError: Temperature type {type} is not supported.")

    return anneal_T


def get_seed_crystal(data_file, num, atom_types, wyck_types, n_max, tol, Nf, device, num_workers=1):
    df = pd.read_csv(data_file)
    df = df.sample(n=num, replace=True)

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [atom_types] * len(df),
        [wyck_types] * len(df),
        [n_max] * len(df),
        [tol] * len(df),
        num_cpus=num_workers)

    mpid_to_results = {result['mp_id']: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]['material_id']]
                       for idx in range(len(df))]

    G = []
    num_sites = []
    lattice = []
    frac_coor = []
    FTfrac_coor = []
    wyckoff = []
    atom_type = []
    M = []

    for result in ordered_results:
        FTfrac = [fn(2 * np.pi * result['frac_coor'][:, None] * f) for f in range(1, Nf + 1) for fn in
                       (np.sin, np.cos)]
        FTfrac = np.squeeze(np.stack(FTfrac, axis=-1), axis=1)
        m = mult_table[result['G'] - 1, result['wyckoff']]

        G.append(result['G'])
        num_sites.append(result['num_sites'])
        lattice.append(result['lattice'])
        frac_coor.append(result['frac_coor'])
        FTfrac_coor.append(FTfrac)
        wyckoff.append(result['wyckoff'])
        atom_type.append(result['atom_type'])
        M.append(m)

    data = {
        'G': torch.LongTensor(G).view(-1, 1).to(device),
        'num_sites': torch.LongTensor(num_sites).view(-1, 1).to(device),
        'lattice': torch.Tensor(lattice).view(-1, 1, 6).to(device),
        'frac_coor': torch.Tensor(frac_coor).to(device),
        'FTfrac_coor': torch.Tensor(FTfrac_coor).to(device),
        'wyckoff': torch.LongTensor(wyckoff).to(device),
        'atom_type': torch.LongTensor(atom_type).to(device),
        'M': torch.LongTensor(M).to(device),
    }

    return data


def data2struc(data, num_io_process=5):
    XYZ, A, W, G, L, M = data['frac_coor'].to('cpu'), data['atom_type'].to('cpu'), data['wyckoff'].to('cpu'), data['G'].to(
        'cpu').squeeze(), data['lattice'].to('cpu').squeeze(1), data['M'].to('cpu')
    num_atoms = torch.sum(M, dim=-1)
    length, angle = torch.split(L, [3, 3], dim=-1)
    length1 = length * (num_atoms.unsqueeze(1).repeat(1, 3) ** (1 / 3))
    angle = angle * (180.0 / np.pi)  # to deg
    L = torch.cat([length1, angle], dim=-1)
    # L = np.array(L).tolist()
    # X = np.array(XYZ).tolist()
    # A = np.array(A).tolist()
    # W = np.array(W).tolist()
    # G = np.array(G).tolist()
    L = L.detach().numpy().tolist()
    X = XYZ.detach().numpy().tolist()
    A = A.detach().numpy().tolist()
    W = W.detach().numpy().tolist()
    G = G.detach().numpy().tolist()
    # print('he')
    if isinstance(G, int):
        G = [G]
    num_io_process = min(len(G), num_io_process)

    structures = p_map(
        get_struct_from_lawx,
        G, L, A, W, X,
        num_cpus=num_io_process)

    return structures


def get_struct_from_lawx(G, L, A, W, X):
    """
    Get the pymatgen.Structure object from the input data

    Args:
        G: space group number
        L: lattice parameters
        A: element number list
        W: wyckoff letter list
        X: fractional coordinates list

    Returns:
        struct: pymatgen.Structure object
    """

    L = np.array(L)
    X = np.array(X)
    A = np.array(A)
    W = np.array(W)
    G = np.array(G)

    A = A[np.nonzero(A)]
    X = X[np.nonzero(A)]
    W = W[np.nonzero(A)]

    lattice = Lattice.from_parameters(*L)
    xs_list = [symmetrize_atoms(G, w, x) for w, x in zip(W, X)]
    as_list = [[A[idx] for _ in range(len(xs))] for idx, xs in enumerate(xs_list)]
    A_list = list(itertools.chain.from_iterable(as_list))
    X_list = list(itertools.chain.from_iterable(xs_list))
    struct = Structure(lattice, A_list, X_list)

    # graph_arrays = build_crystal_graph(struct)

    return struct#.as_dict()


def symmetrize_atoms(g, w, x):
    '''
    symmetrize atoms via, apply all sg symmetry op, finding the generator, and lastly apply symops
    we need to do that because the sampled atom might not be at the first WP
    Args:
       g: int
       w: int
       x: (3,)
    Returns:
       xs: (m, 3) symmetrize atom positions
    '''

    # (1) apply all space group symmetry op to the x
    w_max = wmax_table[g - 1].item()
    m_max = mult_table[g - 1, w_max].item()
    ops = symops[g - 1, w_max, :m_max]  # (m_max, 3, 4)
    affine_point = np.array([*x, 1])  # (4, )
    coords = ops @ affine_point  # (m_max, 3)
    coords -= np.floor(coords)

    # (2) search for the generator which satisfies op0(x) = x , i.e. the first Wyckoff position
    # here we solve it in a jit friendly way by looking for the minimal distance solution for the lhs and rhs
    # https://github.com/qzhu2017/PyXtal/blob/82e7d0eac1965c2713179eeda26a60cace06afc8/pyxtal/wyckoff_site.py#L115
    def dist_to_op0x(coord):
        diff = np.dot(symops[g - 1, w, 0], np.array([*coord, 1])) - coord
        diff -= np.floor(diff)
        return np.sum(diff ** 2)
        #  loc = np.argmin(jax.vmap(dist_to_op0x)(coords))

    loc = np.argmin([dist_to_op0x(coord) for coord in coords])
    x = coords[loc].reshape(3, )

    # (3) lastly, apply the given symmetry op to x
    m = mult_table[g - 1, w]
    ops = symops[g - 1, w, :m]  # (m, 3, 4)
    affine_point = np.array([*x, 1])  # (4, )
    xs = ops @ affine_point  # (m, 3)
    xs -= np.floor(xs)  # wrap back to 0-1
    return xs


def logp_of_pre(structures, pre_models, pre_models_config,temperature=1.0, num_io_process=5, graph_method='mindistance'):
    have_atom = 0
    frac_coords_list, atom_types_list, lengths_list, angles_list, edge_indices_list, to_jimages_list, num_atoms_list = [], [], [], [], [], [], []
    num_bonds_list = []
    batch_list = []
    i = 0
    # print(pre_models_config)

    graphes = p_map(build_crystal_graph,
                    structures,
                    [graph_method]*len(structures),
                    num_cpus=num_io_process)

    for graph_arrays in graphes:
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = graph_arrays
        frac_coords_list.append(frac_coords)
        atom_types_list.append(atom_types)
        lengths_list.append(lengths)
        angles_list.append(angles)
        edge_indices_list.append(edge_indices + have_atom)
        to_jimages_list.append(to_jimages)
        num_atoms_list.append(num_atoms)
        num_bonds_list.append(edge_indices.shape[0])
        batch_list.extend([i] * num_atoms)
        i = i + 1
        have_atom += num_atoms

    frac_coords_all = np.vstack(frac_coords_list)
    atom_types_all = np.concatenate(atom_types_list)
    lengths_all = np.vstack(lengths_list)
    angles_all = np.vstack(angles_list)
    edge_indices_all = np.vstack(edge_indices_list)
    to_jimages_all = np.vstack(to_jimages_list)
    num_atoms_all = np.vstack(num_atoms_list)

    data = Data(
        batch=torch.LongTensor(batch_list).view(-1),
        frac_coords=torch.Tensor(frac_coords_all),
        atom_types=torch.LongTensor(atom_types_all),
        lengths=torch.Tensor(lengths_all).view(len(structures), -1),
        angles=torch.Tensor(angles_all).view(len(structures), -1),
        edge_index=torch.LongTensor(
            edge_indices_all.T).contiguous(),  # shape (2, num_edges)
        to_jimages=torch.LongTensor(to_jimages_all),
        num_bonds=torch.LongTensor(num_bonds_list).view(-1),
        num_atoms=torch.LongTensor(num_atoms_all).view(-1),
        num_nodes=have_atom,  # special attribute used for batching in pytorch geometric
    )

    if torch.cuda.is_available():
        data.to('cuda')
    
    # print('data device',data.batch.device)
    # for name, param in pre_models[0].named_parameters():
    #     print(f"Parameter {name} is on device: {param.device}", flush=True)

    results = []
    logps = []
    target_logps = []
    for i in range(len(pre_models)):
        results.append(pre_models[i](data, type='test'))
        logp = F.log_softmax(results[i], dim=-1)
        logits = logp / temperature
        logp = F.log_softmax(logits, dim=-1)
        logps.append(logp)
        a = logp[: ,pre_models_config[i]['target']]
        target_logps.append(logps[i][:, pre_models_config[i]['target']])

    return target_logps


def proposals(data_input, p_tensor, model, xyz_std=0.01, l_std=0.01, Nf=5):
    data = copy.deepcopy(data_input)
    batch_size = data['G'].shape[0]
    device = data['G'].device
    indices = torch.multinomial(p_tensor, 1, replacement=True).data.cpu().item()

    if indices == 0: # generate some new crystal
        new_data = generate(batch_size, model, data['G'])
        num_sites = new_data['atom_type'] != 0
        num_sites = torch.sum(num_sites, dim=-1, keepdim=True)
        new_data['num_sites'] = num_sites
        return new_data, True

    elif indices == 1:  # update a
        nonzero_indices = torch.count_nonzero(data['atom_type'], dim=1)
        sampled_values = []
        for limit in nonzero_indices:
            intlimit = limit.cpu().item()
            sampled = torch.randint(0, intlimit, (1,)).cpu().item()
            sampled_values.append(sampled)
        random_numbers = [torch.randint(1, 95, (1,), device=device).item() for _ in range(batch_size)]
        for j in range(batch_size):
            data['atom_type'][j, sampled_values[j]] = random_numbers[j]

    elif indices == 2:  # update xyz
        noise = torch.randn((data['frac_coor'].shape), device=device) * xyz_std
        data['frac_coor'] = data['frac_coor'] + noise

    else:  # update l
        # print('l')
        noise = torch.randn((data['lattice'].shape), device=device) * l_std
        L = data['lattice'] + noise
        # L = L.squeeze(1)
        length, angle = torch.split(L, [3, 3], dim=-1)
        angle = angle * (180.0 / np.pi)  # to deg
        L = torch.cat([length, angle], dim=-1)
        for j in range(batch_size):
            L[j, :, :] = symmetrize_lattice(data['G'][j], L[j,:,:])
        length, angle = torch.split(L, [3, 3], dim=-1)
        angle = angle / (180.0 / np.pi)  # to deg
        L_sym = torch.cat([length, angle], dim=-1)
        data['lattice'] = L_sym

    data['frac_coor'] = project_xyz2(data['G'], data['wyckoff'], data['frac_coor'], 0)
    FTfrac_coor = [fn(2 * torch.pi * data['frac_coor'][:, None] * f) for f in range(1, Nf + 1) for fn in
                   (torch.sin, torch.cos)]
    FTfrac_coor = torch.squeeze(torch.stack(FTfrac_coor, dim=-1), dim=1)
    data['FTfrac_coor'] = FTfrac_coor

    return data, False


def accept(data, pro_data, data_logp, pro_data_logp, predict_logp, pro_predict_logp, struc, pro_struc, data_h,
           pro_data_h, full_logp, pro_full_logp):
    # print(f'full_logp:{full_logp}', flush=True)
    # print(f'pro_full_logp:{pro_full_logp}', flush=True)
    accept_logp = pro_full_logp - full_logp
    accept_logp[accept_logp > 0] = 0.0
    accept_p = torch.exp(accept_logp)
    # print(f'accept_p:{accept_p}', flush=True)
    mask = torch.bernoulli(accept_p).bool()
    data_cp = copy.deepcopy(data)
    for key in data.keys():
        data_cp[key][mask] = pro_data[key][mask]
    data_h[mask] = pro_data_h[mask]
    data_logp[mask] = pro_data_logp[mask]
    for i in range(len(predict_logp)):
        predict_logp[i][mask] = pro_predict_logp[i][mask]
    bool_list = mask.to('cpu').numpy().tolist()
    result_struc = [pro_struc[i] if bool_list[i] else struc[i] for i in range(len(struc))]
    average = accept_p.to('cpu').float().mean().item()

    return data_cp, result_struc, data_h, data_logp, predict_logp, average


def MCMC_setp(data, data_logp, data_h, predict_logp, struc, model, pre_models, propose_strategy, use_T, config, cfg):
    pro_data, gen = proposals(data, propose_strategy, model, xyz_std=config['generate_setting']['xyz_std'],
                              l_std=config['generate_setting']['l_std'], Nf=cfg.data.Nf)
    pro_data_h = model(pro_data)
    pro_struc = data2struc(pro_data, num_io_process=config['generate_setting']['num_worker'])
    pro_data_logp = model.compute_logp(pro_data, pro_data_h, temperature=use_T)
    pro_predict_logp = logp_of_pre(pro_struc, pre_models, config['pre_models'])

    full_logp = data_logp.clone()
    pro_full_logp = pro_data_logp.clone()
    for i in range(len(pre_models)):
        full_logp += config['pre_models'][i]['alpha'] * predict_logp[i]
        pro_full_logp += config['pre_models'][i]['alpha'] * pro_predict_logp[i]
    if 'ele_score' in config:
        full_logp, pro_full_logp = add_ele_score(full_logp, pro_full_logp, data, pro_data, config['ele_score'])
    if gen:
        full_logp += data_logp
        pro_full_logp += pro_data_logp
    if torch.isnan(pro_full_logp).any():
        print("pro_full_logp contains NaN values!")
        raise ValueError("pro_full_logp contains NaN values!")
    if torch.isinf(pro_full_logp).any():
        print("pro_full_logp contains Inf values!")
        raise ValueError("pro_full_logp contains NaN values!")

    data, struc, data_h, data_logp, predict_logp, accept_rate = accept(data, pro_data,
                                                                  data_logp, pro_data_logp,
                                                                  predict_logp, pro_predict_logp,
                                                                  struc, pro_struc,
                                                                  data_h, pro_data_h,
                                                                  full_logp, pro_full_logp)

    return data, struc, data_h, data_logp, predict_logp, accept_rate


def add_ele_score(full_logp, pro_full_logp, data, pro_data, ele_score):
    full_logp = full_logp.clone()
    pro_full_logp = pro_full_logp.clone()

    ele_score_tensor = torch.tensor([
        [ele_score.get(int(value), -1000) for value in row]
        for row in data['atom_type']
    ], device=full_logp.device)
    ele_score_tensor, _ = torch.max(ele_score_tensor, dim=1)
    ele_score_tensor[ele_score_tensor < -90] = 0.0
    full_logp += ele_score_tensor

    ele_score_tensor = torch.tensor([
        [ele_score.get(int(value), -1000) for value in row]
        for row in pro_data['atom_type']
    ], device=full_logp.device)
    ele_score_tensor, _ = torch.max(ele_score_tensor, dim=1)
    ele_score_tensor[ele_score_tensor < -90] = 0.0
    pro_full_logp += ele_score_tensor

    return full_logp, pro_full_logp


def logging(step, data_logp, predict_logp, accept_rate, use_T, balance, max_avg_logp, min_avg_logp, config, mode='Annealing'):
    if step % config['generate_setting']['print_step'] == 0 and mode == 'Annealing':
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
        print(mode ,':', 'logp=', data_logp.mean().cpu().item(), 'accept rate=', accept_rate, 'step=',
              step, 'T=', use_T, 'bala=', balance, 'time:', formatted_time, flush=True)
    if mode != 'Annealing':
        return 0
    now_avg_logp = data_logp.clone()
    for i in range(len(predict_logp)):
        now_avg_logp += config['pre_models'][i]['alpha'] * predict_logp[i]
    now_avg_logp = now_avg_logp.mean().cpu().item()
    if now_avg_logp > max_avg_logp:
        max_avg_logp = now_avg_logp
        balance = 0
    elif now_avg_logp < min_avg_logp:
        min_avg_logp = now_avg_logp
        balance = 0
    else:
        balance += 1

    return now_avg_logp, max_avg_logp, min_avg_logp, balance


def sample_2_cif(config, sample_list, cif_name, struc, predict_logp, matcher, chack=False):
    output_file = config['generate_setting']['output_file']
    predict_p = [torch.exp(predict_logpi).cpu().detach().numpy() for predict_logpi in predict_logp]
    for i in range(len(struc)):
        accept = True
        for j in range(len(predict_logp)):
            if predict_p[j][i] < config['pre_models'][j]['threshold']:
                accept = False
                break
        if not accept: continue

        if chack:
            for j in range(len(sample_list)):
                if matcher.fit(sample_list[j], struc[i]):
                    accept = False
                    break
        if not accept: continue

        sample_list.append(struc[i])
        cif_file = os.path.join(output_file, str(cif_name) + '.cif')
        struc[i].to(cif_file, fmt='cif')
        cif_name = cif_name + 1

    return sample_list, cif_name


def sample_list_from_file(output_file, worker=5):
    cif_files = glob.glob(os.path.join(output_file, '*.cif'))
    cif_files = list(cif_files)
    sample_list = p_map(process_cif, cif_files, num_cpus=worker)
    return sample_list


def process_cif(cif_file):
    return Structure.from_file(cif_file).get_primitive_structure()


def saveinannea(path, config, nT, data, avg_logp, accept_list, label=''):
    data_cpu = {key: tensor.cpu() for key, tensor in data.items()}
    save_dict = {
        "config": config,
        "nT": nT,
        "data": data_cpu,
        "avg_logp": torch.Tensor(avg_logp),
        "accept_list": torch.Tensor(accept_list)
    }
    if label == '':
        file = 'middle_' + str(nT) + '.pth'
    else:
        file = 'middle_' + str(nT) + '_' + label + '.pth'
    file = os.path.join(path, file)
    torch.save(save_dict, file)
    return file


def loadinannea(path, config):
    loaded_data = torch.load(path)
    old_config = loaded_data['config']
    # print('old_args: \n', old_config)
    data = loaded_data['data']
    # data = {key: tensor.to(device) for key, tensor in data.items()}
    nT = loaded_data['nT']
    old_config['generate_setting']['output_file'] = config['generate_setting']['output_file']
    old_config['pre_models'] = config['pre_models']
    return data, nT + 1, old_config





def generate(batch_size, model, spacegroup, top_p=1.0, temperature=1.0):
    data = {}
    W = torch.empty(batch_size, 0).long().to(model.device)
    A = torch.empty(batch_size, 0).long().to(model.device)
    XYZ = torch.empty(batch_size, 0, 3).to(model.device)
    L = torch.empty(batch_size, 0, model.lattice_types).to(model.device)
    FTfrac_coor = [fn(2 * torch.pi * XYZ[:, None] * f) for f in range(1, model.hparams.Nf + 1) for fn in
                   (torch.sin, torch.cos)]
    FTfrac_coor = torch.squeeze(torch.stack(FTfrac_coor, dim=-1), dim=1)
    mult_table_tensor = torch.tensor(mult_table).to(model.device)


    data['G'] = spacegroup
    data['wyckoff'] = W
    data['atom_type'] = A
    data['frac_coor'] = XYZ
    data['FTfrac_coor'] = FTfrac_coor
    data['lattice'] = L

    for i in range(model.hparams.n_max):
        # print(i)
        # (1) w
        w_logit = model(data)[:, -1, :]
        w_logit = w_logit[:, :model.hparams.n_wyck_types]
        w = top_p_sampling(w_logit, top_p, temperature)
        data['wyckoff'] = torch.cat([data['wyckoff'], w], dim=1)
        M = mult_table_tensor[data['G'].expand(-1, data['wyckoff'].size(1))-1, data['wyckoff']]
        data['M'] = M

        # (2) A
        data['atom_type'] = torch.cat([data['atom_type'], torch.zeros((batch_size, 1), device=model.device, dtype=torch.int)], dim=1)
        data['frac_coor'] = torch.cat([data['frac_coor'], torch.zeros((batch_size, 1, 3), device=model.device)], dim=1)
        FTfrac_coor = [fn(2 * torch.pi * data['frac_coor'][:, None] * f) for f in range(1, model.hparams.Nf + 1) for fn in (torch.sin, torch.cos)]
        FTfrac_coor = torch.squeeze(torch.stack(FTfrac_coor, dim=-1), dim=1)
        data['FTfrac_coor'] = FTfrac_coor

        h_al = model(data)
        a_logit = h_al[:, -5, :model.hparams.n_atom_types]  # .squeeze(1)
        hl = h_al[:, -5, model.hparams.n_atom_types:model.hparams.n_atom_types + model.lattice_types].unsqueeze(1)
        L = torch.cat([L, hl], dim=1)
        a = top_p_sampling(a_logit, top_p, temperature)
        data['atom_type'][:, -1] = a.squeeze()

        # (3) X
        h_x = model(data)
        h_x = h_x[:, -4, :3*model.hparams.Kx]
        x = sample_x(h_x, model.hparams.Kx, top_p, temperature, spacegroup.size()[0])

        # project to the first WP
        xyz = torch.cat([x,
                         torch.zeros((batch_size, 1), device=model.device),
                         torch.zeros((batch_size, 1), device=model.device)], dim=-1)

        xyz = project_xyz(data['G'], w, xyz, 0)
        x = xyz[:, 0].unsqueeze(1)
        data['frac_coor'][:, -1, 0] = x.squeeze()
        FTfrac_coor = [fn(2 * torch.pi * data['frac_coor'][:, None] * f) for f in range(1, model.hparams.Nf + 1) for fn in
                       (torch.sin, torch.cos)]
        FTfrac_coor = torch.squeeze(torch.stack(FTfrac_coor, dim=-1), dim=1)
        data['FTfrac_coor'] = FTfrac_coor

        # (4) Y
        h_y = model(data)
        h_y = h_y[:, -3, :3 * model.hparams.Kx]
        y = sample_x(h_y, model.hparams.Kx, top_p, temperature, spacegroup.size()[0])

        # project to the first WP
        xyz = torch.cat([x,
                                y,
                                torch.zeros((batch_size, 1), device=model.device)], dim=-1)

        xyz = project_xyz(data['G'], w, xyz, 0)
        y = xyz[:, 1].unsqueeze(1)
        data['frac_coor'][:, -1, 1] = y.squeeze()
        FTfrac_coor = [fn(2 * torch.pi * data['frac_coor'][:, None] * f) for f in range(1, model.hparams.Nf + 1) for fn in
                       (torch.sin, torch.cos)]
        FTfrac_coor = torch.squeeze(torch.stack(FTfrac_coor, dim=-1), dim=1)
        data['FTfrac_coor'] = FTfrac_coor

        # (5) Z
        h_z = model(data)
        h_z = h_z[:, -2, :3 * model.hparams.Kx]
        z = sample_x(h_z, model.hparams.Kx, top_p, temperature, spacegroup.size()[0])

        # project to the first WP
        xyz = torch.cat([x, y, z], dim=-1)

        xyz = project_xyz(data['G'], w, xyz, 0)
        z = xyz[:, 2].unsqueeze(1)
        data['frac_coor'][:, -1, 2] = z.squeeze()
        FTfrac_coor = [fn(2 * torch.pi * data['frac_coor'][:, None] * f) for f in range(1, model.hparams.Nf + 1) for fn in (torch.sin, torch.cos)]
        FTfrac_coor = torch.squeeze(torch.stack(FTfrac_coor, dim=-1), dim=1)
        data['FTfrac_coor'] = FTfrac_coor

    A = data['atom_type']
    num_sites = torch.sum(A != 0, axis=1)
    num_atoms = torch.sum(M, axis=1)

    assert torch.max(num_sites) < model.hparams.n_max, f"Assertion failed: max value {torch.max(num_sites)} is not less than {model.hparams.n_max}"

    trueL = L[torch.arange(batch_size, device=L.device), num_sites, :]
    l_logit, mu, sigma = torch.split(trueL,
                                     [model.hparams.Kl, 6 * model.hparams.Kl, 6 * model.hparams.Kl], dim=-1)
    sigma = F.softplus(sigma) + model.hparams.sigmamin
    mu = mu.view(mu.size()[0], model.hparams.Kl, 6)
    sigma = sigma.view(sigma.size()[0], model.hparams.Kl, 6)
    k = top_p_sampling(l_logit, top_p, temperature)
    mu = mu[torch.arange(batch_size), k.squeeze(1), :]
    sigma = sigma[torch.arange(batch_size), k.squeeze(1), :]

    L = sample_normal(mu, sigma)

    length, angle = torch.split(L, [3, 3], dim=-1)
    length = length * (num_atoms.unsqueeze(1).repeat(1, 3) ** (1 / 3))
    angle = angle * (180.0 / np.pi)  # to deg
    L = torch.cat([length, angle], dim=-1)
    L = torch.abs(L)
    L = symmetrize_lattice(spacegroup, L)

    length, angle = torch.split(L, [3, 3], dim=-1)
    length = length / (num_atoms.unsqueeze(1).repeat(1, 3) ** (1 / 3))
    angle = angle / (180.0 / np.pi)  # to deg
    L_sym = torch.cat([length, angle], dim=-1)
    data['lattice'] = L_sym.unsqueeze(1)

    return data