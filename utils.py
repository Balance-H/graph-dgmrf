import os
import torch
import pickle
import copy
import pickle
import json
import numpy as np
import scipy.stats as sps
import torch_geometric as ptg
import scipy.linalg as spl
import torch.distributions as dists

import constants
from layers.dad_layer import DADLayer
from layers.weighted_dad_layer import WeightedDADLayer

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

def get_optimizer(name):
    assert name in OPTIMIZERS, "Unknown optimizer: {}".format(name)
    return OPTIMIZERS[name]

def load_dataset(ds_name, ds_dir=constants.DS_DIR):
    ds_dir_path = os.path.join(ds_dir, ds_name)

    # Pickled object might be on other device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    graphs = {}
    for filename in os.listdir(ds_dir_path):
        if filename[0] != ".": # Ignore hidden files
            var_name, ending = filename.split(".")

            if ending == "pickle":
                graph_path = os.path.join(ds_dir_path, filename)
                with open(graph_path, "rb") as graph_file:
                    graph = pickle.load(graph_file)

                    if not "_store" in graph.__dict__:
                        # Convert from pre-2.0 pyg Data object
                        graph = ptg.data.Data(**graph.__dict__)

                graphs[var_name] = graph.to(device)

    return graphs

def new_graph(like_graph, new_x=None):
    graph = copy.copy(like_graph) # Shallow copy
    graph.x = new_x
    return graph

def print_params(model, config, header=None):
    if header:
        print(header)

    print("Raw parameters:")
    for param_name, param_value in model.state_dict().items():
        print("{}: {}".format(param_name, param_value))

    print("Aggregation weights:")
    for layer_i, layer in enumerate(model.layers):
        print("Layer {}".format(layer_i))
        if hasattr(layer, "activation_weight"):
            print("non-linear weight: {:.4}".format(layer.activation_weight.item()))
        else:
            print("self: {:.4}, neighbor: {:.4}".format(
                layer.self_weight[0].item(), layer.neighbor_weight[0].item()))

        if hasattr(layer, "degree_power"):
            print("degree power: {:.4}".format(layer.degree_power[0].item()))

    if config["learn_noise_std"]:
        print("noise_std: {}".format(noise_std(config)))

def noise_var(config):
    return torch.exp(2.*config["log_noise_std"])

def noise_std(config):
    return torch.exp(config["log_noise_std"])

def save_graph(graph, name, dir_path):
    graph_path = os.path.join(dir_path, "{}.pickle".format(name))
    graph = graph.to(torch.device("cpu"))
    with open(graph_path, "wb") as graph_file:
        pickle.dump(graph, graph_file)

def save_graph_ds(save_dict, args, ds_name):
    ds_dir_path = os.path.join(constants.DS_DIR, ds_name)
    os.makedirs(ds_dir_path, exist_ok=True)

    for name, graph in save_dict.items():
        save_graph(graph, name, ds_dir_path)

    # Dump cmd-line arguments as json in dataset directory
    json_string = json.dumps(vars(args), sort_keys=True, indent=4)
    json_path = os.path.join(ds_dir_path, "description.json")
    with open(json_path, "w") as json_file:
        json_file.write(json_string)

def get_dataset_zooms(ds_name):
    for zoom_name, zoom_list in constants.DATASET_ZOOMS.items():
        if ds_name.startswith(zoom_name):
            return zoom_list
    return [] # No zooms for this dataset

def crps_score(pred_mean, pred_std, target):
    # Inputs should be numpy arrays
    z = (target - pred_mean)/pred_std

    crps = pred_std*((1./np.sqrt(np.pi)) - 2*sps.norm.pdf(z) - z*(2*sps.norm.cdf(z) - 1))
    return (-1.)*np.mean(crps) # Negative crps, so lower is better

def int_score(pred_mean, pred_std, target, alpha=0.05):
    lower_std, upper_std = sps.norm.interval(1.-alpha)
    lower = pred_mean + pred_std*lower_std
    upper = pred_mean + pred_std*upper_std

    int_score = (upper - lower) + (2/alpha)*(lower-target)*(target < lower) +\
        (2/alpha)*(target-upper)*(target > upper)

    return np.mean(int_score)

# Computes all eigenvalues of the diffusion weight matrix D^(-1)A
def compute_eigenvalues(graph):
    adj_matrix = ptg.utils.to_dense_adj(graph.edge_index)[0]
    node_degrees = ptg.utils.degree(graph.edge_index[0])

    adj_matrix_norm = adj_matrix / node_degrees.unsqueeze(1)
    adj_eigvals = spl.eigvals(adj_matrix_norm.cpu().numpy()).real

    return torch.tensor(adj_eigvals, dtype=torch.float32)

# Computes all eigenvalues of the adjacency matrix A
def compute_eigenvalues_A(graph):
    adj_matrix = ptg.utils.to_dense_adj(graph.edge_index)[0]
    adj_eigvals = spl.eigvals(adj_matrix.cpu().numpy()).real

    return torch.tensor(adj_eigvals, dtype=torch.float32)

# Computes all eigenvalues of the weighted diffusion weight matrix D^(-1)A
def compute_eigenvalues_weighted(graph):
    adj_matrix = ptg.utils.to_dense_adj(graph.edge_index,
            edge_attr=graph.edge_attr).squeeze()
    node_degrees = graph.weighted_degrees

    adj_matrix_norm = adj_matrix / node_degrees.unsqueeze(1)
    adj_eigvals = spl.eigvals(adj_matrix_norm.cpu().numpy()).real

    return torch.tensor(adj_eigvals, dtype=torch.float32)

# Add additional attributes to graph for distance weighting
def dist_weight_graph(graph, dist_eps, compute_eigvals=True):
    # Edge features are (1 / (d + eps))
    ptg.transforms.Distance()(graph)
    graph.edge_attr = 1. / (graph.edge_attr + dist_eps)

    # Compute weighted degrees of all nodes
    D = torch.zeros(graph.num_nodes)
    D.scatter_add_(dim=0, index=graph.edge_index[0,:], src=graph.edge_attr[:,0])
    graph.weighted_degrees = D

# Equirectangular projection
def project_eqrect(long_lat):
    """
    long_lat: (N, 2) array with longitudes and latitudes
    """

    max_pos = long_lat.max(axis=0)
    min_pos = long_lat.min(axis=0)

    center_point = 0.5*(max_pos + min_pos)
    centered_pos = long_lat - center_point

    # Projection will be maximally correct on center of the map
    centered_pos[:,0] *= np.cos(center_point[1]*(np.pi/180.))

    # Rescale to longitude in ~[-1,1]
    pos = centered_pos / centered_pos[:,0].max()
    return pos

def is_none(x):
    return type(x) == type(None)

class SampleDataset(ptg.data.Dataset):
    def __init__(self, samples, edge_index):
        super(SampleDataset, self).__init__("", None, None)

        self.samples = samples.unsqueeze(2)
        self.edge_index = edge_index

    def len(self):
        return self.samples.shape[0]

    def set_samples(self, samples):
        self.samples = samples.unsqueeze(2)

    def get(self, idx):
        return ptg.data.Data(x=self.samples[idx], edge_index=self.edge_index)

@torch.no_grad()
def compute_dad_traces(graph, k_max, n_samples, batch_size=32, weighted=False):
    # Allow for pre-processing on GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    edge_index = graph.edge_index.to(device)
    graph = graph.to(device)

    if weighted:
        dad_layer = WeightedDADLayer(graph)
    else:
        dad_layer = DADLayer(graph)
    N = graph.num_nodes

    # Minimum variance estimator,
    # Shape (n_samples, N)
    rand_samples = -1. + torch.randint(0, 2, (n_samples, N), device=device)*2 #

    # Initially the layer input is just the original batch of samples
    sample_ds = SampleDataset(rand_samples, edge_index)
    sample_loader = ptg.data.DataLoader(sample_ds,
            batch_size=batch_size, shuffle=False)

    power_traces = torch.zeros(k_max, device=device)
    for k in range(k_max):
        batch_outputs = []
        print("k={}".format(k+1))
        for data in sample_loader:
            batch_output = dad_layer(data.x, edge_index=data.edge_index)
            batch_outputs.append(batch_output)

        layer_output = torch.cat(batch_outputs, axis=0
                ).view(n_samples, N) # same shape as samples
        inner_prods = torch.sum(rand_samples*layer_output, axis=1)

        power_traces[k] = torch.mean(inner_prods) # MC estimate
        sample_ds.set_samples(layer_output) # For next k

    power_traces[0] = 0. # By construction, do not need stochastic estimate
    return power_traces.cpu() # Return tensor on cpu

def log_det_preprocess(graph, dist_weight, comp_eigvals, comp_dad_traces,
        dad_k_max, dad_samples):
    if comp_eigvals:
        print("Computing eigenvalues ...")
        graph.eigvals = compute_eigenvalues(graph)

    if comp_dad_traces:
        print("Computing DAD traces ...")
        graph.dad_traces = compute_dad_traces(graph, dad_k_max, dad_samples)

    if dist_weight:
        if comp_eigvals:
            print("Computing weighted eigenvalues ...")
            # Compute eigenvalues of weighted D^(-1) A
            graph.weighted_eigvals = compute_eigenvalues_weighted(graph)

        if comp_dad_traces:
            print("Computing weighted DAD traces ...")
            # Compute DAD traces of weighted D^(-1/2) A D^(-1/2)
            graph.weighted_dad_traces = compute_dad_traces(graph, dad_k_max,
                dad_samples, weighted = True)


#heng
import torch


def decomp_logdet_schur_cached(
    graph,
    degrees,
    a1,
    a2,
    gamma,
    eps=1e-6,
):
    """
    Fast exact log|det(Omega)| using cached atom-tree + cached per-atom internal edges.

    Requires graph to have caches created by prepare_decomp_cache():
      graph.decomp_tree, graph.decomp_par, graph.decomp_post, graph.decomp_sep_to_par,
      graph.decomp_atom_data (list of dict)

    Omega = a1 * D^gamma + a2 * D^(gamma-1) * A
    """
    device = degrees.device
    dtype = degrees.dtype
    degrees = degrees.to(device=device, dtype=dtype).clamp_min(1.0)

    tree = graph.decomp_tree
    par = graph.decomp_par
    post = graph.decomp_post
    sep_to_par = graph.decomp_sep_to_par
    atom_data = graph.decomp_atom_data

    m = len(atom_data)
    msg_schur = [None] * m
    msg_logdet = [None] * m

    # Quick path for tree-like decompositions: when atoms are edges (size 2)
    # and there are N-1 atoms, building full Omega + Cholesky is usually
    # faster and avoids potential corner cases in junction-tree handling.
    try:
        N = int(degrees.numel())
        if m == (N - 1) and all(int(d['nodes'].numel()) == 2 for d in atom_data):
            # build full Omega from graph (use edge attributes if present)
            ei = graph.edge_index
            if ei.dim() == 1:
                ei = ei.view(2, -1)
            src = ei[0].to(device=device).view(-1).long()
            dst = ei[1].to(device=device).view(-1).long()
            E = int(src.numel())
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                ew_all = graph.edge_attr[:, 0].to(device=device, dtype=dtype)
            else:
                ew_all = torch.ones(E, device=device, dtype=dtype)

            # Build full dense Omega
            degrees_clamped = degrees.clamp_min(1.0)
            diag = a1 * torch.pow(degrees_clamped, gamma)
            M = torch.diag(diag)
            off = a2 * torch.pow(degrees_clamped[dst], gamma - 1.0) * ew_all
            M.index_put_((dst, src), off, accumulate=True)
            M = 0.5 * (M + M.t())
            M = M + eps * torch.eye(N, device=device, dtype=dtype)

            L = torch.linalg.cholesky(M)
            return 2.0 * torch.sum(torch.log(torch.diagonal(L)))
    except Exception:
        # If quick path fails for any reason, continue to full decomp
        pass

    for u in post:
        data_u = atom_data[u]
        nodes_u = data_u["nodes"]          # (k,)
        k = int(nodes_u.numel())

        # ---- build local M_u quickly (no scanning full edge list) ----
        d_u = degrees[nodes_u]
        diag = a1 * torch.pow(d_u, gamma)
        M_u = torch.diag(diag)

        ls = data_u["ls"]
        ld = data_u["ld"]
        if ls.numel() > 0:
            dstg = data_u["dstg"]
            ew = data_u["ew"].to(device=device, dtype=dtype)
            d_dst = degrees[dstg]
            off = a2 * torch.pow(d_dst, gamma - 1.0) * ew
            M_u.index_put_((ld, ls), off, accumulate=True)

        # Symmetrize + jitter
        M_u = 0.5 * (M_u + M_u.t())
        M_u = M_u + eps * torch.eye(k, device=device, dtype=dtype)

        total_logdet = M_u.new_zeros(())

        # ---- absorb children Schur messages onto separators ----
        node2local = data_u["node2local"]  # python dict (fast for small sep)
        for v, sep_uv in tree[u]:
            if par[v] == u:
                total_logdet = total_logdet + msg_logdet[v]
                sch = msg_schur[v]

                # Try to use cached local separator indices if available (fast path)
                seps_local = data_u.get("seps_local", None)
                if seps_local is not None and (v in seps_local):
                    idx = seps_local[v]
                else:
                    # fallback (should be rare); compute mapping
                    idx = torch.tensor([node2local[n] for n in sep_uv],
                                       dtype=torch.long, device=device)
                M_u.index_put_((idx[:, None], idx[None, :]), sch, accumulate=True)

        # ---- eliminate internal variables; send Schur to parent ----
        if par[u] != -1:
            sep_nodes = sep_to_par[u]
            keep = torch.tensor([node2local[n] for n in sep_nodes],
                                dtype=torch.long, device=device)

            mask_keep = torch.zeros(k, device=device, dtype=torch.bool)
            mask_keep[keep] = True
            elim = torch.nonzero(~mask_keep, as_tuple=False).view(-1)

            M_ee = M_u.index_select(0, elim).index_select(1, elim)
            M_ek = M_u.index_select(0, elim).index_select(1, keep)
            M_ke = M_ek.t()
            M_kk = M_u.index_select(0, keep).index_select(1, keep)

            L = torch.linalg.cholesky(M_ee)
            logdet_ee = 2.0 * torch.sum(torch.log(torch.diagonal(L)))

            X = torch.cholesky_solve(M_ek, L)     # inv(M_ee) M_ek
            schur = M_kk - M_ke @ X

            schur = 0.5 * (schur + schur.t())
            schur = schur + eps * torch.eye(schur.shape[0], device=device, dtype=dtype)

            msg_schur[u] = schur
            msg_logdet[u] = total_logdet + logdet_ee
        else:
            # root
            L = torch.linalg.cholesky(M_u)
            logdet_root = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
            return total_logdet + logdet_root

    raise RuntimeError("decomp_logdet_schur_cached: did not return at root")
