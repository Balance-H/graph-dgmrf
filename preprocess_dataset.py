#add_edges_to_connect_graph 我添加了一个函数，以使得生成的图确保是连通的
import torch
import heapq
import torch_geometric as ptg
import matplotlib.pyplot as plt
import numpy as np
import argparse
import networkx as nx
import igraph as ig
from torch_sparse import coalesce

import visualization as vis
import constants
import utils
from data_loading.california import load_cal
from data_loading.wind import load_wind_speed, load_wind_cap
from data_loading.wiki import wiki_loader

DATASETS = {
    "cal": load_cal,
    "wind_speed": load_wind_speed,
    "wind_cap": load_wind_cap,
    "wiki_squirrel": wiki_loader("squirrel"),
    "wiki_chameleon": wiki_loader("chameleon"),
    "wiki_crocodile": wiki_loader("crocodile")
}

# Always on cpu
parser = argparse.ArgumentParser(description='Pre-process dataset')
parser.add_argument("--dataset", type=str, help="Dataset to pre-process")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--plot", type=int, default=0,
        help="If plots should be made during generation")

parser.add_argument("--graph_alg", type=str, default="knn",
        help="Algorithm to use for constructing graph")
parser.add_argument("--n_neighbors", type=int, default=5,
        help="Amount of neighbors to include in k-nn graph generation")

parser.add_argument("--mask_fraction", type=float, default=0.25,
        help="Fraction to mask when using random_mask")
parser.add_argument("--random_mask", type=int, default=1,
        help="Use a random mask rather than cut-out areas")

parser.add_argument("--dist_weight", type=int, default=0,
        help="Compute also eigenvalues for distance weighting")
parser.add_argument("--dist_weight_eps", type=int, default=1e-2,
        help="Epsilon to add to distances to prevent division by zero")

# log-det pre-processing steps
parser.add_argument("--compute_eigvals", type=int, default=1,
        help="If eigenvalues should be computed")
parser.add_argument("--compute_dad_traces", type=int, default=0,
        help="If traces of the DAD-matrix should be estimated")
parser.add_argument("--dad_samples", type=int, default=1000,
        help="Amount of samples to use in DAD trace estimates")
parser.add_argument("--dad_k_max", type=int, default=50,
        help="Maximum k to compute DAD trace for")

parser.add_argument("--decom_max_size", type=int, default=1000,
        help="Max component size for decom_graph")
parser.add_argument("--mst_k", type=int, default=32,
        help="k for sparse graph used to build MST (bigger => more connected, slower)")

args = parser.parse_args()

_ = torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

assert args.dataset, "No dataset selected"
assert args.dataset in DATASETS, "Unknown dataset: {}".format(args.dataset)

def add_edges_to_connect_graph(graph_data):
    """
    Add the minimum number of edges to make the graph connected.
    """
    edge_index = graph_data.edge_index
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, f"Invalid edge_index shape: {edge_index.shape}"
    edge_index = edge_index.clone().contiguous()
    num_nodes = graph_data.num_nodes
    pos = graph_data.pos

    # Convert to NetworkX graph
    G = ptg.utils.to_networkx(graph_data, to_undirected=True)

    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return graph_data

    new_edges = []
    for comp_a, comp_b in zip(components[:-1], components[1:]):
        idx_a = torch.tensor(list(comp_a), device=pos.device)
        idx_b = torch.tensor(list(comp_b), device=pos.device)

        # Compute distance between nodes in two components
        dist = torch.cdist(pos[idx_a], pos[idx_b])
        k = torch.argmin(dist)
        i = k // dist.size(1)
        j = k % dist.size(1)

        u = idx_a[i].item()
        v = idx_b[j].item()

        new_edges.append([u, v])
        new_edges.append([v, u])

    if new_edges:
        added = torch.tensor(new_edges, dtype=torch.long, device=edge_index.device).t()
        edge_index = torch.cat([edge_index, added], dim=1)

    # Remove duplicates and ensure shape [2, E]
    edge_index = edge_index.t()
    edge_index = torch.unique(edge_index, dim=0, sorted=True)
    edge_index = edge_index.t()

    assert edge_index.dim() == 2 and edge_index.size(0) == 2, f"Final edge_index shape invalid: {edge_index.shape}"

    graph_data.edge_index = edge_index
    return graph_data


#构建一个knn，n值尽可能大，然后从knn图中构建最小生成树
#然后从树T中任意节点沿着边权游走，直至到达最大分量节点，
#构建每个子分量的knn图，然后合成全图
def mst_from_sparse_knn_with_igraph(pos: torch.Tensor, mst_k: int):
    """
    pos: [N, d] torch.float32 (cpu or cuda)
    return: mst_edge_index [2, N-1] on CPU (torch.long)
    """

    N = pos.size(0)
    point_data = ptg.data.Data(pos=pos)

    # 1) 构造候选稀疏图（kNN）
    cand = ptg.transforms.KNNGraph(k=mst_k, force_undirected=True)(point_data)
    edge_index = cand.edge_index.contiguous()

    # 2) 去重（并保留无向边的一份：u < v）
    edge_index, _ = coalesce(edge_index, None, N, N)   # [2, E]
    u = edge_index[0]
    v = edge_index[1]
    mask = u < v
    u = u[mask].cpu()
    v = v[mask].cpu()

    # 3) 计算边权（欧氏距离）
    pos_cpu = pos.detach().cpu()
    w = torch.norm(pos_cpu[u] - pos_cpu[v], dim=-1).numpy()

    # 4) igraph MST
    edges = list(zip(u.tolist(), v.tolist()))
    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = w.tolist()

    mst = g.spanning_tree(weights="weight")  # MST（若图不连通会返回森林）
    mst_edges = np.array(mst.get_edgelist(), dtype=np.int64)  # [(a,b),...]
    if mst_edges.shape[0] != N - 1:
        # 候选图不连通时会发生：建议调大 mst_k
        raise RuntimeError(f"候选图不连通：MST 边数={mst_edges.shape[0]}，期望={N-1}。请增大 --mst_k")

    mst_edge_index = torch.from_numpy(mst_edges).t().contiguous().long()  # [2, N-1]
    return mst_edge_index


def is_connected_in_mst(comp: list, mst_edge_index: torch.Tensor, N: int) -> bool:
    """
    检查分量在树中是否连通：从该分量的一个节点开始，遍历该分量的所有节点。
    :param comp: 当前分量的节点集合
    :param mst_edge_index: 树上的边
    :param N: 总节点数
    :return: 如果分量在树中连通，返回True，否则返回False
    """
    adj_list = [[] for _ in range(N)]
    for i in range(mst_edge_index.size(1)):
        u, v = mst_edge_index[0, i].item(), mst_edge_index[1, i].item()
        adj_list[u].append(v)
        adj_list[v].append(u)

    visited = [False] * N
    stack = [comp[0]]  # 从分量的第一个节点开始
    visited[comp[0]] = True

    # 深度优先遍历（DFS）
    while stack:
        node = stack.pop()
        for neighbor in adj_list[node]:
            if not visited[neighbor] and neighbor in comp:
                visited[neighbor] = True
                stack.append(neighbor)

    # 判断分量是否完全连通
    return all(visited[node] for node in comp)


def build_components_on_tree_primlike(mst_edge_index: torch.Tensor,
                                      pos: torch.Tensor,
                                      max_size: int):
    """
    在 MST(树) 上按你描述的策略生成分量：
    从一个未访问节点开始，把“分量边界上最短的边”对应的新点加入，直到满 max_size。
    如果分量小于100，找到相邻的分量并合并到维度最小的分量中。
    return: components: List[List[int]]
    """
    N = pos.size(0)
    u = mst_edge_index[0].cpu().numpy()
    v = mst_edge_index[1].cpu().numpy()

    pos_cpu = pos.detach().cpu().numpy()
    w = np.linalg.norm(pos_cpu[u] - pos_cpu[v], axis=1)

    # adjacency list
    adj = [[] for _ in range(N)]
    for a, b, ww in zip(u, v, w):
        adj[a].append((ww, b))
        adj[b].append((ww, a))

    visited = np.zeros(N, dtype=bool)
    components = []

    # 生成分量
    for start in range(N):
        if visited[start]:
            continue

        comp = [start]
        visited[start] = True

        heap = []
        for ww, nb in adj[start]:
            heapq.heappush(heap, (ww, start, nb))

        # 扩展分量，直到达到 max_size
        while heap and len(comp) < max_size:
            ww, frm, nb = heapq.heappop(heap)
            if visited[nb]:
                continue
            visited[nb] = True
            comp.append(nb)
            for ww2, nb2 in adj[nb]:
                if not visited[nb2]:
                    heapq.heappush(heap, (ww2, nb, nb2))

        components.append(comp)

    # 合并小分量：按相邻分量合并
    merged_components = []
    visited_comp = [False] * len(components)

    while True:
        # 标记是否有小分量需要合并
        merged = False
        
        for i in range(len(components)):
            if visited_comp[i]:
                continue
            
            comp_a = components[i]
            
            if len(comp_a) < 100:
                # 找到与当前分量相邻的分量进行合并
                # 找到包含 comp_a 邻居的分量
                neighbors_of_a = set()
                for node in comp_a:
                    for ww, nb in adj[node]:
                        neighbors_of_a.add(nb)

                # 找到所有包含相邻节点的分量
                candidate_comp = None
                for j in range(len(components)):
                    if visited_comp[j] or i == j:
                        continue
                    
                    comp_b = components[j]
                    # 检查分量 b 是否包含 comp_a 的邻居
                    if any(nb in comp_b for nb in neighbors_of_a):
                        candidate_comp = comp_b
                        candidate_idx = j

                # 如果找到了最合适的分量进行合并
                if candidate_comp is not None:
                    comp_a.extend(candidate_comp)
                    visited_comp[candidate_idx] = True
                    merged = True
                    break
        
        # 如果没有分量需要合并，则停止
        if not merged:
            break
        
    merged_components = [comp for i, comp in enumerate(components) if not visited_comp[i]]

    # 检查每个分量是否在树中是连通的
    for i, comp in enumerate(merged_components):
        if not is_connected_in_mst(comp, mst_edge_index, N):
            print(f"Component {i} is not connected in the MST!")
        else:
            continue

    return merged_components



def decom_graph_fast(pos: torch.Tensor,
                     max_size: int = 1000,
                     knn_k: int = 5,
                     mst_k: int = 32):
    """
    返回：graph_y (PyG Data)，edge_index = MST + 各分量内部 knn
    另外把分量信息也塞进 Data（comp_id/comp_ptr/comp_index），方便你后续用。
    """
    N = pos.size(0)

    # A) MST（在稀疏 kNN 候选图上做）
    mst_edge = mst_from_sparse_knn_with_igraph(pos, mst_k=mst_k)  # [2, N-1]

    # MST 加双向
    mst_edge_ud = torch.cat([mst_edge, mst_edge.flip(0)], dim=1).contiguous()

    # B) 在 MST 上生成分量（Prim-like 扩张）
    components = build_components_on_tree_primlike(mst_edge, pos, max_size=max_size)

    # C) 每个分量内部建 knn，再映射回全局编号
    all_edges = [mst_edge_ud.to(pos.device)]

    comp_id = torch.full((N,), -1, dtype=torch.long)
    comp_index_list = []
    comp_ptr = [0]

    for cid, comp in enumerate(components):
        comp_nodes = torch.tensor(comp, dtype=torch.long, device=pos.device)
        comp_id[comp_nodes] = cid

        sub_pos = pos[comp_nodes]  # [m, d]

        # 用 PyG 的 KNNGraph（比 sklearn NearestNeighbors 通常更快）
        sub_data = ptg.data.Data(pos=sub_pos)
        sub_edge = ptg.transforms.KNNGraph(k=min(knn_k, sub_pos.size(0)-1),
                                           force_undirected=True)(sub_data).edge_index  # local [2, E]

        # 映射回全局：global = comp_nodes[local]
        mapped = comp_nodes[sub_edge]  # [2, E] global ids

        all_edges.append(mapped)

        # 记录分量节点（可选）
        comp_index_list.append(comp_nodes.cpu())
        comp_ptr.append(comp_ptr[-1] + comp_nodes.numel())

    # D) 合并边并去重
    edge_index = torch.cat(all_edges, dim=1).contiguous()
    edge_index, _ = coalesce(edge_index, None, N, N)

    graph_y = ptg.data.Data(edge_index=edge_index, pos=pos)

    # 分量信息（可选，但很有用）
    graph_y.comp_id = comp_id
    graph_y.comp_ptr = torch.tensor(comp_ptr, dtype=torch.long)
    graph_y.comp_index = torch.cat(comp_index_list, dim=0) if comp_index_list else torch.empty((0,), dtype=torch.long)

    return graph_y


# Load dataset
print("Loading dataset ...")
load_return = DATASETS[args.dataset](args)

if type(load_return) == ptg.data.Data:
    # Data loading created full graph
    graph_y = load_return

    if hasattr(graph_y, "mask_limits"):
        mask_limits = graph_y.mask_limits
    else:
        mask_limits = None

    full_ds_name = args.dataset

else:
    # Data loading only set up features, y and positions (i.e. spatial data)
    X_features, y, pos, mask_limits = load_return

    # Turn everything into pytorch tensors
    pos = torch.tensor(pos, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Generate graphs
    print("Generating graph ...")
    point_data = ptg.data.Data(pos=pos)

    if args.graph_alg == "decom_graph":
        # 使用 decompose_graph 生成图
        graph_y = decom_graph_fast(
            pos=pos,                           # 这里 pos 已经是 torch.tensor(float32)
            max_size=1000,
            knn_k=args.n_neighbors,            # 子knn的边数
            mst_k=args.mst_k
        )
        full_ds_name = "{}_decom_graph".format(args.dataset)
    else:
        if args.graph_alg == "delaunay":
            graph_transforms = ptg.transforms.Compose((
                ptg.transforms.Delaunay(),
                ptg.transforms.FaceToEdge(),
            ))
            graph_y = graph_transforms(point_data)
            full_ds_name = args.dataset + "_delaunay"
        elif args.graph_alg == "knn":
            graph_transforms = ptg.transforms.Compose((
                ptg.transforms.KNNGraph(k=args.n_neighbors, force_undirected=True),
            ))
            graph_y = graph_transforms(point_data)
            full_ds_name = "{}_{}nn".format(args.dataset, args.n_neighbors)
        elif args.graph_alg == "radknn":
            # Add first neighbors within radius, then k-nn until a minimum is reached

            # Compute 95%-quantile of distance to closest neighbor
            nn_transforms = ptg.transforms.Compose((
                ptg.transforms.KNNGraph(k=1, force_undirected=False),
            ))
            nn_graph = nn_transforms(point_data)

            distances = torch.norm(nn_graph.pos[nn_graph.edge_index[0,:]] -\
                    nn_graph.pos[nn_graph.edge_index[1,:]], dim=-1).numpy()
            r = np.quantile(distances, q=0.95)

            # Create radius graph
            rad_transforms = ptg.transforms.Compose((
                ptg.transforms.RadiusGraph(r=r, max_num_neighbors=100), # High max
            ))
            rad_edges = rad_transforms(point_data).edge_index

            # Compute k-nn edges
            knn_transforms_ud = ptg.transforms.Compose((
                ptg.transforms.KNNGraph(k=args.n_neighbors, force_undirected=True),
            ))
            knn_edges = knn_transforms_ud(point_data).edge_index

            # Join rad and knn edge-index
            joined_edge_index = torch.cat((rad_edges, knn_edges), axis=1)
            # Remove duplicate edges
            joined_edge_index, _ = coalesce(joined_edge_index, None,
                    point_data.num_nodes, point_data.num_nodes)
            graph_y = point_data
            graph_y.edge_index = joined_edge_index

            full_ds_name = "{}_rad{}nn".format(args.dataset, args.n_neighbors)
        else:
            assert False, "Unknown graph algorithm"

    if len(y.shape) == 1:
        # Make sure y tensor has 2 dimensions
        y = y.unsqueeze(1)
    graph_y.x = y

    if not utils.is_none(X_features):
        X_features = torch.tensor(X_features, dtype=torch.float32)
        graph_y.features = X_features

# Check if graph is connected or contains isolated components
#graph_y = add_edges_to_connect_graph(graph_y)
nx_graph = ptg.utils.to_networkx(graph_y, to_undirected=True)
n_components = nx.number_connected_components(nx_graph)
contains_iso = ptg.utils.contains_isolated_nodes(graph_y.edge_index, graph_y.num_nodes)
print("Graph connected: {}, n_components: {}".format((n_components == 1), n_components))
print("Contains isolated components (1-degree nodes): {}".format(contains_iso))

# Create Mask
if args.random_mask:
    n_mask = int(args.mask_fraction*graph_y.num_nodes)
    unobs_indexes = torch.randperm(graph_y.num_nodes)[:n_mask]

    unobs_mask = torch.zeros(graph_y.num_nodes).to(bool)
    unobs_mask[unobs_indexes] = True
else:
    assert not utils.is_none(mask_limits), "No mask limits exists for dataset"
    unobs_masks = torch.stack([torch.bitwise_and(
        (graph_y.pos >= limits[0]),
        (graph_y.pos < limits[1])
        ) for limits in mask_limits], dim=0) # Shape (n_masks, n_nodes, 2)

    unobs_mask = torch.any(torch.all(unobs_masks, dim=2), dim=0) # Shape (n_nodes,)

    graph_y.mask_limits = mask_limits

obs_mask = unobs_mask.bitwise_not()

n_masked = torch.sum(unobs_mask)
print("Masked {} / {} nodes".format(n_masked, graph_y.num_nodes))

graph_y.mask = obs_mask

if args.plot:
    vis.plot_graph(graph_y, "y", show=True, title="y")

# Additional computation if weighting by node distances
if args.dist_weight:
    utils.dist_weight_graph(graph_y, args.dist_weight_eps,
            compute_eigvals=bool(args.compute_eigvals))

# Log-determinant pre-processing steps
utils.log_det_preprocess(graph_y, args.dist_weight, args.compute_eigvals,
        args.compute_dad_traces, args.dad_k_max, args.dad_samples)

if args.compute_eigvals and args.plot:
    # Plot histogram of eigenvalues
    plt.hist(graph_y.eigvals.numpy(), bins=100, range=(-1,1))
    plt.title("Histogram of eigenvalues")
    plt.show()

# Save dataset
print("Saving graphs ...")
if args.random_mask:
    full_ds_name += "_random_{}".format(args.mask_fraction)

utils.save_graph_ds({"graph_y": graph_y}, args, full_ds_name)



