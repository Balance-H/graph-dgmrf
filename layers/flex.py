import torch
import torch_geometric as ptg
import scipy.linalg as spl

from layers.linear import LinearLayer
import utils

# Layer: G = D^(gammma)(a1 I + a2 D^(-1)A)
class FlexLayer(LinearLayer):
    def __init__(self, graph, config, vi_layer=False):
        super(FlexLayer, self).__init__(graph, config)

        self.dist_weighted = bool(config["dist_weight"])
        self.log_det_method = config["log_det_method"]
        self.eigvals_log_det = bool(self.log_det_method == "eigvals")
        self.decomp_log_det = bool(self.log_det_method == "decomp")

        if self.dist_weighted:
            if self.log_det_method == "eigvals":
                assert hasattr(graph, "weighted_eigvals"), (
                    "Dataset not pre-processed with weighted eigenvalues")
                self.adj_eigvals = graph.weighted_eigvals
            elif self.log_det_method == "dad":
                assert hasattr(graph, "weighted_dad_traces"), (
                    "Dataset not pre-processed with DAD traces")
                dad_traces = graph.weighted_dad_traces
            elif self.log_det_method == "decomp":
                dad_traces = None
            else:
                assert False, "Unknown log-det method"

            self.degrees = graph.weighted_degrees
            self.dist_edge_weights = graph.edge_attr[:,0]
        else:
            if self.log_det_method == "eigvals":
                assert hasattr(graph, "eigvals"), (
                    "Dataset not pre-processed with eigenvalues")
                self.adj_eigvals = graph.eigvals
            elif self.log_det_method == "dad":
                assert hasattr(graph, "dad_traces"), (
                    "Dataset not pre-processed with DAD traces")
                dad_traces = graph.dad_traces
            elif self.log_det_method == "decomp":
                dad_traces = None
            else:
                assert False, "Unknown log-det method"

        if self.log_det_method == "dad":
            # Complete vector to use in power series for log-det-computation
            k_max = len(dad_traces)
            self.power_ks = torch.arange(k_max) + 1
            self.power_series_vec = (dad_traces * torch.pow(-1., (self.power_ks+1))
                    ) / self.power_ks
        elif self.log_det_method == "decomp":
            assert hasattr(graph, "atoms_decomp"), (
                "Dataset not pre-processed with decomposition atoms")
            assert hasattr(graph, "seps_decomp"), (
                "Dataset not pre-processed with decomposition separators")
            self.atoms_decomp = graph.atoms_decomp
            self.seps_decomp = graph.seps_decomp
            self.edge_index = graph.edge_index

        self.log_degrees = torch.log(self.degrees)
        self.sum_log_degrees = torch.sum(self.log_degrees) # For determinant

        # Degree weighting parameter (can not be fixed for vi)
        self.fixed_gamma = (not vi_layer) and bool(config["fix_gamma"])
        if self.fixed_gamma:
            self.gamma_param = config["gamma_value"]*torch.ones(1)
        else:
            self.gamma_param = torch.nn.parameter.Parameter(2.*torch.rand(1,)-1)

        # edge_log_degrees contains log(d_i) of the target node of each edge
        self.edge_log_degrees = self.log_degrees[graph.edge_index[1]]
        self.edge_log_degrees_transpose = self.log_degrees[graph.edge_index[0]]

    @property
    def degree_power(self):
        if self.fixed_gamma:
            return self.gamma_param
        else:
            # Forcing gamma to be in (0,1)
            return torch.sigmoid(self.gamma_param)

    def log_det(self):
        if self.eigvals_log_det:
            # Eigenvalue-based method
            eigvals = self.neighbor_weight[0]*self.adj_eigvals + self.self_weight[0]
            agg_contrib = torch.sum(torch.log(torch.abs(eigvals))) # from (aI+aD^-1A)
            degree_contrib = self.degree_power*self.sum_log_degrees # From D^gamma
            return agg_contrib + degree_contrib
        elif self.decomp_log_det:
            # Graph decomposition method
            device = self.log_degrees.device
            edge_index = self.edge_index.to(device)
            edge_weights = torch.ones(edge_index.shape[1], device=device)
            if self.dist_weighted:
                edge_weights = self.dist_edge_weights.to(device)

            adjacency = torch.zeros(
                (self.num_nodes, self.num_nodes), device=device, dtype=self.log_degrees.dtype
            )
            adjacency.index_put_(
                (edge_index[1], edge_index[0]), edge_weights, accumulate=True
            )

            degree_gamma = torch.exp(self.degree_power * self.log_degrees)
            degree_gamma_minus = torch.exp((self.degree_power - 1) * self.log_degrees)

            omega = (
                self.self_weight * torch.diag(degree_gamma)
                + self.neighbor_weight * (torch.diag(degree_gamma_minus) @ adjacency)
            )

            def submatrix_logdet(indices):
                idx = torch.tensor(indices, device=device)
                sub = omega.index_select(0, idx).index_select(1, idx)
                sign, logabsdet = torch.linalg.slogdet(sub)
                return logabsdet

            atoms_logdet = torch.sum(torch.stack([
                submatrix_logdet(atom) for atom in self.atoms_decomp
            ]))
            if len(self.seps_decomp) == 0:
                seps_logdet = torch.zeros(1, device=device, dtype=self.log_degrees.dtype)
            else:
                seps_logdet = torch.sum(torch.stack([
                    submatrix_logdet(sep) for sep in self.seps_decomp
                ]))
            return atoms_logdet - seps_logdet
        else:
            # Power series method, using DAD traces
            alpha_contrib = self.num_nodes*self.alpha1_param
            gamma_contrib = self.degree_power*self.sum_log_degrees
            dad_contrib = torch.sum(self.power_series_vec *\
                    torch.pow(torch.tanh(self.alpha2_param), self.power_ks))
            return alpha_contrib + gamma_contrib + dad_contrib

    def weight_self_representation(self, x):
        # Representation of same node weighted with degree (taken to power)
        return (x.view(-1,self.num_nodes)*torch.exp(
            self.degree_power * self.log_degrees)).view(-1,1)

    def message(self, x_j, transpose):
        # x_j are neighbor features
        if transpose:
            log_degrees = self.edge_log_degrees_transpose
        else:
            log_degrees = self.edge_log_degrees

        edge_weights = torch.exp((self.degree_power - 1) * log_degrees)
        if self.dist_weighted:
            edge_weights = edge_weights*self.dist_edge_weights

        weighted_messages = x_j.view(-1, edge_weights.shape[0]) * edge_weights

        return weighted_messages.view(-1,1)
