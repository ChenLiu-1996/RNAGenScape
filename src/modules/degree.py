"""Graph-degree density estimator for latent manifold region analysis."""

from __future__ import annotations

import torch


def build_gaussian_graph(data, sigma=1.0):
    """Build a graph using a Gaussian kernel from data points."""
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    diff = data.unsqueeze(1) - data.unsqueeze(0)
    sq_distances = torch.sum(diff ** 2, dim=2)
    adjacency_matrix = torch.exp(-sq_distances / (2 * sigma ** 2))
    adjacency_matrix.fill_diagonal_(0)
    return adjacency_matrix


def gaussian_kernel_graph_adaptive_bandwidth(X, k=5):
    """Gaussian kernel graph with adaptive bandwidth for each point."""
    pairwise_distances = torch.cdist(X, X, p=2)
    sorted_distances, _ = torch.sort(pairwise_distances, dim=1)
    kth_distances = sorted_distances[:, k].unsqueeze(1)
    adaptive_bandwidth = torch.sqrt(kth_distances @ kth_distances.T)
    adjacency_matrix = torch.exp(-pairwise_distances ** 2 / (2 * adaptive_bandwidth ** 2))
    adjacency_matrix.fill_diagonal_(0)
    return adjacency_matrix


def get_degree(data, bandwidth_type="fixed", sigma=1.0, k=5):
    """Compute node degrees using either fixed or adaptive bandwidth."""
    if bandwidth_type == "fixed":
        graph = build_gaussian_graph(data, sigma)
    elif bandwidth_type == "adaptive":
        graph = gaussian_kernel_graph_adaptive_bandwidth(data, k)
    else:
        raise ValueError("bandwidth_type must be either 'fixed' or 'adaptive'")
    return torch.sum(graph, dim=1)


class GraphDegreeDensityEstimator:
    """Estimate node degrees for new points based on a fitted dataset."""

    def __init__(
        self,
        bandwidth_type="adaptive",
        sigma=1.0,
        k=5,
        boundary_correction=True,
        device="cuda",
    ):
        self.bandwidth_type = bandwidth_type
        self.sigma = sigma
        self.k = k
        self.boundary_correction = boundary_correction
        self.device = device
        self.data = None
        self.adaptive_bandwidths = None

        if bandwidth_type not in ["fixed", "adaptive"]:
            raise ValueError("bandwidth_type must be either 'fixed' or 'adaptive'")

    def fit(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        self.data = data.to(self.device)

        if self.bandwidth_type == "adaptive":
            pairwise_distances = torch.cdist(self.data, self.data, p=2)
            sorted_distances, _ = torch.sort(pairwise_distances, dim=1)
            self.adaptive_bandwidths = torch.sqrt(
                sorted_distances[:, self.k].unsqueeze(1)
            )

        return self

    def transform(self, X):
        if self.data is None:
            raise RuntimeError("The estimator must be fitted before calling transform")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device)

        if X.dim() == 1:
            X = X.unsqueeze(0)

        pairwise_distances = torch.cdist(X, self.data, p=2)

        if self.bandwidth_type == "fixed":
            adjacency = torch.exp(-pairwise_distances ** 2 / (2 * self.sigma ** 2))

            if self.boundary_correction:
                min_vals, _ = torch.min(self.data, dim=0)
                max_vals, _ = torch.max(self.data, dim=0)

                correction_factors = torch.ones_like(adjacency)
                for dim in range(self.data.shape[1]):
                    dist_to_min = (X[:, dim : dim + 1] - min_vals[dim]) / self.sigma
                    dist_to_max = (max_vals[dim] - X[:, dim : dim + 1]) / self.sigma
                    boundary_factor = 0.5 * (
                        torch.erf(dist_to_min) + torch.erf(dist_to_max)
                    )
                    correction_factors = correction_factors * boundary_factor

                adjacency = adjacency / correction_factors.clamp(min=1e-10)
        else:
            sorted_distances, _ = torch.sort(pairwise_distances, dim=1)
            query_bandwidths = torch.sqrt(sorted_distances[:, self.k].unsqueeze(1))
            adaptive_bandwidth = torch.sqrt(
                query_bandwidths @ self.adaptive_bandwidths.T
            )
            adjacency = torch.exp(
                -pairwise_distances ** 2 / (2 * adaptive_bandwidth ** 2)
            )

            if self.boundary_correction:
                min_vals, _ = torch.min(self.data, dim=0)
                max_vals, _ = torch.max(self.data, dim=0)

                correction_factors = torch.ones_like(adjacency)
                for dim in range(self.data.shape[1]):
                    local_bandwidth = adaptive_bandwidth[:, :1]
                    dist_to_min = (
                        X[:, dim : dim + 1] - min_vals[dim]
                    ) / local_bandwidth
                    dist_to_max = (
                        max_vals[dim] - X[:, dim : dim + 1]
                    ) / local_bandwidth
                    boundary_factor = 0.5 * (
                        torch.erf(dist_to_min) + torch.erf(dist_to_max)
                    )
                    correction_factors = correction_factors * boundary_factor

                adjacency = adjacency / correction_factors.clamp(min=1e-10)

        degrees = torch.sum(adjacency, dim=1)
        if X.size(0) == 1:
            degrees = degrees.squeeze()
        return degrees
