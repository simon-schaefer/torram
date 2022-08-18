import torch
import torram


def marginalize_distribution(cov: torch.Tensor, min_variance: float = 0.01, return_stddev: bool = False
                             ) -> torch.Tensor:
    if min_variance <= 0:
        raise ValueError(f"Invalid minimal variance, expected > 0, got {min_variance}")
    var_marginal = torch.diagonal(cov, dim1=-2, dim2=-1)
    stddev_marginal = torch.sqrt(torch.clamp_min(var_marginal, min_variance))
    if return_stddev:
        return stddev_marginal
    return torram.geometry.diag_last(stddev_marginal)
