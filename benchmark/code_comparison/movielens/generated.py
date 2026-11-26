from triton.experimental import Const, Duplicated, autodiff

from shared import launch, squared_loss


def differentiate(factors, indices, ratings, losses, gradients, seed):
    active = [Duplicated(factor, grad) for factor, grad in zip(factors, gradients)]
    args = (*active, *map(Const, indices), Const(ratings), Duplicated(losses, seed))
    return launch(autodiff(squared_loss, mode="ReverseModeCombined"), args, ratings, factors[0].shape[1])
