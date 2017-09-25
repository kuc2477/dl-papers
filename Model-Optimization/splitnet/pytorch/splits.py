import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


def split_loss(w, p, q, gamma1, gamma2, gamma3, cuda=True):
    splits, p_dimension = p.size()
    splits_, q_dimension = q.size()
    out_dimension, in_dimension = w.size()[:2]

    # Validate shape of the weight and split indicators.
    assert len(w.size()) in (2, 4)
    assert splits == splits_
    assert splits > 1
    assert p_dimension == in_dimension
    assert q_dimension == out_dimension
    assert len(p.size()) == 2
    assert len(q.size()) == 2

    # 1. Overlap loss.
    p_overlap_loss = sum([
        torch.sum(p[i, :] * p[j, :])
        for j in range(splits)
        for i in range(splits)
        if i > j
    ]) / ((p_dimension * (splits-1)) / (2*splits))

    q_overlap_loss = sum([
        torch.sum(q[i, :] * q[j, :])
        for j in range(splits)
        for i in range(splits)
        if i > j
    ]) / ((q_dimension * (splits-1)) / (2*splits))

    overlap_loss = p_overlap_loss + q_overlap_loss

    # 2. Uniform loss.
    p_uniform_loss = sum([p[i, :].sum()**2 for i in range(splits)])
    q_uniform_loss = sum([q[i, :].sum()**2 for i in range(splits)])
    uniform_loss = p_uniform_loss + q_uniform_loss

    # 3. Split loss.
    is_tensor = len(w.size()) == 4
    ones_col = Variable(torch.ones((in_dimension,)))
    ones_row = Variable(torch.ones((out_dimension,)))
    if cuda:
        ones_col = ones_col.cuda()
        ones_row = ones_row.cuda()

    if is_tensor:
        w_norm = (w**2).mean(-1).mean(-1)
        stddev = np.sqrt(1./w.size()[2]**2/in_dimension)
    else:
        w_norm = w
        stddev = np.sqrt(1./in_dimension)

    group_split_losses = []
    # 3-1. Find split loss - l2 of inappropriate weights - for each groups.
    for i in range(splits):
        if is_tensor:
            wg_row = (w_norm.t() * q[i, :]**2).t() * (ones_col-p[i, :])**2
            wg_row_l2 = wg_row.sum(dim=0).sqrt().sum() / (
                in_dimension*np.sqrt(out_dimension)
            )
            wg_col = (w_norm.t() * (ones_row-q[i, :])**2).t() * p[i, :]**2
            wg_col_l2 = wg_col.sum(dim=1).sqrt().sum() / (
                out_dimension*np.sqrt(in_dimension)
            )
        else:
            wg_row = (w_norm.t() * q[i, :]).t() * (ones_col-p[i, :])
            wg_row_l2 = (wg_row**2).sum(dim=0) / (
                in_dimension*np.sqrt(out_dimension)
            )
            wg_col = (w_norm.t() * (ones_row-q[i, :])).t() * p[i, :]
            wg_col_l2 = (wg_col**2).sum(dim=1) / (
                out_dimension*np.sqrt(in_dimension)
            )
        group_split_losses.append(wg_row_l2 + wg_col_l2)

    # 3-2. Normalize the total split loss.
    split_loss = sum(group_split_losses) / (2*(splits-1)*stddev / splits)

    # Return the total regularization loss.
    return (
        overlap_loss * gamma1 +
        uniform_loss * gamma2 +
        split_loss * gamma3
    )


def split_indicator(splits, dimension, cuda=True):
    alpha = Variable(
        torch.Tensor(splits, dimension).normal_(std=0.01),
        requires_grad=True
    )
    return nn.Softmax()(alpha.cuda() if cuda else alpha)


def merge_split_indicator(q, supergroups):
    splits, _ = q.size()

    # decide in which supergroup each split belongs to.
    allocated_supergroups = _allocate_supergroups_equally(
        splits, supergroups
    )

    # merge the splits into their according supergroups.
    merged = []
    for supergroup in range(max(allocated_supergroups) + 1):
        merged.append(sum([
            q[subgroup, :] for subgroup in range(splits)
            if allocated_supergroups[subgroup] == supergroup
        ]))
    return torch.stack(merged)


def _allocate_supergroups_equally(subgroups, supergroups):
    assert subgroups >= supergroups
    # decide how many subgroups belongs to each supergroup.
    supergroup_elements = [
        (subgroups + supergroups - i - 1) // supergroups for i in
        range(supergroups)
    ]
    # define the supergroup-to-subgroups allocations.
    supergroup_allocations = [
        [supergroup] * n for supergroup, n in
        enumerate(supergroup_elements)
    ]
    # return the subgroup-to-supergroup allocations.
    return [e for l in supergroup_allocations for e in l]
