"""Exact P/F/P accepted-length objective with a full-support observation law."""
import torch


def accepted_feature_mass(p, q, width=64):
    if p.shape != q.shape or p.ndim != 2:
        raise ValueError('Feature policies must have identical batch and vocabulary shapes')
    support = (q > 0).sum(-1)
    if not bool((support > 0).all()):
        raise ValueError('Draft policy needs positive probability mass')
    width = min(int(width), q.shape[-1])
    if width < 1:
        raise ValueError('Feature candidate width must be positive')
    order = torch.argsort(q, descending=True, stable=True)
    fixed_count = torch.minimum(support-1, torch.full_like(support, width-1))
    selected = torch.arange(q.shape[-1], device=q.device)[None] < fixed_count[:, None]
    fixed = torch.zeros_like(q, dtype=torch.bool).scatter(1, order, selected)
    rest = q.masked_fill(fixed, 0)
    rest = rest / rest.sum(-1, keepdim=True)
    # This is output mass that can use a preverified branch, including residual hits.
    return torch.where(fixed, p, torch.minimum(p, rest))


def window_values(p, q, rows, width=64):
    mass = accepted_feature_mass(p, q, width)
    candidate = torch.tensor([r['candidate'] for r in rows], device=q.device)
    old_probability = q.new_tensor([r['candidate_q'] for r in rows])
    c1 = q.new_tensor([r['weight'] for r in rows])
    c2 = q.new_tensor([r['second_position_probability'] for r in rows])
    if not bool((old_probability > 0).all()) or not bool(((c1 >= 0) & (c1 <= 1) & (c2 >= 0) & (c2 <= 1)).all()):
        raise ArithmeticError('Observation probabilities are outside their valid range')
    chosen_mass = mass.gather(1, candidate[:, None]).squeeze(1)
    first_joint = c1 * mass.sum(-1)
    second_position_joint = c1 * chosen_mass * c2 / old_probability
    return dict(first_position=c1, first_joint=first_joint,
                second_position_joint_sample=second_position_joint,
                accepted_draft_length_sample=c1+first_joint+second_position_joint,
                feature_overlap=torch.minimum(p, q).sum(-1),
                feature_set_acceptance=mass.sum(-1))


def loss(p, q, rows, width=64):
    values = window_values(p, q, rows, width)
    return -(values['first_joint'] + values['second_position_joint_sample']).mean()
