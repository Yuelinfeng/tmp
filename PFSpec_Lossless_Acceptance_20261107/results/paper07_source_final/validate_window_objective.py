"""Enumerate the coupling independently and compare actual window gradients."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from window_objective import window_values


def enumerated_mass(p, q, width):
    fixed = np.argsort(-q, kind='stable')[:width-1]
    rest = q.copy()
    rest[fixed] = 0
    rest /= rest.sum()
    residual = np.maximum(p-rest, 0)
    if residual.sum():
        residual /= residual.sum()
    output = np.zeros_like(p)
    reused = np.zeros_like(p)
    for candidate, probability in enumerate(rest):
        if probability == 0:
            continue
        accepted = min(p[candidate]/probability, 1.)
        output[candidate] += probability*accepted
        reused[candidate] += probability*accepted
        for token, correction in enumerate(residual):
            weight = probability*(1-accepted)*correction
            output[token] += weight
            if token == candidate or token in fixed:
                reused[token] += weight
    return output, reused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261107)
    reports = []
    for case in range(48):
        size, width = 7, 3
        p = rng.dirichlet(np.ones(size))
        if case % 4 == 0:
            p[0] = 0
            p /= p.sum()
        logits = torch.tensor(rng.normal(size=size), dtype=torch.float64, requires_grad=True)
        old = rng.dirichlet(np.ones(size))
        c1 = 0. if case % 6 == 0 else 1. if case % 6 == 1 else float(rng.random())
        c2 = rng.random(size)
        c2[0], c2[1] = 0., 1.
        rows = [dict(candidate=i, candidate_q=float(old[i]), weight=c1,
                     second_position_probability=float(c2[i])) for i in range(size)]
        teacher = torch.tensor(p, dtype=torch.float64)[None].repeat(size, 1)
        weights = torch.tensor(old, dtype=torch.float64)

        def integrated(values):
            q = (values/.9).softmax(-1)[None].repeat(size, 1)
            result = window_values(teacher, q, rows, width)
            return ((result['first_joint']+result['second_position_joint_sample'])*weights).sum()

        objective = integrated(logits)
        objective.backward()
        actual_q = (logits.detach()/.9).softmax(-1).numpy()
        output, reused = enumerated_mass(p, actual_q, width)
        expected = c1*float(reused.sum()+np.dot(reused, c2))
        gradient_errors = []
        for dimension in [0, 3, 6]:
            left, right = logits.detach().clone(), logits.detach().clone()
            left[dimension] -= 1e-6
            right[dimension] += 1e-6
            difference = float((integrated(right)-integrated(left))/(2e-6))
            gradient_errors.append(abs(difference-float(logits.grad[dimension])))
        row = dict(case=case, target_output_maximum_error=float(np.max(np.abs(output-p))),
                   expected_accepted_length=c1+expected,
                   importance_integral_error=abs(float(objective)-expected),
                   gradient_maximum_error=max(gradient_errors),
                   gradient_finite=bool(torch.isfinite(logits.grad).all()),
                   independent_position_approximation_error=abs(c1*np.dot(reused, c2)-c1*reused.sum()*c2.mean()))
        if row['target_output_maximum_error'] > 1e-12 or row['importance_integral_error'] > 1e-12:
            raise ArithmeticError('Window objective differs from enumerated acceptance outcomes')
        if row['gradient_maximum_error'] > 2e-7 or not row['gradient_finite']:
            raise ArithmeticError('Window gradient differs from finite differences')
        reports.append(row)
    result = dict(status='complete', cases=48, rows=reports,
        maximum_output_error=max(r['target_output_maximum_error'] for r in reports),
        maximum_integral_error=max(r['importance_integral_error'] for r in reports),
        maximum_gradient_error=max(r['gradient_maximum_error'] for r in reports),
        maximum_false_independence_error=max(r['independent_position_approximation_error'] for r in reports),
        scope='Complete candidate, acceptance and residual enumeration; first Position zero/certain cases, Feature zeros, varying full conditional second Position, and finite-difference gradients')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+chr(10))
    print('Window enumeration and gradients passed', json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
