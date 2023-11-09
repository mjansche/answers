#! /usr/bin/env python3
"""
How do I formulate the likelihood of a binomial mixture? I know
the number of successes (k) and success probability (p) in a set of
experiments but not the number of trials (n) in each experiment (n
varies between experiments, p is constant)?

https://qr.ae/pKFGGv

"""

import math

import numpy
import scipy


def binom_log_pmf(k, n, p):
    """Logarithm of the binomial PMF."""
    if 0 <= k <= n:
        return math.fsum([
            math.lgamma(n + 1),
            -math.lgamma(k + 1),
            -math.lgamma(n - k + 1),
            k * math.log(p),
            (n - k) * math.log1p(-p),
        ])
    return -math.inf


def mixture_log_pmf(k, p, ns, log_ws):
    """Logarithm of the mixture PMF."""
    assert len(ns) == len(log_ws)
    return scipy.special.logsumexp(
        [lw + binom_log_pmf(k, n, p) for n, lw in zip(ns, log_ws)])


def log_likelihood(ks, p, ns, ws):
    return math.fsum(mixture_log_pmf(k, p, ns, numpy.log(ws)) for k in ks)


def generate_data(approx_size, p, ns, ws):
    assert len(ns) == len(ws)
    return numpy.concatenate([
        scipy.stats.binom(n, p).rvs(round(w * approx_size))
        for n, w in zip(ns, ws)
    ])


def estimate(ks, p, ns, maxiter=10_000, abs_tol=1e-3):
    num_components = len(ns)
    ws = numpy.ones(num_components) / num_components

    L = log_likelihood(ks, p, ns, ws)
    print(L, ns, ws, sep="\t")

    for _ in range(maxiter):
        counts = numpy.zeros(num_components)
        sums = numpy.zeros(num_components)
        for k in ks:
            zs = numpy.array(
                [w * math.exp(binom_log_pmf(k, n, p)) for n, w in zip(ns, ws)])
            zs /= zs.sum()
            counts += zs
            sums += zs * k

        ns = sums / counts / p
        # ns = numpy.floor(ns).astype(int)
        ws = counts / counts.sum()

        L_new = log_likelihood(ks, p, ns, ws)
        print(L_new, ns, ws, sep="\t")
        if L_new - L < abs_tol:
            break
        L = L_new
    return ns, ws


def main():
    p = 0.1
    ks = generate_data(2000, p, [200, 300], [0.25, 0.75])

    ns0 = [int(ks.min() / p), int(ks.max() / p)]
    estimate(ks, p, ns0)
    return


if __name__ == "__main__":
    main()
