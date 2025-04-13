import math
from scipy.stats import norm
import scipy.stats as sps

cabel = [1.78, 1.75, 1.72, 1.74, 1.77]

mean = sum(cabel) / len(cabel)

def varience(values):
    _varience = sum((v - mean) ** 2 for v in values) / len(values)
    return _varience

def std_dev(values):
    return math.sqrt(varience(values))

print(mean, std_dev(cabel))


print(norm.cdf(30, 42, 8) - norm.cdf(20, 42, 8))

def critical_z_values(p):
    norm_dist = norm(loc=0.0, scale=1.0)
    left_tail_area = (1.0 - p) / 2.0
    uppear_area = 1.0 - ((1.0 - p) / 2.0)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(uppear_area)

def confidence_interval(p, sample_mean, sample_std, n):
    lower, uppear = critical_z_values(p)
    lower_c1 = lower * (sample_std / math.sqrt(n))
    uppear_c1 = uppear * (sample_std / math.sqrt(n))

    return sample_mean + lower_c1, sample_mean + uppear_c1


print(critical_z_values(.99), confidence_interval(p=.99, sample_mean=1.715588, sample_std=0.029252, n=34))

_mean = 10345
_std_dev = 552
_n = 45

p1 = 1 - norm.cdf(11641, _mean, _std_dev)
p2 = norm.cdf(9049, _mean, _std_dev)


print("Да, т.к. 0.05 >",p1 + p2)
