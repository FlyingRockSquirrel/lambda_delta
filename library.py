import torch
import numpy as np
import math
import torchquad as tq
import copy


# A bump function R -> R centered at center and with a radius of radius
def bump(x, radius, center):
    if x >= center + radius:
        return torch.tensor(0.)
    elif x <= center - radius:
        return torch.tensor(0.)
    return torch.exp(-1/(1 - ((x - center)**2)/radius**2))


# Returns an anonymous function instead of a number as well as keeps track of
# the support
def lambda_bump(radius, center):
    return (lambda x: bump(x, radius, center), radius, center)


# Returns a dirac delta distribution centered on the point center. Takes list
# of real numbers
def dir_delt(center):
    return lambda x: x.app(torch.tensor(center))


# The bump function object
class Bump_fn:
    def __init__(self, radius, vector):
        inv_bump_area = 1/0.4439938161680794
        dim = len(vector)
        lam = (lambda x: lambda_bump(radius, x)[0])
        self.radius = radius
        self.center = vector
        self.fn_list = [lam(x) for x in vector]
        self.norm_fact = ((1/radius) * inv_bump_area) ** dim

    def app(self, x):
        return torch.mul(self.norm_fact, torch.prod(torch.Tensor(
            [f(y) for f, y in zip(self.fn_list, x)])))

    def diff(self, i):
        dup = copy.deepcopy(self)
        def difftd_fn(x): return torch.autograd.functional.jacobian(
            self.fn_list[i], x, True)
        dup.fn_list = copy.deepcopy(self.fn_list)
        dup.fn_list[i] = difftd_fn
        return dup


# Lifts a function f to the distribution T_f and applies it to
# the bump function x
def lift(f, x):
    def integrand(y):
        return torch.Tensor([torch.mul(f(i), x.app(i)) for i in y])
    domain = torch.tensor([[c - x.radius, c + x.radius] for c in x.center])
    integral = tq.MonteCarlo()
    dim = len(x.center)
    points = math.ceil(100 * x.radius) * dim
    return integral.integrate(integrand, dim, points, domain)


# Lifts a function f to return the distribution T_f
def lift_fun(f):
    return lambda x: lift(f, x)


# Takes the Jacobian of a function f at the point x
def jacobian(f, x):
    return torch.autograd.functional.jacobian(f, x)


# Takes the i'th dimension derivative of the distribution T and applies it to
# the bump function phi, returning a real number
def deriv(T, i, phi):
    return -(T(phi.diff(i)))


# Takes the i'th dimension derivative of the distribution T, returning a new
# distribution
def der(T, i):
    return lambda p: deriv(T, i, p)


# Below are some examples as to how to use this library.

# This bump function is R^3 -> R with radius 1 and centered at point (2, 2, 2)
m = Bump_fn(1, [2., 2., 2.])
# A dirac delta distribution centered at the point (2, 2, 2)
d = dir_delt([2., 2., 2.])
# This bump function is R -> R with radius 1 and centered at the origin
m_1 = Bump_fn(1, [0.])
# This bump function is R -> R with radius 1 and centered at 2
m_2 = Bump_fn(1, [2.])


def Id(x):
    return max(torch.tensor(0.), x) + min(torch.tensor(0.), x)


def sillyId(x):
    if x == 0:
        return torch.tensor(0.)
    else:
        return x


def const(x):
    return 1


def f(x):
    return x[1] * x[2] * x[0]


def sq(x):
    return x**2


def cu(x):
    return x**3


T_f = lift_fun(f)
T_sq = lift_fun(sq)
T_cu = lift_fun(cu)
T_const = lift_fun(const)

der_id = der(lift_fun(Id), 0)
two_der_id = der(der_id, 0)
der_sq = der(T_sq, 0)
double_der_sq = der(der_sq, 0)


# Uncomment these lines to print various distributions and applications. Note
# that Monte Carlo integration is somewhat unstable so there may be some
# variance from run to run, and this inaccuracy compounds at higher derivatives

# print("der_id")
# print(der_id(m_1))

# print("der_id")
# print(der_id(m_1))

# print("const_id")
# print(T_const(m_1))

# print("dirac")
# print(d(m))

# print("sq")
# print(T_sq(m_1))

# print("der_sq")
# print(der_sq(m_1))

# print("cu")
# print(T_cu(m_1))

# print("two_der_id")
# print(two_der_id(m_1))

# print("f")
# print(T_f(m))

# print("T_id offset")
# print(lift_fun(Id)(m_2))

# print("sq")
# print(double_der_sq(m_2))
