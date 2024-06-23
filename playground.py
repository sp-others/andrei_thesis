from enum import Enum


class Names(str, Enum):
    DEGREE = 'degree'
    N_FREQUENCIES = 'n_frequencies'
    THRESHOLD = 'threshold'
    LAMBDA = 'lambda_val'


all_values = [name.lower() for name in Names]

# print(type(all_values))
# print(all_values)


def foo(a, b, c, d):
    print(a, b, c, d)


print(sorted([1, 2, 3], reverse=True))

foo(*all_values)

print()
