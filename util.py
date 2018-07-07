

def permutations(n):
    """
    Generate list of all possible permutations of n bools

    N.B First permutation in list is always the all True permutation and final
    permutation in list is always the all False permutationself.

    perms[1] = [True, ..., True]
    perms[-1] = [False, ..., False]

    Arguments:
    int n : bool list length

    Returns:
    list[list] perms : list of all possible permutations of n bools
    """
    # base cases
    if n <= 0:
        return []
    if n == 1:
        return [[True], [False]]

    perms = []
    for p in permutations(n - 1):
        perms.append([True] + p)
        perms.append([False] + p)
    return perms
