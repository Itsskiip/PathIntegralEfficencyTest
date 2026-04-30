import asyncio
from multiprocessing import Pool, freeze_support
from sympy import *
import numpy as np
import json

from sympy.core.numbers import ImaginaryUnit
from sympy.core.rules import Transform

functions = []
with open('functions.json', 'r') as f:
    functions = json.load(f)
    for fn in functions:
        fn["func"] = sympify(fn["func"])
weights = np.array([fn["weight"] for fn in functions])
weights = weights / weights.sum()

def random_function(weighted=True):
    """
    Returns a random function from the function list.
    :param weighted: If the random selection should be weighted or not
    :return: A function at random
    """
    if weighted:
        return functions[np.random.choice(len(functions), p=weights)]
    return functions[np.random.randint(len(functions))]

def construct_random_scalar_function(params: int | list, weighted=True, max_depth=10,
                                     constant_chance=lambda d: (10-d)/20, constant_range=(-10,10),
                                     single_variable_chance=lambda d: (10-d)/20, complex_functions=False):
    """
    Constructs a random scalar function from recursively applying functions from the function list.
    :param params: A number of parameters for the function to have or a list of symbols to use as parameters
    :param weighted: If the random selection of functions used to construct the final function should be weighted
    :param max_depth: The maximum depth of the function to be constructed (i.e., a*(b*(c*(d*e))) has depth 4)
    :param constant_chance: The chance of a constant being used as an input, as a function of the maximum depth - the current depth
    :param constant_range: The range of constants to be used
    :param single_variable_chance: The chance of a single variable being used as an input, as a function of the maximum depth - the current depth
    :param complex_functions: If the constructed can be complex, or if only the real portion of it should be returned
    :return: A randomly constructed scalar function
    """
    if type(params) is int:
        params = symbols('x:'+str(params), real=not complex_functions)
    func = random_function(weighted)
    inputs = []
    for i in range(func["num_params"]):
        if np.random.rand() < constant_chance(max_depth):
            inputs.append(round(np.random.uniform(*constant_range), 2))
        elif max_depth == 0 or np.random.rand() < single_variable_chance(max_depth):
            inputs.append(params[np.random.randint(len(params))])
        else:
            inputs.append(construct_random_scalar_function(params, weighted, max_depth-1, constant_chance, constant_range, single_variable_chance))
    f_new = func['func'].subs([(symb, inputs[i]) for i, symb in enumerate(func['func'].free_symbols)])
    if not complex_functions:
        return re(f_new)
    return f_new

def construct_clean_random_scalar_function(params: int, weighted=True, max_depth=10, constant_chance=lambda d: (10-d)/20, constant_range=(-10,10), single_variable_chance=lambda d: (10-d)/20, complex_functions=False,
                                           discard_constant_funcs=True, round_n: None | int = 3 ):
    """
        Constructs and cleans a random scalar function from recursively applying functions from the function list.
        :param params: A number of parameters for the function to have
        :param weighted: If the random selection of functions used to construct the final function should be weighted
        :param max_depth: The maximum depth of the function to be constructed (i.e., a*(b*(c*(d*e))) has depth 4)
        :param constant_chance: The chance of a constant being used as an input, as a function of the maximum depth - the current depth
        :param constant_range: The range of constants to be used
        :param single_variable_chance: The chance of a single variable being used as an input, as a function of the maximum depth - the current depth
        :param complex_functions: If the constructed can be complex, or if only the real portion of it should be returned
        :param discard_constant_funcs: If the function should be remade if it contains only constants
        :param round_n: The number of decimal places to round the function to, or None if no rounding should be done
        :return: A randomly constructed and cleaned scalar function
    """
    try:
        round_ = round_n is not None
        func = simplify(construct_random_scalar_function(params, weighted, max_depth, constant_chance, constant_range, single_variable_chance, complex_functions)).evalf(round_n if round_ else None)
        while discard_constant_funcs and func.is_constant():
            func = simplify(construct_random_scalar_function(params, weighted, max_depth, constant_chance, constant_range, single_variable_chance, complex_functions)).evalf(round_n if round_ else None)
        if round_:
            return func.xreplace({n: round(n, round_n) for n in func.atoms(Number)})
        return func
    except:
        x0 = symbols('x0')
        return x0

def construct_random_scalar_functions_branching(params: int | list, weighted=True, max_depth=10,
                                branches_per_layer=lambda d: 4,
                                constant_chance=lambda d: (10-d)/20, constant_range=(-10,10),
                                single_variable_chance=lambda d: (10-d)/20, complex_functions=False) -> set:
    if type(params) is int:
        params = symbols('x:'+str(params), real=not complex_functions)
    branches = set()
    func = random_function(weighted)
    for i in range(max(1, round(branches_per_layer(max_depth)))):
        inputs = []
        m = 1
        for j in range(func["num_params"]):
            if np.random.rand() < constant_chance(max_depth):
                inputs.append([round(np.random.uniform(*constant_range), 2)])
            elif max_depth == 0 or np.random.rand() < single_variable_chance(max_depth):
                inputs.append([params[np.random.randint(len(params))]])
            else:
                i_fs = construct_random_scalar_functions_branching(params, weighted, max_depth-1, branches_per_layer,
                                                                   constant_chance, constant_range, single_variable_chance)
                m = max(m, len(i_fs))
                inputs.append(i_fs)
        for j in range(m):
            f_new = func['func'].subs([(symb, list(inputs[k])[j % len(inputs[k])]) for k, symb in enumerate(func['func'].free_symbols)])
            if not complex_functions:
                branches.add(re(f_new))
            branches.add(f_new)
    return branches

last_progress = 0.0
def print_progress(progress: float):
    global last_progress
    if round(progress, 2) == round(last_progress, 2):
        return
    print(f"\r> Progress: {progress:.0%}", end="")

def prune_function(function: Expr, prune_constants=True, complex_functions=False, round_n=2) -> Expr | None:
    try:
        func_new = simplify(function)
        if func_new == nan or (
                prune_constants and func_new.is_constant()):
            return None
        if round_n is not None:
            func_new = func_new.xreplace({n: round(n, round_n) for n in func_new.atoms(Number)})
        if not complex_functions:
            return re(func_new)
        return func_new
    except:
        return None
"""async def await_prune_function(function: Expr, prune_constants=True, complex_functions=False, round_n=2, max_wait=1.0):
    
    try:
        return await asyncio.wait_for(prune_function(function, prune_constants, complex_functions, round_n), max_wait)
    except:
        return None"""

def prune_function_list(function_list: list | set, prune_constants=True, complex_functions=False, round_n=2, max_wait=1.0, num_processes: int | None = None) -> list | set:
    new_function_list = set()
    with Pool(processes=num_processes) as pool:
        new_function_list |= set(pool.starmap(prune_function,
                                              [(function, prune_constants, complex_functions, round_n) for function in function_list]))
    new_function_list.discard(None)

    if type(function_list) is list:
        function_list.clear()
        function_list.extend(list(new_function_list))
    elif type(function_list) is set:
        function_list.clear()
        function_list |= new_function_list
    return function_list

def generate_dataset(num_functions: int, min_num_trees: int, directory_path: str,
                     params: int, weighted=True, max_depth=10, branches_per_layer=lambda d: 4,
                     constant_chance=lambda d: (10-d)/20, constant_range=(-10,10),
                     single_variable_chance=lambda d: (10-d)/20, complex_functions=False,
                     prune_constants=True, round_n: None | int = 3, max_wait=1.0):
    dataset = set()
    print(f"Generating {min_num_trees} function-trees")
    for i in range(min_num_trees):
        print_progress(i / float(min_num_trees))
        dataset |= construct_random_scalar_functions_branching(params, weighted, max_depth, branches_per_layer=branches_per_layer,
                                                                 constant_chance=constant_chance, constant_range=constant_range,
                                                                 single_variable_chance=single_variable_chance, complex_functions=complex_functions)
    print(f"\rPruning {len(dataset)} functions")
    dataset = prune_function_list(dataset, prune_constants=prune_constants, complex_functions=complex_functions, round_n=round_n, max_wait=max_wait)
    while len(dataset) < num_functions:
        print(f"\rGenerating and pruning an additional function-tree")
        temp_dataset = construct_random_scalar_functions_branching(params, weighted, max_depth, branches_per_layer=branches_per_layer,
                                                                 constant_chance=constant_chance, constant_range=constant_range,
                                                                 single_variable_chance=single_variable_chance, complex_functions=complex_functions)
        dataset |= prune_function_list(temp_dataset, prune_constants=prune_constants, complex_functions=complex_functions, round_n=round_n, max_wait=max_wait)
    print("\rDataset fully generated, saving to file")
    dataset = list(dataset)
    np.random.shuffle(dataset)
    dataset = dataset[:num_functions]
    print(dataset)
    with open(f"{directory_path}/n{num_functions}-x{params}-y1.json", 'w') as f:
        json.dump([str(datum) for datum in dataset], f)
    print(f"Dataset saved to {directory_path}/n{num_functions}-x{params}-y1.json")
    return dataset

def main():
    generate_dataset(
        100,
        2,
        "datasets",
        2,
        weighted=True,
        max_depth=4,
        branches_per_layer=lambda d: 4,
        max_wait=0.5,
    )


if __name__ == "__main__":
    freeze_support()
    main()