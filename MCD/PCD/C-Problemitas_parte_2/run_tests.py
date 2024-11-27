from pathlib import Path
from dna import solve

def read_lines(path):
    with open(Path(__file__).resolve().parent.joinpath(path)) as f:
        return [line.strip().split() for line in f]


def load_inputs(i):
    [[genome], [k, L, t]] = read_lines(f'./inputs/input_{i}.txt')
    k = int(k)
    L = int(L)
    t = int(t)
    return genome, k, L, t

def load_outputs(i):
    [patterns] = read_lines(f"./outputs/output_{i}.txt")
    return set(patterns)

def run_test(i):
    genome, k, L, t = load_inputs(i)
    patterns = load_outputs(i)
    if patterns != solve(genome, k, L, t):
        print(f"Test {i} FAILED!")
        return
    print(f"Test {i} ok")
    
if __name__ == "__main__":
    run_test(1)
    run_test(2)
    run_test(3)
    run_test(4)
    # run_test(5)
