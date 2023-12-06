import numpy as np
import time
import random
import copy

# making a problem instance
def make_grid_python(n):
    grid = np.empty((n ** 2, n ** 2), int)
    x = 0
    for i in range(n):
        for j in range(n):
            for k in range(n ** 2):
                grid[n * i + j, k] = x % (n ** 2) + 1
                x += 1
            x += n
        x += 1
    return grid


def make_grid_numpy(n):
    return np.fromfunction(lambda i, j: (i * n + i // n + j) % (n ** 2) + 1, (n ** 2, n ** 2), dtype=int)
    # a comparison between native python and numpy


def get_diff(i) -> int:
    diff = -1
    if i < 3:
        diff = 0
    elif i < 6:
        diff = 3
    else:
        diff = 6
    return diff


# vary n to see their performances
# test
grid = make_grid_numpy(3)
grid

class Sudoku:
    @classmethod
    def create(cls, n, seed=303):
        rng = np.random.default_rng(seed)
        init_grid = make_grid_numpy(n)

        # randomly mask out some cells to create a problem instance
        # cells marked by *1* is given and fixed
        mask = rng.integers(0, 2, size=init_grid.shape)
        grid = init_grid * mask

        return cls(n, mask, grid, seed)

    def __init__(self, n, mask, grid, seed) -> None:
        self.seed = seed
        self.mask = mask
        self.grid = grid
        self.n = n
        self.all = set(range(1, n ** 2 + 1))
        self.can_swap = self.swap_gen()

    def swap_gen(self):
        can_swap = []
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                sub_grid = []
                for row in range(i, i + 3):
                    for col in range(j, j + 3):
                        if self.mask[row][col] == 0:
                            sub_grid.append((row, col))
                can_swap.append(sub_grid)

        for sub in can_swap:
            if len(sub)<2:
                can_swap.remove(sub)
        return can_swap

    def value(self):
        cost = 0
        for i in range(9):
            nums = []
            for j in range(9):
                if nums != 0:
                    if self.grid[i][j] in nums:
                        cost = cost + 1
                    else:
                        nums.append(self.grid[i][j])

        for i in range(9):
            nums = []
            for j in range(9):
                if nums != 0:
                    if self.grid[j][i] in nums:
                        cost = cost + 1
                    else:
                        nums.append(self.grid[j][i])
        return cost

    def local_search(self):
        sub_id = random.randint(0, len(self.can_swap)-1)
        sub_len = len(self.can_swap[sub_id])
        numbers = list(range(sub_len))
        sample = random.sample(numbers, 2)

        i1, j1 = self.can_swap[sub_id][sample[0]]
        i2, j2 = self.can_swap[sub_id][sample[1]]

        next_state = copy.deepcopy(self)

        next_state.grid[i1][j1], next_state.grid[i2][j2] = self.grid[i2][j2], self.grid[i1][j1]
        return next_state

    def init_solution(self):
        rng = np.random.default_rng(self.seed)
        n = self.n
        grid = self.grid.reshape(n, n, n, n).transpose(0, 2, 1, 3)
        for I in np.ndindex(n, n):
            idx = grid[I] == 0
            grid[I][idx] = rng.permutation(list(self.all - set(grid[I].flat)))
        return self

    def __repr__(self) -> str:
        return self.grid.__repr__()


# test
sudoku = Sudoku.create(3)
sudoku.init_solution()
sudoku, sudoku.value()


def simulated_annealing(initial: Sudoku, schedule, halt, log_interval=200):
    state = initial.init_solution()
    t = 0  # time step
    T = schedule(t)  # temperature
    f = [state.value()]  # a recording of values
    while not halt(T):
        T = schedule(t)
        new_state = state.local_search()
        new_value = new_state.value()
        # TODO: implement the replacement here
        if new_value < state.value() or random.random() < np.exp((state.value() - new_value) / T):
            state = new_state
            f.append(state.value())

        # update time and temperature
        if t % log_interval == 0:
            print(f"step {t}: T={T}, current_value={state.value()}")
        if new_value == 0:
            break
        t += 1
        T = schedule(t)
    print(f"step {t}: T={T}, current_value={state.value()}")
    return state, f
#%%
import matplotlib.pyplot as plt

# define your own schedule and halt condition
# run the algorithm on different n with different settings
n = 3
solution, record = simulated_annealing(
    initial=Sudoku.create(n),
    schedule=lambda t: 0.999 ** t,
    halt=lambda T: T < 1e-7
)
solution, solution.value()

# visualize the curve
plt.plot(record)
plt.xlabel("time step")
plt.ylabel("value")