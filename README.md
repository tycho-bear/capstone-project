# Capstone Project: Combinatorial and Engineering Optimization with Metaheuristic Algorithms

---

## What is this?

Many interesting problems cannot be solved efficiently by exact algorithms. One such example is the traveling salesman problem.
Additionally, some engineering problems, such as the pressure vessel design problem, are difficult to solve analytically.
In these cases, metaheuristic algorithms may be used to find high-quality solutions in an efficient manner.
This work applies metaheuristic algorithms to solve NP-hard combinatorial optimization problems, as well as engineering optimization problems.


---

## Requirements

Both `numpy` and `matplotlib` are required to run the code. To install them, run

```bash
pip install -r requirements.txt
```


## Usage

To run the code, run 

```bash
python main.py <ALGORITHM> <PROBLEM>
```

`<ALGORITHM>` = `sa`, `ga`, or `pso`

`<PROBLEM>` = `tsp-grid`, `tsp-random`, `bpp`, or `pvd`

See the "Algorithms Used" and "Problems Examined" subsections for more information.

---


## Additional information

### Full Report

The full report for this project may be found in [Capstone_Project_Report.pdf]([url](https://github.com/tycho-bear/capstone-project/blob/main/Capstone_Project_Report.pdf)).

### Algorithms Used

Simulated annealing (SA), genetic algorithms (GA), and particle swarm optimization (PSO) are implemented and evaluated.
SA iteratively improves a given solution, occasionally accepting worse solutions to escape local optima.
GA maintains a population of solutions that evolve through crossover and mutation operators, mimicking real populations in nature.
PSO simulates "swarm intelligence" behavior to find good solutions.

### Problems Examined

These algorithms are tested on benchmark instances of the traveling salesman problem (TSP), bin packing problem (BPP), and pressure vessel design problem (PVD).
Results show that properly tuned configurations achieve solutions within 1-10% of optimal or best known results.
For the pressure vessel problem, solutions consistent with the best in the literature are found.
These results demonstrate the effectiveness of metaheuristics for finding high-quality solutions to optimization problems.





