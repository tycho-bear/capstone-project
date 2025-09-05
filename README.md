# Capstone Project: Combinatorial and Engineering Optimization with Metaheuristic Algorithms

---

# What is this?

Many interesting problems cannot be solved efficiently by exact algorithms. One such example is the traveling salesman problem.
Additionally, some engineering problems, such as the pressure vessel design problem, are difficult to solve analytically.
In these cases, metaheuristic algorithms may be used to find high-quality solutions in an efficient manner.
This work applies metaheuristic algorithms to solve NP-hard combinatorial optimization problems, as well as engineering optimization problems.


---

# Requirements

Both `numpy` and `matplotlib` are required to run the code. To install them, run

```bash
pip install -r requirements.txt
```


# Usage

This project can be run straight from the command line, as well as through Docker.

## Command Line

To run the code from the command line, install the requirements, then run 

```bash
python main.py <ALGORITHM> <PROBLEM>
```

where

`<ALGORITHM>` = `sa`, `ga`, or `pso`

`<PROBLEM>` = `tsp-grid`, `tsp-random`, `bpp`, or `pvd`

See the "Algorithms Used" and "Problems Examined" subsections for more information.

An example run (and a good place to start) might look like:

```bash
python main.py pso pvd
```

---

## Docker

This project can also be run with Docker. The simplest way is through `make`,
but the actual commands are provided as well.

### Makefile

The project must be built before running. To do this, run 

```bash
make build
```

Alternatively, to build and run the project all at once, run

```bash
make start
```

To simply run the project after building, run

```bash
make run
```

### Additional Arguments

To specify an algorithm and problem, put `ARGS="<ALGORITHM> <PROBLEM>"` at the
end of the command. For example:

```bash
make run ARGS="sa pvd"
```

### Docker Commands

Instead of running `make ...`, the raw Docker commands may be used. To build the
project this way, run

```bash
docker build -t capstone-project .
```

To run, execute either

```bash
docker run capstone-project
```

or, to specify an algorithm and problem,

```bash
docker run capstone-project <ALGORITHM> <PROBLEM>
```

---


## Additional information

### Full Report

The full report for this project may be found in [Capstone_Project_Report.pdf](https://github.com/tycho-bear/capstone-project/blob/main/Capstone_Project_Report.pdf).

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





