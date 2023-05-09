# Quasi-deterministic Hidden Markov Models.

Stochastic models such as Hidden Markov Models (HMM) are widely used to model the
temporal evolution of a process. In a HMM, the state duration is a-priori and implicitly
assumed to be geometrically distributed in order to make the underlying process Markovian.

However, this assumption does not always hold. Existing HMM state duration modelling
methods are reviewed and their drawbacks in the context of load modelling are revealed.
This thesis aims to address their drawbacks in a specific context by proposing a
Quasi-Deterministic Hidden Markov Model (QDHMM). Specifically, we extend the HMM to
model sequential data where the state durations follow a truncated distribution and the
dynamics of the model are dependant on whether the truncation was reached.

We formalize the model and adapt the Expectation Maximization (EM) algorithm to
estimate maximum likelihood solutions of the model parameters. To obtain good initial
estimates for the QDHMM EM algorithm, a distribution free method is developed to obtain
expected values of state durations in a HMM. To drive the EM algorithm towards a good
solution space, combinatorial optimization heuristics and meta-heuristics are researched.
Simulated annealing is identified as a solution and a heuristic is developed to sample
candidate solutions which lead to a good approximation of the global optimum.

Experiments were performed on modelling the internal electrical power consumption
characteristic of printers based on real power data. The QDHMM is shown to provide an
accurate descriptive model in comparison to the standard HMM without loss of parsimony.
