@def title = "Compilation"

> This is preliminary work! This system is designed to be exploratory in the direction of compiling and optimizing probabilistic programs (especially universal ones). Thus, this page will likely evolve over time.

As a research system, the goal of Jaynes is to facilitate the exploration of compilation techniques for probabilistic programs. This includes static techniques (facilitated by dataflow analysis and abstract interpretation) and partial evaluation/specialization for inference ops. To approach these goals, Jaynes implements the generative function interface - but it does so by implementing each interface as staged programming pipeline. For interface methods which operate as lightweight tracing on top of normal program execution (e.g. `simulate`, `generate`, `assess`, and `propose`) - this staging is not yet utilized (although it's certainly possible there are useful optimization for these methods).

The methods `regenerate` and `update` are used for iterative inference algorithms like Metropolis-Hastings and sequential Monte Carlo. It is highly important that these interface methods are efficient - because the user is likely to call them many times during the deployment of iterative inference algorithms. [To specialize these operations, Jaynes uses a combination of static analysis techniques.](https://femtomc.github.io/mrb_dynamic_specialization.pdf) [A simple "flow chart" style version is also available here.](https://femtomc.github.io/mrb_dynamic_specialization_prez.pdf)

Leaving a discussion of these techniques to the links above, here's a speculative list of other interesting "static analysis" questions which might be interesting to explore:

1. Representation transformations - this is the category of optimizations which alter the representation of the model program $P$ irrespective of inference operation or observations.

2. Specialization transformations - this is the category of the optimizations discussed above. Roughly, this category likely looks like the following: given a tuple $(P, op, obs)$ - where $P$ is a model, $op$ is an inference operation, and $obs$ is a set of observations, specialize $op$.

3. Intermediate representations - currently, Jaynes operates using an SSA form IR representation provided by [IRTools.jl](https://github.com/FluxML/IRTools.jl). Is this the most convenient representation for the above transformations?

4. Compiler hints and errors - given access to the IR for a model, we might provide analysis-driven hints to the user which helps them author models which are easier to optimize. Additionally, one could imagine more complicated analysis tools - [like type inference based upon trace types](https://dl.acm.org/doi/10.1145/3371087) which operate during model staging and become associated with the model for use in inference.
