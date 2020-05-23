<p align="center">
<img height="250px" src="img/walkman.jpeg"/>
</p>
<br>

_Walkman_ is a probabilistic programming framework which uses IR transformations to implement the core routines for modeling and inference.

Currently, _Walkman_ supports a dynamic modeling DSL which is syntactically close (and semantically equivalent) to the dynamic DSL in [Gen](https://www.gen.dev/). It is _unoptimized_ and does not feature things like combinators (for those familiar with Gen) to improve performance. This will likely improve with time.

Currently supported inference algorithms for this DSL:
- [X] Importance sampling
- [ ] Programmable MCMC
- [ ] HMC
- [ ] Metropolis-Hastings
- [ ] Particle filtering

_Walkman_ also aims to support a restricted _graph-based_ DSL which allows the user to utilize graphical model inference algorithms. This is a WIP, and requires a bit more research at the IR level.

## Other notes

The motivation for this project is to identify interfaces and techniques to combine programmable inference with graphical model inference. These techniques have complementary strengths and weaknesses - programmable sampling algorithms tend to have difficulties in high-dimensions (but can answer joint queries about a model efficiently when they are efficient) whereas the asymptotic complexity of graphical model algorithms is typically not dependent on the dimensionality of the model (and instead depends on the topology of the dependence graph) but queries are typically restricted to be marginal queries.


Work in progress :)

---
