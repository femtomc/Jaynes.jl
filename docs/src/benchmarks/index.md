This page contains a set of benchmarks comparing Jaynes to other probabilistic programming systems. [The code for each of these benchmarks is available here.](https://github.com/femtomc/JaynesBenchmarks)

!!! warning
    Jaynes is currently in open alpha, which means that these benchmarks should not be taken seriously (any benchmarks between systems _must always_ be taken with a grain of salt anyways) until the system is fully tested.

    Furthermore, I'm not in the business of inflating the performance characteristics of systems I build. This means:

    1. If you notice that I'm not testing Jaynes against optimized programs in other systems, _please let me know_ so that I can perform accurate comparisons.
    2. If you suspect that there's an issue or bug with Jaynes, [please open an issue.](https://github.com/femtomc/Jaynes.jl/issues)
    3. If you'd like to perform a benchmark, also [please open an issue.](https://github.com/femtomc/Jaynes.jl/issues)

    Benchmarking is an inherently sensitive topic - I'd like to make these as fair and open as possible, so don't hesitate to reach out.

## Particle filtering in hidden Markov models

This benchmark is a single-shot time comparison between `Gen` and `Jaynes` on a single thread. The horizontal axis is number of particle filter steps (with resampling). The vertical axis is time in seconds.

```@raw html
<div style="text-align:center">
    <img src="../images/benchmark_hmmpf_gen_singlethread.png" alt="" width="70%"/>
</div>
```

This benchmark is a single-shot time comparison between `Gen` and `Jaynes` with adaptive multi-threading in `Jaynes.filter_step!` and `Jaynes.importance_sampling!`. This benchmark was executed with `JULIA_NUM_THREADS=4`.

```@raw html
<div style="text-align:center">
    <img src="../images/benchmark_hmmpf_gen_multithread.png" alt="" width="70%"/>
</div>
```

The bottom plots are the estimated log marginal likelihood of the data.
