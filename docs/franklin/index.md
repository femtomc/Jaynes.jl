@def title = "E.T. Jaynes home phone"
@def tags = ["probabilistic programming", "programmable inference"]

**Jaynes.jl** (Jaynes) is a domain-specific compiler for [the generative function interface of Gen.jl](https://www.gen.dev/dev/ref/gfi/#Generative-function-interface-1)[^1]. The modeling language is (mostly) all of Julia.

```julia
jmodel = @jaynes function model()
    z = ({:z} ~ Bernoulli(0.5))
    if z
        m1 = ({:m1} ~ Gamma(1, 1))
        m2 = ({:m2} ~ Gamma(1, 1))
    else
        m = ({:m} ~ Gamma(1, 1))
        (m1, m2) = (m, m)
    end
    {:y1} ~ Normal(m1, 0.1)
    {:y2} ~ Normal(m2, 0.1)
end
```

The interfaces between Julia code and modeling code is intentionally kept very minimal.

```julia
function model()
    z = rand(:z, Bernoulli(0.5))
    if z
        m1 = rand(:m1, Gamma(1, 1))
        m2 = rand(:m2, Gamma(1, 1))
    else
        m = rand(:m, Gamma(1, 1))
        (m1, m2) = (m, m)
    end
    y1 = rand(:y1, Normal(m1, 0.1))
    y2 = rand(:y2, Normal(m2, 0.1))
end
jmodel = JFunction(model)
```

> This package is in open alpha. Expect some bumps, especially as [new compiler interfaces](https://github.com/Keno/Compiler3.jl) stabilize in Julia `VERSION` > 1.6.

---

[^1]: Roughly, this interface describes the set of capabilities which, when implemented for a model class, allows for the construction of customizable sampling-based inference algorithms. This idea originally appeared under _stochastic procedure interface_ in [Venture](https://arxiv.org/abs/1404.0099).
