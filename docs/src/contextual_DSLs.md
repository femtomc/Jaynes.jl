Often times, developers of machine learning systems must construct domain-specific languages to express restrictions which are required by the system, but can't be expressed solely through the type system. An excellent example is MCMC kernel DSLs (or Gaussian process kernel DSLs) where a set of mathematical restrictions (which cannot be enforced by the compiler) must apply to user code for the code to qualify as valid in the kernel interpretation. Restricted languages which enable efficient (and cohesive) automatic differentiation are another example. 

One of the dominant methodologies in languages with macro systems is to construct the DSL using macros. The library can check at macro-expansion time if the user has utilized syntax or language features which are disallowed by the DSL, returning immediately to the user to express an error or issue. In extreme cases, the user may only be allowed to use other macros (which represent primitive features of the DSL) inside the DSL macro. Because Jaynes aims for a broad standard of composability as a plugin to the compiler, we prefer a complementary viewpoint which "carves" the DSL out of the host language. This viewpoint can also be found in the philosophy of packages such as [Zygote](https://github.com/FluxML/Zygote.jl) where the user is allowed to write arbitrary code, but may encounter a runtime error if Zygote is unable to identify and emit pullback code for a method call (i.e. the user has stepped outside the bounds of the language (or set of `ChainRules`) which the system knows how to differentiate). 

We call our approach to this viewpoint _contextual domain-specific languages_ because the inclusion of any language feature representable by a method call is handled by the interpretation context. The interpretation context contains a "language core" which is a piece of metadata which tells the interpreter what method calls are allowed in the DSL. These cores have a natural set of operations: set intersection is the minimal feature set which is compatible with both languages, where union is the set which covers both. Interpretation without a core is just interpretation of the entire host language in the context - nothing is excluded.

In our system, these languages are only active for specific contexts (i.e. those associated with the validation of construction of kernels or programming in the graphical model DSL) - so this section is purely optional. These languages are designed to help a user construct a program which is valid according to the assumptions of the domain - but they incur a small runtime cost (equivalent to normal `Cassette` execution with a call to `prehook` before `overdub`). For inference contexts, these language restrictions can be turned on and off by the user.

---

As an example of this idea, here's a small functional core which performs runtime checks to prevent the use of mutation on mutable structures or key-accessed collections:

```julia
ctx = DomainCtx(metadata = Language(BaseLang()))

mutable struct Foo
    x::Float64
end

function foo(z::Float64)
    z = Foo(10.0)
    x = 10
    if x < 15
        y = 20
    end
    y += 1
    z.x = 10.0
    return y
end

# Accepted!
ret = interpret(ctx, foo, 5.0)

@corrode! BaseLang setfield!
@corrode! BaseLang setproperty!
@corrode! BaseLang setindex!

# Rejected!
ret = interpret(ctx, foo, 5.0)

# ERROR: LoadError: Main.LanguageCores.Jaynes.BaseLangError: setproperty! with Tuple{Main.LanguageCores.Foo,Symbol,Float64} is disallowed in this language.
```

`BaseLang` lets all calls through. We corrode `BaseLang` to prevent calls to `setfield!`, `setproperty!`, and `setindex!`. Note that, at the method level, we can't prevent re-assignment to variables because assignment is not a method. If we wanted to, we could prevent this using an IR pass. We could extend this to prevent iteration-based control flow:

```julia
function foo(z::Float64)
    z = Foo(10.0)
    x = 10
    for i in 1:10
        println(i)
    end
    if x < 15
        y = 20
    end
    y += 1
    z.x = 10.0
    return y
end

@corrode! BaseLang iterate

# Rejected!
ret = interpret(ctx, foo, 5.0)

# ERROR: LoadError: Main.LanguageCores.Jaynes.BaseLangError: iterate with Tuple{UnitRange{Int64}} is disallowed in this language.
```
