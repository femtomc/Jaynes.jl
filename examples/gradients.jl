module Gradients

include("../src/Jaynes.jl")
using .Jaynes
using Distributions
using Flux

function foo1()
    # Literals are tracked as trainable.
    x = rand(:x, 10.0)
    m = rand(:m, 5.0)
    t = rand(:t, 7.0)

    # Rand calls on distributions also get tracked and the dependency graph is created.
    y = rand(:y, Normal, (x, 1.0))
    z = rand(:z, Normal, (t, 3.0))
    for i in 1:10
        q = rand(:q => i, Normal, (m, 1.0))
    end
    return z
end

function foo2()
    x = rand(:x, 1.0)
    y = rand(:y, Normal, (x, 1.0))
    z = rand(:z, Normal, (x + 10, 13.0))
    for i in 1:10
        q = rand(:q => i, Normal, (x, 1.0))
    end
    return z
end

trainer = () -> begin
    train_ctx = Gradient()
    gen_ctx = Generate(Trace())
    for i in 1:10000
        gen_ctx, tr, _ = trace(gen_ctx, foo2)
        train_ctx.metadata.tr = tr
        train_ctx = trace(train_ctx, foo1)
        Jaynes.update!(ADAM(), train_ctx)
        println(train_ctx.metadata.loss)
        train_ctx.metadata.loss = 0.0
        reset_keep_constraints!(gen_ctx)
    end
end
trainer()

end # module
