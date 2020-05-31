module InfComp

using Flux
using Zygote
using Distributions
using DistributionsAD

model = Dense(10, 2)
rnn = RNN(10, 10)
params = Flux.params(model, rnn)

function logpdf_loss(model, rnn, sample)
    proposal_args = exp.(model(rnn.state))
    println(proposal_args)
    return logpdf(Normal(proposal_args...), sample)
end

sample = 0.5
gs = gradient(() -> logpdf_loss(model, rnn, sample), params)
println(gs.grads)

end # module
