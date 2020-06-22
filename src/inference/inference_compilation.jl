# --------------- INFERENCE COMPILER --------------- #

Cassette.@context CompileCtx

function reset!(ctx::CompileCtx{M}) where M <: Meta
    ctx.metadata.tr = Trace()
end

mutable struct InferenceCompiler
    spine::Flux.Recur
    decoder_heads::Dict{Address, Chain}
    encoder_heads::Dict{Address, Chain}
    latent_dim::Int
    observation_head::Chain
    function InferenceCompiler(latent_dim::Int)
        rnn = RNN(latent_dim, latent_dim)
        decoder_heads = Dict{Address, Dense}()
        encoder_heads = Dict{Address, Dense}()
        rnn.state = rand(Normal(0.0, 1.0), latent_dim)
        new(rnn, decoder_heads, encoder_heads, latent_dim)
    end
end

# Dynamic generation of encoding and decoding heads.
function make_observation_head!(ic::InferenceCompiler, target::Address, tr::Trace)
    !haskey(tr.chm, target) && return
    obs = [tr.chm[target].val...]
    shape = length(obs)
    head = Chain(Dense(shape, 128), Dense(128, ic.latent_dim))
    ic.observation_head = head
end

function generate_decoder_head!(ic::InferenceCompiler, addr::Address, args::Array{Float64, N}) where {T, N}
    shape = foldr((x, y) -> x * y, size(args))
    head = Chain(Dense(ic.latent_dim, 128), Dense(128, shape))
    ic.decoder_heads[addr] = head
end

function generate_decoder_head!(ic::InferenceCompiler, addr::Address, args::Tuple{Vararg{Float64}})
    shape = length(args)
    head = Chain(Dense(ic.latent_dim, 128), Dense(128, shape))
    ic.decoder_heads[addr] = head
end

function generate_encoder_head!(ic::InferenceCompiler, addr::Address, args::Array{Float64, N}) where {T, N}
    shape = foldr((x, y) -> x * y, size(args))
    head = Chain(Dense(shape, 128), Dense(128, ic.latent_dim))
    ic.encoder_heads[addr] = head
end

function generate_encoder_head!(ic::InferenceCompiler, addr::Address, args::Tuple{Vararg{Float64}})
    shape = length(args)
    head = Chain(Dense(shape, 128), Dense(128, ic.latent_dim))
    ic.encoder_heads[addr] = head
end

# The assumption here is that the trace has been generated by another context already. So we don't need to error check support, or keep a visited list around.
mutable struct InferenceCompilationMeta{T} <: Meta
    tr::Trace
    stack::Vector{Address}
    constraints::Dict{Address, Any}
    target::Address
    opt::T
    compiler::InferenceCompiler
    loss::Float64
    func::Function
    args::Tuple
    ret::Any
    InferenceCompilationMeta(tr::Trace, target::Address; latent_dim = 64) = new{ADAM}(tr, Address[], Address[], Dict{Address, Any}(), target, ADAM(), InferenceCompiler(latent_dim), 0.0)
end

# Inference compilation loss.
function logpdf_loss(dist::Type, head::Chain, rnn::Flux.Recur, sample)
    proposal_args = exp.(head(rnn.state))
    return -logpdf(dist(proposal_args...), sample)
end

function logpdf_loss(dist::Type, head::Chain, rnn::Flux.Recur, obs_head::Chain, val, sample)
    encoding = obs_head(val)
    proposal_args = exp.(head(rnn(obs_head(val))))
    return -logpdf(dist(proposal_args...), sample)
end

@inline function Cassette.overdub(ctx::CompileCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where {M <: InferenceCompilationMeta, 
                                               T <: Address}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr((x, y) -> x => y, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # If in target, return immediately (required by inference compilation objective).
    #addr == ctx.metadata.target && return ctx.metadata.tr.chm[addr].val

    # Check if head is defined - otherwise, generate a new one.
    !haskey(ctx.metadata.compiler.decoder_heads, addr) && begin
        generate_decoder_head!(ctx.metadata.compiler, addr, args)
    end

    # Get args from inference compiler.
    decoder_head = ctx.metadata.compiler.decoder_heads[addr]
    spine = ctx.metadata.compiler.spine
    observation_head = ctx.metadata.compiler.observation_head
    params = Flux.params(decoder_head, observation_head, spine)

    # Get choice from trace choice map.
    choice = ctx.metadata.tr.chm[addr]
    sample = choice.val
    score = choice.score

    # Train.
    if haskey(ctx.metadata.tr.chm, ctx.metadata.target)
        val = ctx.metadata.tr.chm[ctx.metadata.target].val
        loss = s -> logpdf_loss(dist, decoder_head, spine, observation_head, [val...], s)
        Flux.train!(loss, params, [sample], ctx.metadata.opt; cb = () -> ctx.metadata.loss += loss(sample))
    else
        loss = s -> logpdf_loss(dist, decoder_head, spine, s)
        Flux.train!(s -> logpdf_loss(dist, decoder_head, spine, s), params, [sample], ctx.metadata.opt; cb = () -> ctx.metadata.loss += loss(sample))
    end

    # Check if encoder head is available.
    !haskey(ctx.metadata.compiler.encoder_heads, addr) && begin
        generate_encoder_head!(ctx.metadata.compiler, addr, [sample...])
    end

    # Transition.
    encoder_head = ctx.metadata.compiler.encoder_heads[addr]
    ctx.metadata.compiler.spine(encoder_head([sample...]))
    return sample
end

function inference_compilation(model::Function, 
                               args::Tuple,
                               target::Address;
                               batch_size::Int = 512,
                               epochs::Int = 100) where T
    trs = Vector{Trace}(undef, batch_size)
    model_ctx = disablehooks(TraceCtx(metadata = UnconstrainedGenerateMeta(Trace())))
    inf_comp_ctx = disablehooks(CompileCtx(metadata = InferenceCompilationMeta(Trace(), target)))
    for i in 1:epochs
        # Collect a batch.
        for j in 1:batch_size
            # Generate.
            if isempty(args)
                ret = Cassette.overdub(model_ctx, model)
            else
                ret = Cassette.overdub(model_ctx, model, args...)
            end

            # Track.
            trs[j] = model_ctx.metadata.tr
            reset!(model_ctx)
        end

        # Ascent!
        map(trs) do tr
            # Initialize observation head of recurrent model (if uninitialized).
            !isdefined(inf_comp_ctx.metadata.compiler, :observation_head) && begin
                make_observation_head!(inf_comp_ctx.metadata.compiler, target, tr)
            end

            # Optimize.
            inf_comp_ctx.metadata.tr = tr
            if isempty(args)
                ret = Cassette.overdub(inf_comp_ctx, model)
            else
                ret = Cassette.overdub(inf_comp_ctx, model, args...)
            end
            inf_comp_ctx.metadata.stack = Vector{Address}[]
        end
        println("Epoch loss: $(inf_comp_ctx.metadata.loss/batch_size)")
        inf_comp_ctx.metadata.loss = 0.0
    end
    ctx = disablehooks(TraceCtx(metadata = inf_comp_ctx.metadata))
    ctx.metadata.func = model
    ctx.metadata.args = args
    return ctx
end

@inline function Cassette.overdub(ctx::TraceCtx{M}, 
                                  call::typeof(rand), 
                                  addr::T, 
                                  dist::Type,
                                  args) where {M <: InferenceCompilationMeta, 
                                               T <: Address}
    # Check stack.
    !isempty(ctx.metadata.stack) && begin
        push!(ctx.metadata.stack, addr)
        addr = foldr((x, y) -> x => y, ctx.metadata.stack)
        pop!(ctx.metadata.stack)
    end

    # Get heads.
    encoding_head = ctx.metadata.compiler.encoder_heads[addr]
    spine = ctx.metadata.compiler.spine
    observation_head = ctx.metadata.compiler.observation_head
    decoding_head = ctx.metadata.compiler.decoder_heads[addr]
    encoding = observation_head([ctx.metadata.constraints[ctx.metadata.target]...])
    trans = spine(encoding)
    proposal_args = exp.(decoding_head(spine(encoding)))
    d = dist(proposal_args...)

    if addr == ctx.metadata.target
        sample = ctx.metadata.constraints[addr]
        spine(encoding_head([sample...]))
        return sample
    else
        sample = rand(d)
        score = Float64(logpdf(d, sample))
        ctx.metadata.tr.chm[addr] = Choice(sample, score)
        ctx.metadata.tr.score += score
        spine(encoding_head([sample...]))
        return sample
    end
end
