module InvolutionMCMC

include("../src/Jaynes.jl")
using .Jaynes
using Gen

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

jprop = @jaynes function fixed_structure_proposal(tr)
    if tr[:z] == true
        {:m1} ~ Normal(tr[:m1], 0.1)
        {:m2} ~ Normal(tr[:m2], 0.1)
    else
        {:m} ~ Normal(tr[:m], 0.1)
    end
end

select_mh_structure_kernel(tr) = mh(tr, select([(:z, )]))[1]
fixed_structure_kernel(tr) = mh(tr, jprop, ())[1]

# ------------ Simple MH ------------ #

test = () -> begin
    (y1, y2) = (1.0, 1.3)
    obs = choicemap(Pair{Tuple, Any}[(:y1, ) => y1, 
                                     (:y2, ) => y2,
                                     (:z, ) => false,
                                     (:m, ) => 1.2])
    tr, _ = generate(jmodel, (), obs)
    for iter=1:100
        tr = select_mh_structure_kernel(tr)
        tr = fixed_structure_kernel(tr)
    end
end
test()

# ------------ Involution DSL ------------ #

function merge_means(m1, m2)
    m = sqrt(m1 * m2)
    dof = m1 / (m1 + m2)
    (m, dof)
end

function split_mean(m, dof)
    m1 = m * sqrt((dof / (1 - dof)))
    m2 = m * sqrt(((1 - dof) / dof))
    (m1, m2)
end

sm_prop = @jaynes function split_merge_proposal(tr)
    if tr[:z]
        # currently two segments, switch to one
    else
        # currently one segment, switch to two
        {:dof} ~ Uniform(0, 1)
    end
end

@involution function split_merge_involution(model_args, proposal_args, proposal_retval)

    if @read_discrete_from_model(:z)

        # currently two segments, switch to one
        @write_discrete_to_model(:z, false)
        m1 = @read_continuous_from_model(:m1)
        m2 = @read_continuous_from_model(:m2)
        (m, dof) = merge_means(m1, m2)
        @write_continuous_to_model(:m, m)
        @write_continuous_to_proposal(:dof, dof)

    else

        # currently one segments, switch to two
        @write_discrete_to_model(:z, true)
        m = @read_continuous_from_model(:m)
        dof = @read_continuous_from_proposal(:dof)
        (m1, m2) = split_mean(m, dof)
        @write_continuous_to_model(:m1, m1)
        @write_continuous_to_model(:m2, m2)
    end
end


split_merge_kernel(tr) = mh(tr, sm_prop, (), split_merge_involution)[1]

test_involution = () -> begin
    (y1, y2) = (1.0, 1.2)
    obs = choicemap(Pair{Tuple, Any}[(:y1, ) => y1, 
                                     (:y2, ) => y2,])
    tr, = generate(jmodel, (), obs)
    trs = []
    for iter=1:300
        tr = split_merge_kernel(tr)
        tr = fixed_structure_kernel(tr)
        push!(trs, tr)
    end
    trs
end
trs = test_involution()

display(lineplot(map(trs) do tr
                     tr[:z]
                 end))

end # module
