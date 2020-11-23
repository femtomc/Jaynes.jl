# ------------ Identify if-else patterns ------------ #

struct SwitchHint <: ProgramStructureHint
    blocks::Set{Tuple{Block, Block}}
    SwitchHint() = new(Set{Block}([]))
end
function Base.display(sh::SwitchHint)
    if isempty(sh.blocks)
        println("\u001b[32m ✓ (SwitchHint): Compiler detected no branching in your model code.\u001b[0m")
    else
        println("\u001b[33m (SwitchHint): Compiler detected the following switch (if-else) patterns in your model code.\u001b[0m")
        println("________________________\n")
        for (b1, b2) in sh.blocks
            display(b1)
            display(b2)
        end
        println("________________________\n")
        println("\u001b[32m (Recommendation): extract kernels into a Switch combinator for easier analysis and optimization.")
    end
end

function detect_switches(ir::IR)
    cfg = CFG(ir)
    sh = SwitchHint()
    for (ind, tars) in enumerate(cfg.graph)
        count(==(tars), cfg.graph) == 1 || continue
        length(tars) > 1 || continue
        all(map(tars) do t
                ind < t
            end) && push!(sh.blocks, tuple(map(b -> block(ir, b), sort(tars))...))
    end
    sh
end

function detect_switches(fn, arg_types...)
    ir = lower_to_ir(fn, arg_types...)
    display(detect_switches(ir))
end
