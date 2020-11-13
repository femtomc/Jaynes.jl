# ------------ Identify if-else patterns ------------ #

struct SwitchHint <: ProgramStructureHint
    blocks::Set{Block}
end
function Base.display(sh::SwitchHint)
    if isempty(sh.loops)
        println("\u001b[32m ✓ (SwitchHint): Compiler detected no branching in your model code.\u001b[0m")
    else
        println("\u001b[33m (SwitchHint): Compiler detected the following switch (if-else) patterns in your model code.\u001b[0m")
        println("________________________\n")
        for b in sh.loops
            for b in sh.mapped_ir[l]
                display(b)
            end
        end
        println("________________________\n")
        println("\u001b[32m (Recommendation): extract kernels into a Fold combinator for easier analysis and optimization.")
    end
end

function detect_switches(ir::IR)
    cfg = CFG(ir)
end

function detect_kernels(fn, arg_types...)
    ir = lower_to_ir(fn, arg_types...)
    display(detect_switches(ir))
end
