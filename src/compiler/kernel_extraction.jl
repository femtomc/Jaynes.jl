# ------------ Identify blocks which are viable as kernels ------------ #

struct KernelHint <: ProgramStructureHint
    loops::Set{NaturalLoop}
    mapped_ir::Dict{NaturalLoop, Set{Block}}
end
function Base.display(kh::KernelHint)
    if isempty(kh.loops)
        println("(KernelHint): Compiler detected no kernels in your model code.")
    else
        println("(KernelHint): Compiler detected the following kernels in your model code.")
        for l in kh.loops
            for b in kh.mapped_ir[l]
                display(b)
            end
        end
        println("Recommendation: extract kernels into a Fold combinator for easier analysis and optimization.")
    end
end

function detect_kernels(ir)
    loops = detectloops(ir)
    kh = KernelHint(Set{NaturalLoop}([]), Dict{NaturalLoop, Set{Block}}())
    for l in loops
        length(l.backedges) > 1 && continue
        backedge = collect(l.backedges)[1]
        back_blk = block(ir, backedge)
        header = l.header
        head_blk = block(ir, header)
        any(map(branches(back_blk)) do br
                br.block == header && length(br.args) == length(arguments(head_blk))
            end) || continue
        push!(kh.loops, l)
        kh.mapped_ir[l] = Set(map(collect(l.body)) do ind
                                  block(ir, ind)
                              end)
    end
    kh
end
