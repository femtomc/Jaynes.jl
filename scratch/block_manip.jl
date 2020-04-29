module BlockManip

include("../src/Jaynes.jl")
using .Jaynes

using IRTools
using IRTools: func, IR, block!, argument!, branch!, renumber, push!, Variable, blocks, xcall, insertafter!, @code_ir, Statement, branches, Block, Branch, isreturn, func, deletearg!
using MacroTools
using MacroTools: postwalk

using Distributions
using DistributionsAD: logpdf

using Zygote: gradient

function foo_det()
    if y > 1.0
        z = y + 1
    else
        z = 5
    end
    return z
end

ir = @code_ir foo_det()
println("--- IR (foo_det) ---\n$(ir)\n")

# Simple control flow.
function foo()
    y = rand(Normal(0, 1))
    if y > 1.0
        z = rand(Normal(0, 1))
        rand(Normal(5.0, 10.5))
    else
        z = rand(Normal(5, 10))
    end
    return z
end

ir = @code_ir foo()
transformed = @code_ir logpdf_transform! foo()
println("--- IR (foo) ---\n$(ir)\n")
println("Transformed:\n$(transformed)\n")

# Bad control flow.
function foo2()
    z = 0.0
    for i in 1:5
        z = z + rand(Normal(0, 1))
    end
    return z
end

# ----

function block_successors(b::Block, var::Variable)
    succ = Variable[]
    for (v, stmt) in b
        MacroTools.postwalk(x -> (x isa Variable && x == var) ? (push!(succ, v); x) : x, stmt)
    end
    return succ
end

function substitute_var(stmt::Statement, from::Variable, to::Variable)
    stmt = MacroTools.postwalk(x -> (x isa Variable && x == from) ? to : x, stmt)
    return stmt
end

function block_transform(ir)
    ir = copy(ir)
    for bb in reverse(blocks(ir))
        log_tracks = Variable[]
        for (v, stmt) in bb
            stmt.expr.head == :call &&
            stmt.expr.args[1] isa GlobalRef &&
            stmt.expr.args[1].name == :rand &&
            begin
                succ = block_successors(bb, v)
                new = argument!(ir)
                new_stmt = Statement(
                                  Expr(:call, 
                                       GlobalRef(stmt.expr.args[1].mod, :logpdf),
                                       stmt.expr.args[2:length(stmt.expr.args)]..., new),
                                  stmt.type,
                                  stmt.line
                                 )
                ir[v] = new_stmt
                map(x -> ir[x] = substitute_var(ir[x], v, new), succ)
                push!(log_tracks, v)
            end
        end

        if !isempty(log_tracks)
            v = log_tracks[1]
            if length(log_tracks) > 1
                v = insertafter!(ir, log_tracks[length(log_tracks)], xcall(:+, log_tracks...))
            else
                v = insertafter!(ir, log_tracks[length(log_tracks)], v)
            end
            
            bbranches = branches(bb)
            pass_in = []
            if !(foldl((x, y) -> x && y, map(x -> isreturn(x), bbranches))) && bb.id > 1
                pass_in = [argument!(bb)]
                ir[v] = xcall(:+, log_tracks..., pass_in...)
            end
            for (i, bran) in enumerate(bbranches)
                new_br = Branch(bran.condition, bran.block, [v])
                bbranches[i] = new_br
            end
        end
    end
    deletearg!(ir, 1)
    ir = renumber(ir)
    ir
end

ir = @code_ir foo()
println("--- IR (foo2) ---\n$(ir)\n")
new_ir = block_transform(ir)
println(new_ir)
fn = func(new_ir)

println(fn(3.0, 3.0, 3.0, 3.0))

end #module
