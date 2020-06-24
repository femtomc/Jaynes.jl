# Allows updating of CodeInfo instances with new IR.
function update!(ci::Core.CodeInfo, ir::Core.Compiler.IRCode)
  Core.Compiler.replace_code_newstyle!(ci, ir, length(ir.argtypes)-1)
  ci.inferred = false
  ci.ssavaluetypes = length(ci.code)
  slots!(ci)
  fill!(ci.slotflags, 0)
  return ci
end

function update!(ci::Core.CodeInfo, ir::IR)
  if ir.meta isa Meta
    ci.method_for_inference_limit_heuristics = ir.meta.method
    if isdefined(ci, :edges)
      ci.edges = Core.MethodInstance[ir.meta.instance]
    end
  end
  update!(ci, Core.Compiler.IRCode(slots!(ir)))
end

# This is a pass which indicates to native Cassette passes that the only thing of interest is calls to rand.
function ignore_transform(::Type{<:TraceCtx}, r::Reflection)
    syn = r.code_info.code
    new_code = Expr[]
    for k in syn
        if k isa Expr && k.head == :call
            arg = k.args[1]
            if !(arg isa Expr && arg.head == :nooverdub) && !(arg isa GlobalRef && arg.name == :rand)
                k.args[1] = Expr(:nooverdub, arg)
            end
        end
    end
    return r.code_info
end

const ignore_pass = Cassette.@pass ignore_transform
