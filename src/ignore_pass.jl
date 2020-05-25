function ignore_transform(::Type{<:TraceCtx}, r::Reflection)
    syn = r.code_info.code
    map(syn) do expr
        MacroTools.prewalk(expr) do k
            k isa Expr && k.head == :nooverdub && return
            k isa Expr && k.head == :call && begin
                call = k.args[1]
                if !(call isa GlobalRef && call.name == :rand)
                    k.args[1] = Expr(:nooverdub, call)
                    return
                end
            end
            k
        end
    end
    return r.code_info
end

const ignore_pass = Cassette.@pass ignore_transform
