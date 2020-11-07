module Examples

blacklist = ["runexamples.jl", "trace_translation.jl", "support_checks.jl"]

for p in readdir("examples"; join=false)
    !(p in blacklist) && include(p)
end

end # module
