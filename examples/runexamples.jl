module Examples

for p in readdir("examples"; join=false)
   p != "runexamples.jl" && include(p)
end

end # module
