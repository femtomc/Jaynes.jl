module Examples

for p in readdir("examples"; join=false)
    include(p)
end

end # module
