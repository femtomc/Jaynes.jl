#!/usr/bin/env julia

# Check if Revise is installed.
using Pkg
try 
    Pkg.status("Revise")
catch e
error("\n\033[0;31mRevise not installed to your global environment. Please install Revise.\033[0m\n\033[0;32mUse 'using Pkg; Pkg.add(\"Revise\")' at REPL.\033[0m\n")
end
using Revise

# Activate env.
Pkg.activate(".")

# Works on *nix systems.
cmd = `find -name '*.jl'`

# Filenames get passed in as CL args.
include_filenames = Base.ARGS

# Uses Revise to watch for all updates to Julia files in repo.
entr(split(read(cmd, String), "\n")[1:end-1]) do
println("\n--------------------------------\n")
    if include_filenames[1] == "test"
        try
            Pkg.test()
        catch e
            println("\n\033[0;31mCaught error in test.\033[0m\n")
            println(e)
        end
    else
        map(include_filenames) do path
            try
                include(path)
            catch e
                println("\n\033[0;31mCaught error in $(path).\033[0m\n")
                for (exc, bt) in Base.catch_stack()
                   showerror(stdout, exc, bt)
                   println()
               end
            end
        end
    end
end
