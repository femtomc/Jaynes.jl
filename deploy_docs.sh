#!/bin/bash

julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs/ docs/make.jl
julia --project=docs/website/ -e 'cd("docs/website"); println(pwd()); using Franklin; Franklin.serve()'
git subtree push --prefix docs/website/__site origin gh-pages
rm -rf docs/website/__site
