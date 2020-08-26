#!/bin/bash

julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs/ docs/make.jl
julia --project=docs/franklin/ -e 'cd("docs/franklin"); println(pwd()); using Franklin; Franklin.optimize()'
cd docs/
mv franklin/__site website
cd website
git init
git remote add origin https://github.com/femtomc/Jaynes.jl
git checkout -b gh-pages
git add .
git commit -a -m "Publish website."
git push -f origin gh-pages
cd ..
rm -rf franklin/library_api
rm -rf website
