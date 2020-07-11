#!/bin/bash
julia --project=. -e 'using Pkg; Pkg.add("Test"); Pkg.instantiate(); Pkg.test(); Pkg.rm("Test")'
