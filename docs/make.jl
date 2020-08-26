using Documenter
using Jaynes

makedocs(sitename = "Jaynes.jl",
         pages = ["Introduction" => "index.md",
                  "Traces, choices, and call sites" => "library_api/sites.md",
                  "Execution contexts" => "library_api/contexts.md",
                  "Selection interface" => "library_api/selection_interface.md",
                  "Inference" => ["Importance sampling" => "inference/is.md",
                                  "Metropolis-Hastings" => "inference/mh.md",
                                  "Hamiltonian Monte Carlo" => "inference/hmc.md",
                                  "Particle filtering" => "inference/pf.md",
                                  "Automatic differentiation variational inference" => "inference/vi.md"],
                  "Foreign model interface" => "library_api/fmi.md",
                  "Differentiable programming" => "library_api/diff_prog.md",
                 ],
         format = Documenter.HTML(prettyurls = true,
                                  assets = ["assets/favicon.ico"]),
         clean = true,
         doctest = true,
         build = "franklin/library_api"
         )
