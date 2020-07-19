using Documenter
using Jaynes

makedocs(sitename = "Jaynes.jl",
         pages = ["Introduction" => "index.md",
                  "Modeling language" => "modeling_lang.md",
                  "Examples" => "examples.md",
                  "Concepts" => "concepts.md",
                  "Architecture" => "architecture.md",
                  "Library" => ["Traces, choices, and call sites" => "library_api/sites.md",
                                "Execution contexts" => "library_api/contexts.md",
                                "Selection interface" => "library_api/selection_interface.md",
                                "Inference" => ["Importance sampling" => "inference/is.md",
                                                "Metropolis-Hastings" => "inference/mh.md",
                                                "Particle filtering" => "inference/pf.md",
                                                "Automatic differentiation variational inference" => "inference/vi.md"],
                                "Foreign model interface" => "library_api/fmi.md",
                                "Differentiable programming" => "library_api/diff_prog.md",
                               ],
                  #"Tutorials" => [
                  #               "Bayesian linear regression" => "bayeslinreg.md",
                  #               "Autoencoding with black box extensions" => "autoencoding.md",
                  #               "Gaussian process kernel synthesis" => "gp_kernel_synth.md",
                  #               "Inference compilation" => "infcomp.md"],
                  "Benchmarks" => "benchmarks/index.md",
                  "Related work" => "related_work.md"
                 ],
         format = Documenter.HTML(prettyurls = true,
                                  assets = ["assets/favicon.ico"])
        )

deploydocs(repo = "github.com/femtomc/Jaynes.jl.git")
