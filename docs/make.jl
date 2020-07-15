using Documenter
using Jaynes

makedocs(sitename = "Jaynes.jl",
         pages = ["Introduction" => "index.md",
                  "Concepts" => "concepts.md",
                  "Modeling language" => "modeling_lang.md",
                  "Examples" => "examples.md",
                  "Architecture" => "architecture.md",
                  "Library" => ["Traces, choices, and call sites" => "sites.md",
                                "Execution contexts" => "contexts.md",
                                "Selection interface" => "selection_interface.md",
                                "Inference" => ["inference/index.md",
                                                "Importance sampling" => "inference/is.md",
                                                "Metropolis-Hastings" => "inference/mh.md",
                                                "Particle filtering" => "inference/pf.md",
                                                "Automatic differentiation variational inference" => "inference/vi.md"],
                                "Foreign model interface" => "fmi.md",
                                "Differentiable programming" => "diff_prog.md",
                               ],
                  #"Tutorials" => [
                  #               "Bayesian linear regression" => "bayeslinreg.md",
                  #               "Autoencoding with black box extensions" => "autoencoding.md",
                  #               "Gaussian process kernel synthesis" => "gp_kernel_synth.md",
                  #               "Inference compilation" => "infcomp.md"],
                  "Related work" => "related_work.md"
                 ],
         format = Documenter.HTML(prettyurls = true,
                                  assets = ["assets/favicon.ico"])
        )

deploydocs(repo = "github.com/femtomc/Jaynes.jl.git")
