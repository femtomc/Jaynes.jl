using Documenter
using Jaynes

makedocs(sitename = "Jaynes.jl",
         pages = ["Architecture" => "index.md",
                  "Modeling language" => "modeling_lang.md",
                  "Examples" => "examples.md",
                  #"Concepts" => "concepts.md",
                  #"Trace types" => "trace_types.md",
                  "Library" => ["Execution contexts" => "contexts.md",
                                "Selection interfaces" => "selection_interface.md",
                                "Inference" => "inference.md",
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
