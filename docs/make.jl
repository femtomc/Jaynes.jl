using Documenter
using Jaynes

makedocs(sitename = "Jaynes.jl",
         pages = [#"Architecture" => "architecture.md",
                  "Architecture" => "index.md",
                  "Modeling language" => "modeling_lang.md",
                  "Examples" => "examples.md",
                  #"Concepts" => "concepts.md",
                  #"Trace types" => "trace_types.md",
                  #"Contextual domain-specific languages" => "contextual_DSLs.md",
                  #"Implementation architecture" => "architecture.md",
                  #"Differentiable programming" => "gradients.md",
                  "Library" => ["Execution contexts" => "contexts.md",
                                "Selection interfaces" => "selection_interface.md",
                                "Inference" => "inference.md",
                                "Differentiable programming" => "diff_prog.md",
                               ],
                  #"Examples" => [
                  #               "Bayesian linear regression" => "bayeslinreg.md",
                  #               "Autoencoding with black box extensions" => "autoencoding.md",
                  #               "Gaussian process kernel synthesis" => "gp_kernel_synth.md",
                  #               "Inference compilation" => "infcomp.md"],
                  #"Related work" => "related_work.md"
                 ],
         format = Documenter.HTML(prettyurls = true,
                                  assets = ["assets/favicon.ico"])
        )

deploydocs(repo = "github.com/femtomc/Jaynes.jl.git")
