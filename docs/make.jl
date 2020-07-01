using Documenter

makedocs(sitename = "Jaynes.jl",
         pages = [#"Architecture" => "architecture.md",
                  "Architecture" => "index.md",
                  #"Concepts" => "concepts.md",
                  #"Trace types" => "trace_types.md",
                  #"Contextual domain-specific languages" => "contextual_DSLs.md",
                  #"Implementation architecture" => "architecture.md",
                  #"Differentiable programming" => "gradients.md",
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
