using Documenter

makedocs(sitename = "Jaynes",
         pages = [
                  "Introduction" => "index.md",
                  "Concepts" => "concepts.md",
                  "Implementation architecture" => "architecture.md"],
         format = Documenter.HTML(prettyurls = false)
        )
