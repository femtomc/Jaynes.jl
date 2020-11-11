The `gfi` directory contains a set of staged execution contexts which interpret function calls in their own unique way :)

Dangerous? Yes. But we try to be very controlled about this - the execution contexts (when using the `DefaultPipeline`) will only look at addressed `trace` calls. Alternative pipelines can provide specializations, automatic addressing, etc.

The `pipelines.jl` file contains the method definitions for the default pipelines. Users of Jaynes can implement their own staging pipelines by creating their own `CompilationOptions` inheritor and then specifying their own `pipeline` function for each of the execution contexts.
