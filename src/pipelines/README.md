The `gfi` directory contains a set of staged execution contexts which interpret function calls in their own unique way :)

Dangerous? Yes. But we try to be very controlled about this - the execution contexts (when using the `DefaultPipeline`) will only look at addressed `trace` calls. Alternative pipelines can provide specializations, automatic addressing, etc.

The `default_pipeline.jl` file contains the method definitions for the default pipeline. Users of Jaynes can implement their own staging pipelines by creating their own `CompilationOptions` inheritor and then specifying their own `instantiation_pipeline` and `pipeline` functions. Note that `pipeline` needs to be implemented for each of the execution contexts.
