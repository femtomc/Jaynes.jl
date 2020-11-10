This directory contains a set of staged execution contexts which interpret function calls in their own unique way :)

Dangerous? Yes. But we try to be very controlled about this - the execution contexts (when using the `DefaultPipeline`) will only look at addressed `trace` calls. Alternative pipelines can provide specializations, automatic addressing, etc.
