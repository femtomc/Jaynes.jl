# 02/13/2020

IRTracker provides a dynamic representation of the program (the equivalent of a Wengert list with richer metadata and tracking at all levels of the IR). Inside this representation, I think it is possible to extract the trace of the randomness in the program. However, it's heavyweight compared to Gen's lightweight tracing because you get everything else as well.

Instead, I think a better approach is to use IRTools to insert statements in the IR to automatically track randomness. The advantage here is that this can be done statically, so it's much faster than tracing everything first and then extracting what you need.
