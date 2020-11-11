# This is a generic glue function.
@inline unwrap(gr) = gr
@inline unwrap(gr::GlobalRef) = gr.name
@inline unwrap(v::Val{K}) where K = K
@inline unwrap(d::Diffed) = d.value
@inline unwrap(::Type{K}) where K = K
@inline unwrap(::Const{K}) where K = K
@inline unwrap(::Partial{K}) where K = K
@inline unwrap(::Mjolnir.Node{K}) where K = K
@inline unwrap(sym::QuoteNode) = sym.value
