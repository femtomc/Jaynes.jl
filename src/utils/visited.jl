# Regenerate and update.
function visited_walk!(discard, d_score::Float64, rs::ChoiceSite, addr)
    push!(discard, addr, rs.val)
    d_score += rs.score
end

function visited_walk!(par_addr, discard, d_score::Float64, rs::ChoiceSite, addr)
    push!(discard, par_addr => addr, rs.val)
    d_score += rs.score
end

function visited_walk!(par_addr, discard, d_score::Float64, tr::T, vs::VisitedSelection) where T <: Trace
    for addr in keys(tr.chm)
        # Check choices at this stack level.
        if !(addr in vs.addrs)
            visited_walk!(par_addr, discard, d_score, tr.chm[addr], addr)
            delete!(tr.chm, addr)
        
        # If it's a CallSite, recurse into it.
        elseif addr in keys(vs.tree)
            visited_walk!(par_addr => addr, discard, d_score, tr.chm[addr].trace, vs.tree[addr])
        end
    end
end

# Toplevel.
function visited_walk!(tr::T, vs::VisitedSelection) where T <: Trace
    discard = ConstrainedHierarchicalSelection()
    d_score = 0.0
    for addr in keys(tr.chm)
        # Check choices at this stack level.
        if !(addr in vs.addrs)
            visited_walk!(discard, d_score, tr.chm[addr], addr)
            delete!(tr.chm, addr)
       
        # If it's a CallSite, recurse into it.
        elseif addr in keys(vs.tree)
            visited_walk!(addr, discard, d_score, tr.chm[addr].trace, vs.tree[addr])
        end
    end
    tr.score -= d_score
    return discard
end
