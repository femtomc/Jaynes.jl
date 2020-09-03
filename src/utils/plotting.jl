function lineplot(v::Float64, title, name, xlabel, ylabel)
    plt = UnicodePlots.lineplot([i for i in 1 : length(v)], 
                                v, 
                                title = title, 
                                name = name, 
                                xlabel = xlabel, ylabel = ylabel)
    plt
end
