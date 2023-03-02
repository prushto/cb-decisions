# Baseline policy that takes no actions
# Not using this file

include("time_elapse.jl")
using Plots

all_chart_data = Plots.Plot{Plots.GRBackend}[]
dt_max = 100
x_axis = collect(1:dt_max)
numSimulations = 12
labelVec = ["i" "π" "π_e" "r" "u"]
i_history = Vector{Float16}()
π_history = Vector{Float16}()
π_e_history = Vector{Float16}()
r_history = Vector{Float16}()
u_history = Vector{Float16}()
labelVec = ["i" "π" "π_e" "r" "u"]
for x = 1:numSimulations
    chart_data = []
    i_history = Vector{Float16}()
    π_history = Vector{Float16}()
    π_e_history = Vector{Float16}()
    r_history = Vector{Float16}()
    u_history = Vector{Float16}()
    sim = economyState()
    for y = 1:dt_max
        time_Elapse(sim, 0, rng)
        push!(i_history, s.i * 100)
        push!(π_history, s.π * 100)
        push!(π_e_history, s.π_e * 100)
        push!(r_history, s.r * 100)
        push!(u_history, s.u * 100)
    end
    push!(chart_data, sim.i_history)
    push!(chart_data, sim.π_history)
    push!(chart_data, sim.π_e_history)
    push!(chart_data, sim.r_history)
    push!(chart_data, sim.u_history)
    push!(all_chart_data, plot(x_axis, chart_data, label=labelVec))
end
plot(all_chart_data..., layout=(4, 3)) # note the splat