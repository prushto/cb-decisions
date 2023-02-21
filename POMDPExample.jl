# using Pkg
# Pkg.add("Plots")
# Pkg.add("POMDPs")

using Plots
using POMDPs
using QuickPOMDPs
using POMDPModelTools
using QMDP
using AdaOPS
using POMDPSimulators
import POMDPs: solve
# using POMDPSolve
import QuickPOMDPs: QuickPOMDP
using POMDPTools: Deterministic
using POMDPTools: AlphaVectorPolicy
# import Distributions: Normal

struct EconomyState
    y::Float16
    i::Float16
    π::Float16
    π_e::Float16
    r::Float16
    u::Float16
end


y_bar = 1.0 # non-inflationary output level
r_bar = 0.02 # non-inflationary real interest rate
u_bar = 0.045 # non-inflationary unemployment rate

# sigmas (control rates of change)
σπ = 2.15
σπ_e = 1.3
σy = 1.2
σu = 2

# http://juliapomdp.github.io/POMDPs.jl/stable/def_pomdp/

initialEconomyState = EconomyState(
    1.0, # y output level
    0.04, # i nominal interest rate
    0.02, # π inflation
    0.02, # π_e inflation expectation
    0.02, # r real interest rate (CHANGE TO FORMULA?)
    0.045 # u unemployment rate
) # initialize an economyState

cbsimulator = QuickPOMDP(
    actions = [-0.5, 0., 0.5],
    discount = 0.95,

    gen = function (c::EconomyState, a, rng)
        y::Float16 = c.y
        i::Float16 = c.i
        π::Float16 = c.π
        π_e::Float16 = c.π_e
        r::Float16 = c.r
        u::Float16 = c.u
        π = π + a # new line: updates interest rate to reflect CB action
        dt = 0.1 # some arbitary measure of time elapsing
        ϵ = rand() * 0.003 
        flip = rand() < 0.5 ? -1 : 1
        η = rand() < 0.25 ? rand() * 0.015 * flip : 0
        flip = rand() < 0.5 ? -1 : 1
        demandShock = rand() < 0.04 ? rand() * 0.04 * flip : 0
        r = i - π_e
        y_dot = - (r - r_bar) / σy + η + demandShock
        y += y_dot * dt
        y = min(y, y_bar + (u_bar * σu))
        π = π_e + (y - y_bar) / σπ
        π_e_dot = (π - π_e) / σπ_e + ϵ
        π_e += dt * π_e_dot
        #u_dot = -y_dot / σu
        #c.u += u_dot * dt
        #c.u = max(c.u, 0) # unemployment rate can't be less than 0

        
        return (sp=EconomyState(y::Float16,
        i::Float16,
        π::Float16,
        π_e::Float16,
        r::Float16,
        u::Float16), r= (c.π == 0.02 ? 100.0 : -1.0))
    end,

    observation = (a, sp) -> sp,

    # reward = function (s, a, sp)
    #     if sp.i == 0.02
    #         return 100.0
    #     else 
    #         return -1.0
    #     end
    # end,

    initialstate = Deterministic(initialEconomyState),
    isterminal = (s::EconomyState) -> (s.i > 0.5),
)


#solver = QMDPSolver()
#policy = solve(solver, m)

"""
struct GreedyOfflineSolver <: Solver end

struct DictPolicy{S,A} <: Policy
    actions::Dict{S,A}
end

function POMDPs.solve(::GreedyOfflineSolver, m::POMDP)

    alphas = Vector{Float64}[]

    for a in actions(m)
        alpha = zeros(length(states(m)))
        for s in states(m)
            if !isterminal(m, s)
                r = 0.0
                td = transition(m, s, a)
                for sp in support(td)
                    tp = pdf(td, sp)
                    od = observation(m, s, a, sp)
                    for o in support(od)
                        r += tp * pdf(od, o) * reward(m, s, a, sp, o)
                    end
                end
                alpha[stateindex(m, s)] = r
            end
        end
        push!(alphas, alpha)
    end
    
    return AlphaVectorPolicy(m, alphas, collect(actions(m)))
end

# MDP Offline Solver
POMDPs.action(p::DictPolicy, s) = p.actions[s]

function solve(::GreedyOfflineSolver, m::MDP)

    best_actions = Dict{statetype(m), actiontype(m)}()

    for s in states(m)
        if !isterminal(m, s)
            best = -Inf
            for a in actions(m)
                td = transition(m, s, a)
                r = 0.0
                for sp in support(td)
                    r += pdf(td, sp) * reward(m, s, a, sp)
                end
                if r >= best
                    best_actions[s] = a
                    best = r
                end
            end
        end
    end
    
    return DictPolicy(best_actions)
end
"""

# For chart
chart_data = []
labelVec = ["i" "π" "π_e" "r" "u"]


policy = solve(AdaOPSSolver(bounds=IndependentBounds(-20.0, 0.0)), cbsimulator)

s = nothing
for (s, a, o, r) in stepthrough(cbsimulator, policy, "s,a,o,r", max_steps=10)
    println("State was $s,")
    println("action $a was taken,")
    println("received observation $o and reward $r")
end


x_axis = collect(1:length(s.i_history))

# need the line below to record and display multiple charts - see notebook
# all_chart_data = Plots.Plot{Plots.GRBackend}[]

# Output chart
push!(chart_data, s.i_history)
push!(chart_data, s.π_history)
push!(chart_data, s.π_e_history)
push!(chart_data, s.r_history)
push!(chart_data, s.u_history)
plot(x_axis, chart_data, label=labelVec)



# Below is code for multiple charts
# push!(all_chart_data, plot(x_axis, chart_data, label=labelVec))

# plot1 = all_chart_data[1]
# plot(plot1)

