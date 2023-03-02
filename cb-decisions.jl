using AdaOPS
using Dates
using Distributions
using Plots
using POMCPOW
using POMDPs: Policy, solve
using POMDPModels
using POMDPPolicies: FunctionPolicy, RandomPolicy
using POMDPSimulators
using POMDPTools: Deterministic, Uniform
using QMDP
using QuickPOMDPs: QuickPOMDP

# using Pkg
# Pkg.add(url="https://github.com/WhiffleFish/ParticleFilterTrees.jl")
using ParticleFilterTrees


function stepthrough_and_plot(cbsimulator, base_policy, output_filename)
    chart_data = []
    labelVec = ["i" "π" "π_e" "r" "u"]

    i_history = Vector{Float16}()
    π_history = Vector{Float16}()
    π_e_history = Vector{Float16}()
    r_history = Vector{Float16}()
    u_history = Vector{Float16}()
    reward_sum = 0
    for (s, a, o, r) in stepthrough(cbsimulator, base_policy, "s,a,o,r", max_steps=100)
        println("State was $s,")
        println("action $a was taken,")
        println("received observation $o and reward $r")
        push!(i_history, s.i * 100)
        push!(π_history, s.π * 100)
        push!(π_e_history, s.π_e * 100)
        push!(r_history, s.r * 100)
        push!(u_history, s.u * 100)
        reward_sum += r
    end

    x_axis = collect(1:length(i_history))

    # need the line below to record and display multiple charts - see notebook
    # all_chart_data = Plots.Plot{Plots.GRBackend}[]

    # Output chart
    push!(chart_data, i_history)
    push!(chart_data, π_history)
    push!(chart_data, π_e_history)
    push!(chart_data, r_history)
    push!(chart_data, u_history)
    y = chart_data
    plot(x_axis, y, label=labelVec)
    println("Total rewards: " * string(reward_sum))
    savefig(output_filename)
end

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

function observation(a, sp)
    y::Float16 = sp.y
    i::Float16 = sp.i
    π::Float16 = sp.π
    π_e::Float16 = sp.π_e
    r::Float16 = sp.r
    u::Float16 = sp.u

    dy = EconomyState(y+0.04, i, π, π_e, r, u)
    dy2 = EconomyState(y-0.04, i, π, π_e, r, u)
    dp = EconomyState(y, i, π+0.01, π_e, r, u)
    dp2 = EconomyState(y, i, π-0.01, π_e, r, u)
    dpe = EconomyState(y, i, π, π_e+0.01, r, u)
    dpe2 = EconomyState(y, i, π, π_e-0.01, r, u)
    dr = EconomyState(y, i, π, π_e, r+0.01, u)
    dr2 = EconomyState(y, i, π, π_e, r-0.01, u)
    du = EconomyState(y, i, π, π_e, r, u+0.01)
    du2 = EconomyState(y, i, π, π_e, r, u-0.01)
    return Uniform([sp dy dy2 dp dp2 dpe dpe2 dr dr2 du du2])
end

function reward(π)
    # return (π == 0.02 ? 100.0 : -1.0))
    # return -100 * abs(π - 0.02)
    return -100 * ((π - 0.02)^2)
end

cbsimulator = QuickPOMDP(
    actions = collect(Float32, -0.05:0.0005:0.05),
    discount = 0.5,

    gen = function (c::EconomyState, a, rng)
        y::Float16 = c.y
        i::Float16 = c.i
        π::Float16 = c.π
        π_e::Float16 = c.π_e
        r::Float16 = c.r
        u::Float16 = c.u
        i = i + a # new line: updates interest rate to reflect CB action
        dt = 0.1 # some arbitary measure of time elapsing
        ϵ = rand() * 0.003 
        flip = rand() < 0.5 ? -1 : 1
        η = rand() < 0.25 ? rand() * 0.015 * flip : 0
        flip = rand() < 0.5 ? -1 : 1
        demandShock = rand() < 0.04 ? rand() * 0.02 * flip : 0 # 4% chance of demand shock, either positive or negative
        r = i - π_e
        y_dot = - (r - r_bar) / σy + η + demandShock # change to rate of output
        # println(y_dot)
        y += y_dot * dt
        y = min(y, y_bar + (u_bar * σu)) # change in y is either determined by demand shock or change in unemployment, whichever is more negative
        π = π_e + (y - y_bar) / σπ
        π_e_dot = (π - π_e) / σπ_e + ϵ
        π_e += dt * π_e_dot
        u_dot = -y_dot / σu
        u += u_dot * dt
        u = max(u, 0) # unemployment rate can't be less than 0
        
        return (
            sp=EconomyState(y::Float16,
                i::Float16,
                π::Float16,
                π_e::Float16,
                r::Float16,
                u::Float16),
            r=reward(π)
            )
    end,
    observation = observation,
    obstype = EconomyState,
    initialstate = Deterministic(initialEconomyState),
    isterminal = (s::EconomyState) -> (s.i > 0.5),
)

solver = AdaOPSSolver()
# solver = POMCPOWSolver(criterion=MaxUCB(1.0))
# solver = PFTDPWSolver() #From Anka: Try this solver, didn't work :(

policy = solve(solver, cbsimulator)
base_policy = FunctionPolicy((pol) -> 0)
random_policy = RandomPolicy(cbsimulator)

timestamp = Dates.format(now(), "yyyy-mm-ddTHHMMSS")
stepthrough_and_plot(cbsimulator, policy, "output/output" * timestamp * ".png")
stepthrough_and_plot(cbsimulator, base_policy, "output/output_zero" * timestamp * ".png")
stepthrough_and_plot(cbsimulator, random_policy, "output/output_rand" * timestamp * ".png")
