using Plots
using POMCPOW
using POMDPModelTools
using POMDPSimulators
using POMDPs: solve
using POMDPTools: Deterministic
using QMDP
using QuickPOMDPs: QuickPOMDP
using Distributions

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

function reward(π)
    return -100 * abs(π - 0.02)
end


cbsimulator = QuickPOMDP(
    actions = [-0.0075, -0.005, -0.0025, 0., 0.0025, 0.005, 0.0075],
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
        demandShock = rand() < 0.04 ? rand() * 0.04 * flip : 0
        r = i - π_e
        y_dot = - (r - r_bar) / σy + η + demandShock
        y += y_dot * dt
        y = min(y, y_bar + (u_bar * σu))
        π = π_e + (y - y_bar) / σπ
        π_e_dot = (π - π_e) / σπ_e + ϵ
        π_e += dt * π_e_dot
        u_dot = -y_dot / σu
        u += u_dot * dt
        u = max(u, 0) # unemployment rate can't be less than 0

        
        return (sp=EconomyState(y::Float16,
        i::Float16,
        π::Float16,
        π_e::Float16,
        r::Float16,
        u::Float16),
        # r= (c.π == 0.02 ? 100.0 : -1.0))
        r = reward(π))
    end,

    observation = (a, sp) -> Deterministic(sp),
    obstype = EconomyState,
    initialstate = Deterministic(initialEconomyState),
    isterminal = (s::EconomyState) -> (s.i > 0.5),
)


#solver = QMDPSolver()
solver = POMCPOWSolver(criterion=MaxUCB(20.0))

policy = solve(solver, cbsimulator)

# For chart
chart_data = []
labelVec = ["i" "π" "π_e" "r" "u"]



i_history = Vector{Float16}()
π_history = Vector{Float16}()
π_e_history = Vector{Float16}()
r_history = Vector{Float16}()
u_history = Vector{Float16}()
for (s, a, o, r) in stepthrough(cbsimulator, policy, "s,a,o,r", max_steps=100)
    println("State was $s,")
    println("action $a was taken,")
    println("received observation $o and reward $r")
    push!(i_history, s.i * 100)
    push!(π_history, s.π * 100)
    push!(π_e_history, s.π_e * 100)
    push!(r_history, s.r * 100)
    push!(u_history, s.u * 100)
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

savefig("output.png")

# Below is code for multiple charts
# push!(all_chart_data, plot(x_axis, chart_data, label=labelVec))

# plot1 = all_chart_data[1]
# plot(plot1)

