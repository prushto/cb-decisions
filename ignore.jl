using Pkg; Pkg.add("POMDPs"); Pkg.add("QMDP"); Pkg.add("QuickPOMDPs"); Pkg.add("POMDPModelTools"); Pkg.add("POMDPSimulators"); Pkg.add("DiscreteValueIteration")

using POMDPs, QuickPOMDPs, POMDPModelTools, POMDPSimulators, QMDP, DiscreteValueIteration

mountaincar = QuickMDP(
    function (s, a, rng)        
        x, v = s
        vp = clamp(v + a*0.001 + cos(3*x)*-0.0025, -0.07, 0.07)
        xp = x + vp
        if xp > 0.5
            r = 100.0
        else
            r = -1.0
        end
        return (sp=(xp, vp), r=r)
    end,
    actions = [-1., 0., 1.],
    initialstate = (-0.5, 0.0),
    discount = 0.95,
    isterminal = s -> s[1] > 0.5
)

solver = ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true) # creates the solver
policy = solve(solver, mountaincar) # runs value iterations