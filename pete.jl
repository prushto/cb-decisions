using Pkg
Pkg.add("Plots")
using Plots
using POMDPs, POMDPModelTools, QuickPOMDPs

mutable struct economyState
    # To store states
    y_history::Vector
    i_history::Vector
    π_history::Vector
    π_e_history::Vector
    r_history::Vector
    u_history::Vector
    y::Float16
    i::Float16
    π::Float16
    π_e::Float16
    r::Float16
    u::Float16

    function economyState(dummy)
        # Starting variables
        new(
            [], # history empty
            [], # history empty
            [], # history empty
            [], # history empty
            [], # history empty
            [], # history empty
            1.0, # y output level
            0.04, # i nominal interest rate
            0.02, # π inflation
            0.02, # π_e inflation expectation
            0.02, # r real interest rate (CHANGE TO FORMULA?)
            0.045 # u unemployment rate
        )
    end
end



