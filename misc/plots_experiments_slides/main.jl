using PyPlot

d_list = [4, 5, 6, 7, 8, 9]
γ_list = [1, 0.1, 0.01]
Tfeas_list = [
    [0.37, 0.84, 4.48],
    [0.47, 2.82, 11.54],
    [1.31, 11.67, 12.32],
    [2.22, 26.42, 34.09],
    [20.20, 47.27, 160.18],
    [13.85, 92.77, 206.85]
]
Mfeas_list = [
    [12, 21, 37],
    [14, 30, 53],
    [20, 46, 52],
    [23, 61, 68],
    [50, 87, 133],
    [45, 98, 149]
]
Tinfeas_list = [
    [0.01, 0.23, 1.17],
    [0.01, 0.33, 2.47],
    [0.01, 1.07, 4.58],
    [0.01, 2.16, 11.94],
    [0.01, 5.65, 7.95],
    [0.01, 10.22, 24.61]
]
Minfeas_list = [
    [2, 13, 25],
    [2, 14, 31],
    [2, 22, 40],
    [2, 25, 41],
    [2, 35, 46],
    [2, 42, 54]
]

fig = figure(figsize=(12, 7))
axMf = fig.add_subplot(2, 2, 3)
axTf = fig.add_subplot(2, 2, 1)
axMi = fig.add_subplot(2, 2, 4)
axTi = fig.add_subplot(2, 2, 2)

axTf.set_title("Lyapunov function found", pad=20, fontsize=20)
axTi.set_title(L"$\delta$-Lyapunov function infeasible", pad=20, fontsize=20)

for (i, γ) in enumerate(γ_list)
    axTf.plot(
        d_list, getindex.(Tfeas_list, i),
        ls="solid", marker=".", ms=15, label=string("γ = ", γ)
    )
    axMf.plot(
        d_list, getindex.(Mfeas_list, i),
        ls="solid", marker=".", ms=15, label=string("γ = ", γ)
    )
    axTi.plot(
        d_list, getindex.(Tinfeas_list, i),
        ls="solid", marker=".", ms=15, label=string("γ = ", γ)
    )
    axMi.plot(
        d_list, getindex.(Minfeas_list, i),
        ls="solid", marker=".", ms=15, label=string("γ = ", γ)
    )
end

axTf.set_ylabel("Computation time (s)", fontsize=15)
axMf.set_ylabel("# iterations", fontsize=15)
axMf.set_xlabel(L"dimension $d$", fontsize=15)
axMi.set_xlabel(L"dimension $d$", fontsize=15)

axTi.legend(bbox_to_anchor=(1.1, 1.05), fontsize=15)

fig.savefig("experiments.png", bbox_inches="tight")