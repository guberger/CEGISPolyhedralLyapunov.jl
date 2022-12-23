module ExamplePerformance_Plot

using PyPlot

str = readlines(string(@__DIR__, "/measurements.txt"))
N_list = Int[]
γ_list = Float64[]
M_list = Int[]
TIMEs_list_list = Vector{Vector{Tuple{Float64,Float64}}}[]
NITERs_list_list = Vector{Vector{Tuple{Float64,Float64}}}[]

sec = ""

for ln in str
    if length(ln) ≥ 4 && ln[1:4] == "sec:"
        global sec = ln[6:end]
        continue
    end
    if sec == "N"
        push!(N_list, parse(Int, ln))
        continue
    end
    if sec == "γ"
        push!(γ_list, parse(Float64, ln))
        continue
    end
    if sec == "M"
        push!(M_list, parse(Int, ln))
        continue
    end
    local ITEMs_list = Vector{Tuple{Float64,Float64}}[]
    ln = replace(ln, "[[("=>"[(")
    ln = replace(ln, ")]]"=>")]")
    ln = replace(ln, ")], "=>")];;; ")
    local words1 = split(ln, ";;; ")
    @assert length(words1) == length(γ_list)
    for word2 in words1
        local ITEMs = Tuple{Float64,Float64}[]
        word2 = replace(word2, "[("=>"(")
        word2 = replace(word2, ")]"=>")")
        word2 = replace(word2, "), "=>");;; ")
        local words2 = split(word2, ";;; ")
        @assert length(words2) == length(M_list)
        for word_tuple in words2
            word_tuple = replace(word_tuple, r"[\(\)]"=>"")
            local ws = Tuple{String,String}(split(word_tuple, ", "))
            push!(ITEMs, map(w -> parse.(Float64, w), ws))
        end
        push!(ITEMs_list, ITEMs)
    end
    if sec == "TIME"
        push!(TIMEs_list_list, ITEMs_list)
        continue
    end
    if sec == "NITER"
        push!(NITERs_list_list, ITEMs_list)
        continue
    end
end

fig = figure(0, figsize=(12, 9))
nγ = length(γ_list)
ax_ = fig.subplots(nγ, 2, gridspec_kw=Dict(("hspace"=>0.2)))
axTIME_ = ax_[1:nγ]
axNITER_ = ax_[nγ+1:2*nγ]
for (i, ax) in enumerate(axTIME_)
    ax.set_ylabel(
        "\$\\gamma=$(γ_list[i])\$", rotation=0, fontsize=12, labelpad=15,
        horizontalalignment="right"
    )
end
for (i, ax) in Iterators.flatten(enumerate(axs) for axs in (axTIME_, axNITER_))
    ax.set_xticks(N_list)
end
ax_[1].set_title("Computation time (s)", pad=14)
ax_[nγ + 1].set_title("# iterations", pad=14)
ax_[nγ].set_xlabel(L"dimension $d$")
ax_[2*nγ].set_xlabel(L"dimension $d$")

for (i, γ) in enumerate(γ_list)
    local TIMEs_list = getindex.(TIMEs_list_list, i)
    local NITERs_list = getindex.(NITERs_list_list, i)
    for (j, M) in enumerate(M_list)
        local TIMEs = getindex.(TIMEs_list, j)
        local NITERs = getindex.(NITERs_list, j)
        axTIME_[i].errorbar(
            N_list, getindex.(TIMEs, 1), yerr=getindex.(TIMEs, 2),
            ls="solid", marker="^", ms=10, label="\$m=$(M_list[j])\$"
        )
        axNITER_[i].errorbar(
            N_list, getindex.(NITERs, 1), yerr=getindex.(NITERs, 2),
            ls="solid", marker="^", ms=10, label="\$m=$(M_list[j])\$"
        )
    end
end

ax_[1].legend(fontsize=15)

fig.savefig(string(
        @__DIR__, "/../figures/fig_performance.png"
    ), dpi=200, transparent=false, bbox_inches="tight")

end # module