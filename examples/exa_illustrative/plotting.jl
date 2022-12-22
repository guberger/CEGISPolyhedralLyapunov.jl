function plot_field!(ax, piece, xlims, ylims, ngrid; c="gray")
    x1_grid = range(xlims..., length=ngrid)
    x2_grid = range(ylims..., length=ngrid)
    X = map(x -> [x...], Iterators.product(x1_grid, x2_grid))
    X1 = getindex.(X, 1)
    X2 = getindex.(X, 2)
    D = fill([NaN, NaN], ngrid, ngrid)
    for (k, x) in enumerate(X)
        any(lf -> dot(x, lf) > 0, piece.lfs_dom) && continue
        D[k] = piece.A*x
    end
    D1 = getindex.(D, 1)
    D2 = getindex.(D, 2)
    ax.quiver(X1, X2, D1, D2, color=c)
end

function plot_wits!(
        ax, wit_cls, lfs, β;
        mcx="black", msx=15, mcy="blue", msy=15, lc="green", lw=2.5
    )
    for wit_cl in wit_cls
        for wit in wit_cl
            x_norm = maximum(lf -> dot(lf, wit.x), lfs)
            γ = β/x_norm
            x = wit.x*γ
            y = wit.y*γ
            # --- Print for Tikz ---
            # println((x[1], x[2]))
            # println((y[1], y[2]))
            # println((x[1], x[2]), " ", (y[1], y[2]))
            # ---
            ax.plot((x[1], y[1]), (x[2], y[2]), c=lc, lw=lw)
            ax.plot(x..., marker=".", ms=msx, c=mcx)
            ax.plot(y..., marker=".", ms=msy, c=mcy)
        end
    end
end