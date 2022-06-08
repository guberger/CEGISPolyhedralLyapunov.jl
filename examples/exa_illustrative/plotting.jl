function plot_field_cont!(ax, sys, xlims, ylims, ngrid; c="gray")
    x1_grid = range(xlims..., length=ngrid)
    x2_grid = range(ylims..., length=ngrid)
    X = map(x -> [x...], Iterators.product(x1_grid, x2_grid))
    X1 = getindex.(X, 1)
    X2 = getindex.(X, 2)

    for piece in sys.cont_pieces
        D = Matrix{Vector{Float64}}(undef, ngrid, ngrid)
        for (k, x) in enumerate(X)
            if x ∈ piece.domain
                D[k] = piece.A*x
            else
                D[k] = [NaN, NaN]
            end
        end
        D1 = getindex.(D, 1)
        D2 = getindex.(D, 2)
        ax.quiver(X1, X2, D1, D2, color=c)
    end
end

function plot_level!(ax, mpf, radmax; fc="gold", fa=0.5, ec="gold", ew=2.0)
    p = Polyhedron()
    for lf in mpf.pfs[1].lfs
        CPV.add_halfspace!(p, lf.lin, -1)
    end
    verts = compute_vertices_2d(p, zeros(2))
    verts_radius = maximum(vert -> norm(vert, Inf), verts)
    scaling = radmax/verts_radius
    verts_scaled = map(vert -> vert*scaling, verts)
    polylist = matplotlib.collections.PolyCollection([verts_scaled])
    fca = matplotlib.colors.colorConverter.to_rgba(fc, alpha=fa)
    polylist.set_facecolor(fca)
    polylist.set_edgecolor(ec)
    polylist.set_linewidth(ew)
    ax.add_collection(polylist)
    return scaling
end

_norm(lfs, point) = maximum(lf -> CPLA._eval(lf, point), lfs)

function plot_witnesses!(
        ax, witnesses, lfs, level, deriv_length;
        mc="blue", ms=15, lc="green", lw=2.5
    )
    derivs_norm = -Inf
    for wit in witnesses
        for lieevid in wit.lie_evids
            point_norm = _norm(lfs, lieevid.point)
            derivs_norm = max(derivs_norm, norm(lieevid.deriv)/point_norm)
        end
    end
    for wit in witnesses
        for posevid in wit.pos_evids
            point_norm = _norm(lfs, posevid.point)
            point_scaled = posevid.point*level/point_norm
            ax.plot(point_scaled..., marker=".", ms=ms, c=mc)
        end
        for lieevid in wit.lie_evids
            point_norm = _norm(lfs, lieevid.point)
            point_scaled = lieevid.point*level/point_norm
            ax.plot(point_scaled..., marker=".", ms=ms, c=mc)
            deriv_scaled = lieevid.deriv/(point_norm*derivs_norm)
            point2_scaled = point_scaled + deriv_length*deriv_scaled
            ax.plot(
                (point_scaled[1], point2_scaled[1]),
                (point_scaled[2], point2_scaled[2]), c=lc, lw=lw
            )
        end
    end
end

function plot_traj!(ax, sys, x0, dt, nstep; c="purple", ms=15, lw=2.5)
    ax.plot(x0..., marker=".", ms=ms, c=c)
    x_seq = Vector{Vector{Float64}}(undef, nstep)
    x_seq[1] = x0
    for i = 2:nstep
        x = x_seq[i - 1]
        q = 0
        for piece in sys.pieces
            if x ∈ piece.domain
                x = exp(piece.A*dt)*x
                break
            end
        end
        x_seq[i] = x
    end
    ax.plot(getindex.(x_seq, 1), getindex.(x_seq, 2), lw=lw, c=c)
end