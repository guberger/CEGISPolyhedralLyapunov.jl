function plot_traj!(ax_, sys, x0, loc0, dt, nstep; c="purple", ms=15, lw=2.5)
    x, loc = x0, loc0
    if nstep == 0
        ax_[loc].plot(x_seq[1]..., marker=".", ms=ms, c=c)
        return nothing
    end
    x_seq = [x]
    iter = 0
    while iter < nstep
        if iter == nstep
            ax_[loc].plot(x_seq[1]..., marker=".", ms=ms, c=c)
            ax_[loc].plot(getindex.(x_seq, 1), getindex.(x_seq, 2), lw=lw, c=c)
        end
        x_next = x
        loc_next = loc
        next = false
        for piece in sys.disc_pieces
            next && break
            if loc == piece.loc1 && x ∈ piece.domain
                x_next = piece.A*x
                loc_next = piece.loc2
                next = true
            end
        end
        for piece in sys.cont_pieces
            next && break
            if loc == piece.loc && x ∈ piece.domain
                x_next = exp(piece.A*dt)*x
                loc_next = loc
                next = true
            end
        end
        iter += 1
        if loc_next != loc
            ax_[loc].plot(x_seq[1]..., marker=".", ms=ms, c=c)
            ax_[loc].plot(getindex.(x_seq, 1), getindex.(x_seq, 2), lw=lw, c=c)
            ax_[loc].plot((x[1], x_next[1]), (x[2], x_next[2]), ls="--", lw=lw, c=c)
            empty!(x_seq)
        elseif iter == nstep
            push!(x_seq, x_next)
            ax_[loc].plot(x_seq[1]..., marker=".", ms=ms, c=c)
            ax_[loc].plot(getindex.(x_seq, 1), getindex.(x_seq, 2), lw=lw, c=c)
        end
        x, loc = x_next, loc_next
        push!(x_seq, x)
    end
    return nothing
end