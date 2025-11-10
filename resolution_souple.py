import numpy as np
import matplotlib.pyplot as plt


def loi_lambert(Pt, Dmax, alpha0=0.2, P1=500.0, P2=500.0, n1=2.0, n2=2.0):
    """Loi de Lambert pour la déformation du tube (retourne le diamètre)."""
    Pt = np.asarray(Pt, dtype=float)
    D = np.zeros_like(Pt)
    mask_left = Pt <= 0.0
    mask_right = ~mask_left
    # Formulation piecewise (Lambert 1982)
    if np.any(mask_left):
        D[mask_left] = Dmax * np.sqrt(alpha0 * np.power(1.0 - Pt[mask_left] / P1, -n1))
    if np.any(mask_right):
        D[mask_right] = Dmax * np.sqrt(1.0 - (1.0 - alpha0) * np.power(1.0 - Pt[mask_right] / P2, -n2))
    return D


def _ny_from_D(D, Dmax, N):
    """Calcule le nombre de cellules verticales à partir du diamètre.
    Grille centrée: -Dmax à +Dmax, hy = 2*Dmax/N"""
    hy = 2.0 * float(Dmax) / float(N)
    D_array = np.atleast_1d(np.asarray(D, dtype=float))
    Ny = np.floor(D_array / hy).astype(int)
    Ny = np.clip(Ny, 1, int(N))
    return Ny


def plot_initial_final_grid(Lx, Dmax, Nx, N, j_start_init, j_end_init, 
                            j_start_final, j_end_final, y_bottom_init, y_top_init,
                            y_bottom_final, y_top_final,
                            grid_color='0.85', init_color='tab:blue', final_color='tab:red'):
    """Visualise les grilles initiale et finale (centrées et symétriques)."""
    hx = float(Lx) / int(Nx)
    hy = 2.0 * float(Dmax) / int(N)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Grille globale
    for i in range(int(Nx) + 1):
        x = i * hx
        ax.plot([x, x], [-Dmax, Dmax], color=grid_color, linewidth=0.8)
    for j in range(int(N) + 1):
        y = -Dmax + j * hy
        ax.plot([0, Lx], [y, y], color=grid_color, linewidth=0.8)

    # Axe de symétrie
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--', alpha=0.5, label='Axe de symétrie')

    # Dessin du tube initial (bleu) et final (rouge)
    for i in range(int(Nx)):
        # Initial
        js0 = j_start_init[i]
        je0 = j_end_init[i]
        for j in range(js0, je0):
            y_cell_bottom = -Dmax + j * hy
            ax.add_patch(
                plt.Rectangle((i * hx, y_cell_bottom), hx, hy, 
                            fill=False, edgecolor=init_color, linewidth=0.8, alpha=0.7)
            )
        
        # Final
        js1 = j_start_final[i]
        je1 = j_end_final[i]
        for j in range(js1, je1):
            y_cell_bottom = -Dmax + j * hy
            ax.add_patch(
                plt.Rectangle((i * hx, y_cell_bottom), hx, hy, 
                            fill=False, edgecolor=final_color, linewidth=1.0)
            )
        
        # Lignes de contour
        ax.plot([i * hx, (i + 1) * hx], [y_bottom_init[i], y_bottom_init[i]], 
                color=init_color, linewidth=1.6, alpha=0.7,
                label='Initial' if i == 0 else None)
        ax.plot([i * hx, (i + 1) * hx], [y_top_init[i], y_top_init[i]], 
                color=init_color, linewidth=1.6, alpha=0.7)
        
        ax.plot([i * hx, (i + 1) * hx], [y_bottom_final[i], y_bottom_final[i]], 
                color=final_color, linewidth=1.8,
                label='Final' if i == 0 else None)
        ax.plot([i * hx, (i + 1) * hx], [y_top_final[i], y_top_final[i]], 
                color=final_color, linewidth=1.8)

    ax.set_xlim(0, Lx)
    ax.set_ylim(-Dmax, Dmax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Tube initial (bleu) et final (rouge) - Symétrique par rapport à y=0')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def solve_stokes_sections(Lx, Ly_sections, mu, Pin, Pout, Nx, Dmax, N):
    """
    Résolution de Stokes 2D (schéma MAC) dans un domaine composé de Nx sections.
    Tube centré symétrique autour de y=0.
    
    Entrées:
    - Lx: longueur totale en x
    - Ly_sections: scalaire (diamètre uniforme) ou tableau de taille Nx (diamètre par section)
    - mu: viscosité dynamique
    - Pin, Pout: pression aux entrées (x=0) et sorties (x=Lx)
    - Nx: nombre de sections en x
    - Dmax: demi-hauteur maximale (domaine: -Dmax à +Dmax)
    - N: nombre de divisions verticales pour la grille englobante

    Sorties:
    - p, u, v: champs de pression et vitesse
    - Ny_cols: nombre de cellules par section
    - hx, N: pas et nombre de cellules
    - j_start, j_end: indices de début et fin par section
    - y_bottom, y_top: positions réelles du tube
    """

    hx = float(Lx) / int(Nx)
    N = int(N)
    hy = 2.0 * float(Dmax) / float(N)  # Grille centrée

    # Normaliser Ly_sections (diamètres)
    if np.isscalar(Ly_sections):
        H = np.full(int(Nx), float(Ly_sections), dtype=float)
    else:
        H = np.asarray(Ly_sections, dtype=float).ravel()
        if H.size != int(Nx):
            raise ValueError("Ly_sections doit être scalaire ou de taille Nx.")

    if hy <= 0:
        raise ValueError("hy doit être strictement positif.")

    # Calcul des positions verticales (centrées sur y=0) - SYMÉTRIE FORCÉE
    y_bottom = -H / 2.0
    y_top = H / 2.0
    
    # Indices de grille
    j_start = np.floor((y_bottom + Dmax) / hy).astype(int)
    j_end = np.ceil((y_top + Dmax) / hy).astype(int)
    
    j_start = np.clip(j_start, 0, N)
    j_end = np.clip(j_end, 0, N)
    
    Ny_cols = j_end - j_start

    # Comptage des inconnues
    M = int(Nx) * N
    K = (int(Nx) + 1) * N
    L = int(Nx) * (N + 1)
    ntot = M + K + L

    # Helpers d'indexation
    def idx_p(i, j):
        return i * N + j

    def idx_u(i, j):
        return M + i * N + j

    def idx_v(i, j):
        return M + K + i * (N + 1) + j

    # Marqueurs d'appartenance
    in_p = np.zeros((int(Nx), N), dtype=bool)
    in_u = np.zeros((int(Nx) + 1, N), dtype=bool)
    in_v = np.zeros((int(Nx), N + 1), dtype=bool)

    for i in range(int(Nx)):
        js = j_start[i]
        je = j_end[i]
        in_p[i, js:je] = True
        in_v[i, js:je + 1] = True

    # u sur les faces verticales
    for j in range(N):
        if j >= j_start[0] and j < j_end[0]:
            in_u[0, j] = True
        if j >= j_start[int(Nx) - 1] and j < j_end[int(Nx) - 1]:
            in_u[int(Nx), j] = True
        for i in range(1, int(Nx)):
            js_left = j_start[i - 1]
            je_left = j_end[i - 1]
            js_right = j_start[i]
            je_right = j_end[i]
            if j >= max(js_left, js_right) and j < min(je_left, je_right):
                in_u[i, j] = True

    # Matrices
    A = np.zeros((ntot, ntot), dtype=float)
    B = np.zeros((ntot, 1), dtype=float)

    # 1) Continuité
    for i in range(int(Nx)):
        for j in range(N):
            m = idx_p(i, j)
            if not in_p[i, j]:
                A[m, m] = 1.0
                B[m, 0] = 0.0
                continue

            u_right = idx_u(i + 1, j)
            u_left = idx_u(i, j)
            v_top = idx_v(i, j + 1)
            v_bot = idx_v(i, j)
            A[m, u_right] += 1.0 / hx
            A[m, u_left]  += -1.0 / hx
            A[m, v_top]   += 1.0 / hy
            A[m, v_bot]   += -1.0 / hy

    # 2) Équations pour u
    for i in range(int(Nx) + 1):
        for j in range(N):
            k = idx_u(i, j)

            if not in_u[i, j]:
                A[k, k] = 1.0
                B[k, 0] = 0.0
                continue

            # Déterminer les limites locales
            if i == 0:
                js_local = j_start[0]
                je_local = j_end[0]
            elif i == int(Nx):
                js_local = j_start[int(Nx) - 1]
                je_local = j_end[int(Nx) - 1]
            else:
                js_left = j_start[i - 1]
                je_left = j_end[i - 1]
                js_right = j_start[i]
                je_right = j_end[i]
                js_local = max(js_left, js_right)
                je_local = min(je_left, je_right)

            # Parois horizontales
            if j == js_local or j == je_local - 1:
                A[k, k] = 1.0
                B[k, 0] = 0.0
                continue

            # Frontières verticales: Neumann
            if i == 0:
                A[k, k] = 1.0
                A[k, idx_u(1, j)] = -1.0
                B[k, 0] = 0.0
                continue
            if i == int(Nx):
                A[k, k] = 1.0
                A[k, idx_u(int(Nx) - 1, j)] = -1.0
                B[k, 0] = 0.0
                continue

            # Intérieur
            u_ip1 = idx_u(i + 1, j)
            u_im1 = idx_u(i - 1, j)
            u_jp1 = idx_u(i, j + 1)
            u_jm1 = idx_u(i, j - 1)
            p_r = idx_p(i, j)
            p_l = idx_p(i - 1, j)

            A[k, u_ip1] += mu / (hx ** 2)
            A[k, u_im1] += mu / (hx ** 2)
            A[k, u_jp1] += mu / (hy ** 2)
            A[k, u_jm1] += mu / (hy ** 2)
            A[k, k]     += -2.0 * mu * (1.0 / (hx ** 2) + 1.0 / (hy ** 2))

            A[k, p_r]   += -1.0 / hx
            A[k, p_l]   +=  1.0 / hx

    # 3) Équations pour v
    for i in range(int(Nx)):
        for j in range(N + 1):
            l = idx_v(i, j)

            if not in_v[i, j]:
                A[l, l] = 1.0
                B[l, 0] = 0.0
                continue

            js = j_start[i]
            je = j_end[i]

            # Parois haut/bas
            if j == js or j == je:
                A[l, l] = 1.0
                B[l, 0] = 0.0
                continue

            # Frontières verticales: Neumann
            if i == 0:
                A[l, l] = 1.0
                A[l, idx_v(1, j)] = -1.0
                B[l, 0] = 0.0
                continue
            if i == int(Nx) - 1:
                A[l, l] = 1.0
                A[l, idx_v(int(Nx) - 2, j)] = -1.0
                B[l, 0] = 0.0
                continue

            # Intérieur
            v_ip1 = idx_v(i + 1, j)
            v_im1 = idx_v(i - 1, j)
            v_jp1 = idx_v(i, j + 1)
            v_jm1 = idx_v(i, j - 1)
            p_t = idx_p(i, j)
            p_b = idx_p(i, j - 1)

            A[l, v_im1] += mu / (hx ** 2)
            A[l, v_ip1] += mu / (hx ** 2)
            A[l, v_jp1] += mu / (hy ** 2)
            A[l, v_jm1] += mu / (hy ** 2)
            A[l, l]     += -2.0 * mu * (1.0 / (hx ** 2) + 1.0 / (hy ** 2))

            A[l, p_t]   += -1.0 / hy
            A[l, p_b]   +=  1.0 / hy

    # 4) Conditions de pression
    for i in range(int(Nx)):
        for j in range(N):
            if not in_p[i, j]:
                continue
            m = idx_p(i, j)
            if i == 0:
                A[m, :] = 0.0
                A[m, m] = 1.0
                B[m, 0] = Pin
            elif i == int(Nx) - 1:
                A[m, :] = 0.0
                A[m, m] = 1.0
                B[m, 0] = Pout

    # Résolution
    X = np.linalg.solve(A, B)

    # Rangement
    p = X[0:M].reshape((int(Nx), N))
    u = X[M:M + K].reshape((int(Nx) + 1, N))
    v = X[M + K:].reshape((int(Nx), N + 1))

    return p, u, v, Ny_cols, hx, N, j_start, j_end, y_bottom, y_top


def solve_stokes_souple_sections(Lx, Nx, mu, Pin, Pout, D_init, Dmax, N,
                                 Pext, alpha0=0.2, P1=500.0, P2=500.0, 
                                 n1=2.0, n2=2.0, relax=0.3, tol=1e-4, maxiter=40):
    """
    Résolution itérative de Stokes avec déformation du tube selon la loi de Lambert.
    Tube centré symétrique.
    
    Entrées:
    - Lx, Nx: domaine et discrétisation en x
    - mu: viscosité
    - Pin, Pout: pressions d'entrée/sortie
    - D_init: diamètre initial uniforme
    - Dmax: demi-hauteur maximale (domaine: -Dmax à +Dmax)
    - N: nombre de divisions verticales
    - Pext: pression extérieure pour la déformation
    - alpha0, P1, P2, n1, n2: paramètres de la loi de Lambert
    - relax: facteur de relaxation
    - tol: tolérance de convergence
    - maxiter: nombre maximal d'itérations
    
    Sorties:
    - Dictionnaire avec résultats et historique de convergence
    """
    
    # Initialisation avec diamètre uniforme
    D_init_array = np.full(Nx, D_init, dtype=float)
    D_current = D_init_array.copy()
    
    convergence_history = []
    
    # Configuration initiale pour visualisation
    p_init, u_init, v_init, _, _, _, j_start_init, j_end_init, y_bottom_init, y_top_init = solve_stokes_sections(
        Lx=Lx, Ly_sections=D_init_array, mu=mu, 
        Pin=Pin, Pout=Pout, Nx=Nx, Dmax=Dmax, N=N
    )
    
    for iteration in range(maxiter):
        # Résolution de Stokes
        p, u, v, Ny_cols, hx, _, j_start, j_end, y_bottom, y_top = solve_stokes_sections(
            Lx=Lx, Ly_sections=D_current, mu=mu, 
            Pin=Pin, Pout=Pout, Nx=Nx, Dmax=Dmax, N=N
        )
        
        # Calcul de la pression transmise moyenne par section
        Pt = np.zeros(Nx, dtype=float)
        for i in range(Nx):
            js = j_start[i]
            je = j_end[i]
            if je > js:
                Pt[i] = np.mean(p[i, js:je]) - Pext
        
        # Calcul de la nouvelle géométrie selon Lambert
        D_new = loi_lambert(Pt, 2.0 * Dmax, alpha0, P1, P2, n1, n2)
        
        # Calcul de l'erreur
        error = np.max(np.abs(D_new - D_current))
        convergence_history.append(error)
        
        print(f"Itération {iteration + 1}: erreur = {error:.6e}")
        
        if error < tol:
            print(f"Convergence atteinte en {iteration + 1} itérations")
            break
        
        # Mise à jour avec relaxation
        D_current = relax * D_new + (1.0 - relax) * D_current
    
    else:
        print(f"Nombre maximal d'itérations ({maxiter}) atteint")
    
    return {
        'p': p,
        'u': u,
        'v': v,
        'p_init': p_init,
        'u_init': u_init,
        'v_init': v_init,
        'D_final': D_current,
        'convergence_history': convergence_history,
        'j_start_init': j_start_init,
        'j_end_init': j_end_init,
        'y_bottom_init': y_bottom_init,
        'y_top_init': y_top_init,
        'j_start_final': j_start,
        'j_end_final': j_end,
        'y_bottom_final': y_bottom,
        'y_top_final': y_top
    }


def plot_fields_comparison_2d(
    Lx, Dmax, Nx, N,
    p_init, u_init, v_init,
    p_final, u_final, v_final,
    j_start_init, j_end_init, j_start_final, j_end_final,
    y_bottom_init, y_top_init, y_bottom_final, y_top_final
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    hx = float(Lx) / int(Nx)
    hy = 2.0 * float(Dmax) / int(N)

    # Centres des cellules
    Xc = (np.arange(Nx) + 0.5) * hx
    Yc = -Dmax + (np.arange(N) + 0.5) * hy
    Xgrid, Ygrid = np.meshgrid(Xc, Yc, indexing='ij')

    # Interpolation vitesses vers centres
    u_c_init  = 0.5 * (u_init[0:Nx, :] + u_init[1:Nx + 1, :])
    v_c_init  = 0.5 * (v_init[:, 0:N] + v_init[:, 1:N + 1])
    u_c_final = 0.5 * (u_final[0:Nx, :] + u_final[1:Nx + 1, :])
    v_c_final = 0.5 * (v_final[:, 0:N] + v_final[:, 1:N + 1])

    # Masks
    mask_init  = np.zeros((Nx, N), dtype=bool)
    mask_final = np.zeros((Nx, N), dtype=bool)
    for i in range(Nx):
        mask_init[i,  j_start_init[i]:j_end_init[i]]   = True
        mask_final[i, j_start_final[i]:j_end_final[i]] = True

    p_init_masked    = np.where(mask_init,  p_init,    np.nan)
    u_c_init_masked  = np.where(mask_init,  u_c_init,  np.nan)
    v_c_init_masked  = np.where(mask_init,  v_c_init,  np.nan)
    p_final_masked   = np.where(mask_final, p_final,   np.nan)
    u_c_final_masked = np.where(mask_final, u_c_final, np.nan)
    v_c_final_masked = np.where(mask_final, v_c_final, np.nan)

    # -------------- WALLS DRAWING FUNCTION + SCALE BAR ----------------
    def draw_walls_and_scale(ax, y_bottom, y_top, title):
        for i in range(Nx):
            x0, x1 = i * hx, (i + 1) * hx
            ax.plot([x0, x1], [y_bottom[i], y_bottom[i]], 'w-', linewidth=2)
            ax.plot([x0, x1], [y_top[i],    y_top[i]],    'w-', linewidth=2)
            ax.plot([x0, x1], [y_bottom[i], y_bottom[i]], 'k-', linewidth=3.2)
            ax.plot([x0, x1], [y_top[i],    y_top[i]],    'k-', linewidth=3.2)

        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.set_xlim(0, Lx)
        ax.set_ylim(-Dmax, Dmax)
        ax.set_aspect('equal')
        ax.set_title(title + " (m)", fontweight="bold")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # ----- SCALE BAR (length = Lx/5) -----
        scale_len = Lx/5  
        x0 = 0.02*Lx
        y0 = -Dmax*0.92
        ax.plot([x0, x0 + scale_len], [y0, y0], 'k-', linewidth=4)
        ax.text(x0 + scale_len/2, y0 - Dmax*0.07,
                f"{scale_len:.3g} m", ha='center', va='top', fontsize=10, color='k')

    # ================= PRESSURE (Pa) =================
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    im1 = ax1.contourf(Xgrid.T, Ygrid.T, p_init_masked.T, levels=20, cmap='inferno')
    draw_walls_and_scale(ax1, y_bottom_init, y_top_init, "Pression - État Initial")
    cbar = fig1.colorbar(im1, ax=ax1)
    cbar.set_label("Pression (Pa)")

    im2 = ax2.contourf(Xgrid.T, Ygrid.T, p_final_masked.T, levels=20, cmap='inferno')
    draw_walls_and_scale(ax2, y_bottom_final, y_top_final, "Pression - État Final")
    cbar = fig1.colorbar(im2, ax=ax2)
    cbar.set_label("Pression (Pa)")

    plt.tight_layout()
    plt.show()

    # ================= Ux (m/s) =================
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 10))
    im3 = ax3.contourf(Xgrid.T, Ygrid.T, u_c_init_masked.T, levels=20, cmap='viridis')
    draw_walls_and_scale(ax3, y_bottom_init, y_top_init, "Vitesse Ux - État Initial")
    cbar = fig2.colorbar(im3, ax=ax3)
    cbar.set_label("Vitesse Ux (m/s)")

    im4 = ax4.contourf(Xgrid.T, Ygrid.T, u_c_final_masked.T, levels=20, cmap='viridis')
    draw_walls_and_scale(ax4, y_bottom_final, y_top_final, "Vitesse Ux - État Final")
    cbar = fig2.colorbar(im4, ax=ax4)
    cbar.set_label("Vitesse Ux (m/s)")

    plt.tight_layout()
    plt.show()

    # ================= Uy (m/s) =================
    fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(14, 10))
    im5 = ax5.contourf(Xgrid.T, Ygrid.T, v_c_init_masked.T, levels=20, cmap='plasma')
    draw_walls_and_scale(ax5, y_bottom_init, y_top_init, "Vitesse Uy - État Initial")
    cbar = fig3.colorbar(im5, ax=ax5)
    cbar.set_label("Vitesse Uy (m/s)")

    im6 = ax6.contourf(Xgrid.T, Ygrid.T, v_c_final_masked.T, levels=20, cmap='plasma')
    draw_walls_and_scale(ax6, y_bottom_final, y_top_final, "Vitesse Uy - État Final")
    cbar = fig3.colorbar(im6, ax=ax6)
    cbar.set_label("Vitesse Uy (m/s)")

    plt.tight_layout()
    plt.show()



def plot_streamlines_comparison(Lx, Dmax, Nx, N, u_init, v_init, u_final, v_final,
                                j_start_init, j_end_init, j_start_final, j_end_final,
                                y_bottom_init, y_top_init, y_bottom_final, y_top_final):
    """Lignes de courant (streamlines) pour configuration initiale et finale."""
    hx = float(Lx) / int(Nx)
    hy = 2.0 * float(Dmax) / int(N)
    
    # Centres pour interpolation
    Xc = (np.arange(Nx) + 0.5) * hx
    Yc = -Dmax + (np.arange(N) + 0.5) * hy
    Xgrid, Ygrid = np.meshgrid(Xc, Yc, indexing='ij')
    
    # Interpolation aux centres
    u_c_init = 0.5 * (u_init[0:Nx, :] + u_init[1:Nx + 1, :])
    v_c_init = 0.5 * (v_init[:, 0:N] + v_init[:, 1:N + 1])
    
    u_c_final = 0.5 * (u_final[0:Nx, :] + u_final[1:Nx + 1, :])
    v_c_final = 0.5 * (v_final[:, 0:N] + v_final[:, 1:N + 1])
    
    # Masques
    mask_init = np.zeros((Nx, N), dtype=bool)
    mask_final = np.zeros((Nx, N), dtype=bool)
    for i in range(Nx):
        mask_init[i, j_start_init[i]:j_end_init[i]] = True
        mask_final[i, j_start_final[i]:j_end_final[i]] = True
    
    # Application masques (mettre 0 hors domaine)
    u_c_init_plot = np.where(mask_init, u_c_init, 0.0)
    v_c_init_plot = np.where(mask_init, v_c_init, 0.0)
    
    u_c_final_plot = np.where(mask_final, u_c_final, 0.0)
    v_c_final_plot = np.where(mask_final, v_c_final, 0.0)
    
    # Vitesse magnitude
    speed_init = np.sqrt(u_c_init**2 + v_c_init**2)
    speed_final = np.sqrt(u_c_final**2 + v_c_final**2)
    
    speed_init_plot = np.where(mask_init, speed_init, np.nan)
    speed_final_plot = np.where(mask_final, speed_final, np.nan)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Configuration initiale
    im1 = ax1.contourf(Xgrid.T, Ygrid.T, speed_init_plot.T, levels=20, cmap='jet', alpha=0.7)
    ax1.streamplot(Xgrid.T, Ygrid.T, u_c_init_plot.T, v_c_init_plot.T, 
                   color='black', linewidth=1.0, density=1.5, arrowsize=1.2)
    
    # Contour du tube initial
    for i in range(Nx):
        ax1.plot([i * hx, (i + 1) * hx], [y_bottom_init[i], y_bottom_init[i]], 
                'b-', linewidth=2)
        ax1.plot([i * hx, (i + 1) * hx], [y_top_init[i], y_top_init[i]], 
                'b-', linewidth=2)
    ax1.axhline(y=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(-Dmax, Dmax)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('Lignes de courant - Configuration initiale', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(im1, ax=ax1, label='Vitesse magnitude')
    
    # Configuration finale
    im2 = ax2.contourf(Xgrid.T, Ygrid.T, speed_final_plot.T, levels=20, cmap='jet', alpha=0.7)
    ax2.streamplot(Xgrid.T, Ygrid.T, u_c_final_plot.T, v_c_final_plot.T,
                   color='black', linewidth=1.0, density=1.5, arrowsize=1.2)
    
    # Contour du tube final
    for i in range(Nx):
        ax2.plot([i * hx, (i + 1) * hx], [y_bottom_final[i], y_bottom_final[i]], 
                'r-', linewidth=2)
        ax2.plot([i * hx, (i + 1) * hx], [y_top_final[i], y_top_final[i]], 
                'r-', linewidth=2)
    ax2.axhline(y=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    
    ax2.set_xlim(0, Lx)
    ax2.set_ylim(-Dmax, Dmax)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title('Lignes de courant - Configuration finale', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(im2, ax=ax2, label='Vitesse magnitude')
    
    plt.tight_layout()
    plt.show()


def plot_resistance(
    Lx, Nx, hy, eta,
    j_start_init, j_end_init,
    j_start_final, j_end_final,
):
    """
    Trace la résistance de Poiseuille sans utiliser Q :
      - r(x) = 128*eta / (pi * D(x)^4)        [Pa·s·m^-4]  (résistance par unité de longueur)
      - R(x) = ∫ r(x) dx  ≈ cumsum(r_i * hx)  [Pa·s·m^-3]  (résistance cumulée)

    D(x) est estimé du maillage: D_i = (j_end[i] - j_start[i]) * hy.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    hx = float(Lx) / int(Nx)
    x_centers = (np.arange(Nx) + 0.5) * hx
    x_edges   = (np.arange(Nx + 1)) * hx

    # --- Diamètres locaux issus du domaine actif ---
    D_init  = (np.array(j_end_init)  - np.array(j_start_init))  * hy
    D_final = (np.array(j_end_final) - np.array(j_start_final)) * hy

    # Sécurité numérique : éviter D=0
    eps = 1e-12
    D_init  = np.where(D_init  > eps, D_init,  np.nan)
    D_final = np.where(D_final > eps, D_final, np.nan)

    # --- Résistance par unité de longueur (densité) ---
    r_init  = 128.0 * eta / (np.pi * D_init**4)    # Pa·s·m^-4
    r_final = 128.0 * eta / (np.pi * D_final**4)   # Pa·s·m^-4

    # --- Résistance cumulée R(x) ---
    R_init  = np.cumsum(r_init)  * hx              # Pa·s·m^-3
    R_final = np.cumsum(r_final) * hx              # Pa·s·m^-3

             # Pa·s·m^-3 (linéaire en x)

    # ================== FIGURE 1 : r(x) ==================
    fig1, ax1 = plt.subplots(figsize=(10, 4.6))
    ax1.plot(x_centers, r_init,  'b-', lw=2, label='r(x) initiale')
    ax1.plot(x_centers, r_final, 'r-', lw=2, label='r(x) finale')

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel(r"Résistance par longueur $r(x)$ (Pa·s·m$^{-4}$)")
    ax1.set_title("Densité de résistance de Poiseuille")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    plt.show()

    # ================== FIGURE 2 : R(x) ==================
    fig2, ax2 = plt.subplots(figsize=(10, 4.6))
    ax2.plot(x_centers, R_init,  'b-', lw=2, label='R cumulée initiale')
    ax2.plot(x_centers, R_final, 'r-', lw=2, label='R cumulée finale')

    ax2.set_xlabel("x (m)")
    ax2.set_ylabel(r"Résistance cumulée $R(x)$ (Pa·s·m$^{-3}$)")
    ax2.set_title("Résistance de Poiseuille cumulée le long du tube")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()


def plot_convergence():
    plt.figure(figsize=(8, 5))
    plt.semilogy(res['convergence_history'], 'o-')
    plt.xlabel('Itération')
    plt.ylabel('Erreur')
    plt.title('Convergence de la méthode itérative')
    plt.grid(True)
    plt.tight_layout()
    plt.show()






if __name__ == '__main__':
    # Paramètres de simulation
    Lx = 12.0
    Nx = 30
    mu = 0.01
    Pin, Pout = 340.0, 12.0
    Dmax = 1.5
    P1=500.0
    P2=500.0
    Pext=600.0

    # Profil initial: diamètre uniforme
    D_init = loi_lambert(Pt=(P1-Pext), Dmax=Dmax, alpha0=0.2, P1=500.0, P2=500.0, n1=2.0, n2=2.0)

    N = 30

    # Résolution
    res = solve_stokes_souple_sections(
        Lx=Lx, Nx=Nx, mu=mu, Pin=Pin, Pout=Pout,
        D_init=D_init, Dmax=Dmax, N=N,
        Pext=400.0,
        alpha0=0.2, P1=500.0, P2=500.0, n1=2.0, n2=2.0,
        relax=0.3, tol=1e-4, maxiter=100
    )
    
    # 1) Visualisation grille initiale vs finale
    plot_initial_final_grid(
        Lx=Lx, Dmax=Dmax, Nx=Nx, N=N,
        j_start_init=res['j_start_init'],
        j_end_init=res['j_end_init'],
        j_start_final=res['j_start_final'],
        j_end_final=res['j_end_final'],
        y_bottom_init=res['y_bottom_init'],
        y_top_init=res['y_top_init'],
        y_bottom_final=res['y_bottom_final'],
        y_top_final=res['y_top_final']
    )
    
    # 2) Comparaison des champs p, u, v (2D)
    plot_fields_comparison_2d(
        Lx=Lx, Dmax=Dmax, Nx=Nx, N=N,
        p_init=res['p_init'], u_init=res['u_init'], v_init=res['v_init'],
        p_final=res['p'], u_final=res['u'], v_final=res['v'],
        j_start_init=res['j_start_init'], j_end_init=res['j_end_init'],
        j_start_final=res['j_start_final'], j_end_final=res['j_end_final'],
        y_bottom_init=res['y_bottom_init'], y_top_init=res['y_top_init'],
        y_bottom_final=res['y_bottom_final'], y_top_final=res['y_top_final']
    )
    
    # 3) Lignes de courant (streamlines)
    plot_streamlines_comparison(
        Lx=Lx, Dmax=Dmax, Nx=Nx, N=N,
        u_init=res['u_init'], v_init=res['v_init'],
        u_final=res['u'], v_final=res['v'],
        j_start_init=res['j_start_init'], j_end_init=res['j_end_init'],
        j_start_final=res['j_start_final'], j_end_final=res['j_end_final'],
        y_bottom_init=res['y_bottom_init'], y_top_init=res['y_top_init'],
        y_bottom_final=res['y_bottom_final'], y_top_final=res['y_top_final']
    )


    # 4) Résistance hydraulique
    hy = 2.0 * Dmax / N
    plot_resistance(
    Lx=Lx, Nx=Nx, hy=hy,
    j_start_init=res['j_start_init'], j_end_init=res['j_end_init'],
    j_start_final=res['j_start_final'], j_end_final=res['j_end_final'],
    eta=1.8e-5,  # viscosité air (Pa·s)
)
    # 5) Affichage de la convergence

    plot_convergence()



    

    
