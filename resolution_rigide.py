import numpy as np
import matplotlib.pyplot as plt


def solve_stokes_sections(Lx, Ly_sections, mu, Pin, Pout, Nx, Dmax, N):
    """
    Résolution de Stokes 2D (schéma MAC) dans un domaine composé de Nx sections
    de largeur hx = Lx/Nx. Chaque section i a sa propre hauteur D[i] = Ly_sections[i].
    
    MODIFICATION PRINCIPALE: Le tube est maintenant centré sur y=0, symétrique par rapport
    à l'axe x. La hauteur D[i] définit la demi-hauteur, donc le tube s'étend de 
    y = -D[i]/2 à y = +D[i]/2.
    
    La grille englobante s'étend de -Dmax à +Dmax (hauteur totale 2*Dmax).
    Le maillage vertical hy est calculé sur cette grille étendue: hy = 2*Dmax / N.

    Entrées
    - Lx: longueur totale en x
    - Ly_sections: scalaire (diamètre uniforme) ou tableau de taille Nx (diamètre par section)
    - mu: viscosité dynamique
    - Pin, Pout: pression aux entrées (x=0) et sorties (x=Lx)
    - Nx: nombre de sections en x
    - Dmax: demi-hauteur maximale du domaine (domaine total: -Dmax à +Dmax)
    - N: nombre de divisions verticales pour la grille englobante

    Sorties
    - p: pression de taille (Nx, N)
    - u: vitesse u de taille (Nx+1, N)
    - v: vitesse v de taille (Nx, N+1)
    - Ny: tableau Ny[i] (nombre de cellules actives par section)
    - hx: pas horizontal
    - N: nombre de lignes sur la grille rectangulaire englobante
    - y_bottom: tableau des positions y du bas du tube par section
    - y_top: tableau des positions y du haut du tube par section
    """

    # largeurs et hauteurs
    hx = float(Lx) / int(Nx)
    N = int(N)
    # Grille de -Dmax à +Dmax
    hy = 2.0 * float(Dmax) / float(N)

    # normaliser Ly_sections en tableau de taille Nx (ce sont les diamètres)
    if np.isscalar(Ly_sections):
        H = np.full(int(Nx), float(Ly_sections), dtype=float)
    else:
        H = np.asarray(Ly_sections, dtype=float).ravel()
        if H.size != int(Nx):
            raise ValueError("Ly_sections doit être scalaire ou de taille Nx.")

    # Calcul du nombre de cellules par colonne
    if hy <= 0:
        raise ValueError("hy doit être strictement positif.")
    Ny_cols = np.floor(H / hy).astype(int)
    Ny_cols = np.clip(Ny_cols, 1, N)  # au moins 1 et au plus N

    # Calcul des positions verticales du tube pour chaque section
    # Chaque section i a une demi-hauteur H[i]/2
    # Elle s'étend de -H[i]/2 à +H[i]/2
    # En indices de grille (avec origine à y=-Dmax), on trouve:
    y_bottom = -H / 2.0  # position réelle du bas
    y_top = H / 2.0      # position réelle du haut
    
    # Indices de début (j_start) et de fin (j_end) pour chaque section
    # La grille va de y=-Dmax (j=0) à y=+Dmax (j=N)
    # Position y(j) = -Dmax + j*hy
    j_start = np.floor((y_bottom + Dmax) / hy).astype(int)
    j_end = np.ceil((y_top + Dmax) / hy).astype(int)
    
    # Assurer que les indices sont dans les limites
    j_start = np.clip(j_start, 0, N)
    j_end = np.clip(j_end, 0, N)
    
    # Recalculer Ny_cols comme le nombre de cellules actives
    Ny_cols = j_end - j_start

    # Comptage des inconnues (sur la grille englobante)
    M = int(Nx) * N                   # p : (Nx, N)
    K = (int(Nx) + 1) * N             # u : (Nx+1, N)
    L = int(Nx) * (N + 1)             # v : (Nx, N+1)
    ntot = M + K + L

    # Helpers d'indexation, i en [0..Nx-1], j en [0..N-1]
    def idx_p(i, j):
        return i * N + j

    def idx_u(i, j):  # i en [0..Nx], j en [0..N-1]
        return M + i * N + j

    def idx_v(i, j):  # i en [0..Nx-1], j en [0..N]
        return M + K + i * (N + 1) + j

    # Marqueurs d'appartenance au domaine
    in_p = np.zeros((int(Nx), N), dtype=bool)
    in_u = np.zeros((int(Nx) + 1, N), dtype=bool)
    in_v = np.zeros((int(Nx), N + 1), dtype=bool)

    for i in range(int(Nx)):
        js = j_start[i]
        je = j_end[i]
        in_p[i, js:je] = True                    # cellules actives
        in_v[i, js:je + 1] = True                # faces horizontales actives
    
    # u sur les faces verticales: actif si des cellules existent de part et d'autre
    for j in range(N):
        # Frontière gauche (i=0)
        if j >= j_start[0] and j < j_end[0]:
            in_u[0, j] = True
        # Frontière droite (i=Nx)
        if j >= j_start[int(Nx) - 1] and j < j_end[int(Nx) - 1]:
            in_u[int(Nx), j] = True
        # faces internes
        for i in range(1, int(Nx)):
            js_left = j_start[i - 1]
            je_left = j_end[i - 1]
            js_right = j_start[i]
            je_right = j_end[i]
            # u actif si j est dans les deux sections
            if j >= max(js_left, js_right) and j < min(je_left, je_right):
                in_u[i, j] = True

    # Matrices
    A = np.zeros((ntot, ntot), dtype=float)
    B = np.zeros((ntot, 1), dtype=float)

    # 1) Continuité (div u = 0) sur les cellules actives
    for i in range(int(Nx)):
        for j in range(N):
            m = idx_p(i, j)
            if not in_p[i, j]:
                # en dehors du domaine: fixer p=0 pour découpler
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

    # 2) Équations pour u (faces verticales)
    for i in range(int(Nx) + 1):
        for j in range(N):
            k = idx_u(i, j)

            # En dehors du domaine -> u=0
            if not in_u[i, j]:
                A[k, k] = 1.0
                B[k, 0] = 0.0
                continue

            # Déterminer si on est sur une paroi (haut ou bas local)
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

            # Parois horizontales: u=0
            if j == js_local or j == je_local - 1:
                A[k, k] = 1.0
                B[k, 0] = 0.0
                continue

            # Frontières verticales globales: condition Neumann du/dx = 0
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

    # 3) Équations pour v (faces horizontales)
    for i in range(int(Nx)):
        for j in range(N + 1):
            l = idx_v(i, j)

            # En dehors du domaine
            if not in_v[i, j]:
                A[l, l] = 1.0
                B[l, 0] = 0.0
                continue

            js = j_start[i]
            je = j_end[i]

            # Parois haut/bas: v=0
            if j == js or j == je:
                A[l, l] = 1.0
                B[l, 0] = 0.0
                continue

            # Frontières verticales globales: Neumann dv/dx = 0
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

    # 4) Conditions de pression aux colonnes gauche/droite (cellules actives)
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


def plot_fields_2d(Lx, Dmax, Nx, N, p, u, v, j_start, j_end, y_bottom, y_top):
    """Visualisation 2D des champs de pression et vitesse avec contours du tube"""
    hx = float(Lx) / int(Nx)
    hy = 2.0 * float(Dmax) / int(N)
    
    # Centres des cellules
    Xc = (np.arange(Nx) + 0.5) * hx
    Yc = -Dmax + (np.arange(N) + 0.5) * hy
    Xgrid, Ygrid = np.meshgrid(Xc, Yc, indexing='ij')
    
    # Interpolation des vitesses aux centres
    u_c = 0.5 * (u[0:Nx, :] + u[1:Nx + 1, :])
    v_c = 0.5 * (v[:, 0:N] + v[:, 1:N + 1])
    
    # Masque pour le domaine actif
    mask = np.zeros((Nx, N), dtype=bool)
    for i in range(Nx):
        mask[i, j_start[i]:j_end[i]] = True
    
    # Application des masques
    p_masked = np.where(mask, p, np.nan)
    u_c_masked = np.where(mask, u_c, np.nan)
    v_c_masked = np.where(mask, v_c, np.nan)
    
    # Magnitude de vitesse
    speed = np.sqrt(u_c**2 + v_c**2)
    speed_masked = np.where(mask, speed, np.nan)
    
    # Figure avec 4 sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1) Pression
    ax1 = axes[0, 0]
    im1 = ax1.contourf(Xgrid.T, Ygrid.T, p_masked.T, levels=20, cmap='inferno')
    # Contours du tube
    for i in range(Nx):
        ax1.plot([i * hx, (i + 1) * hx], [y_bottom[i], y_bottom[i]], 'w-', linewidth=2)
        ax1.plot([i * hx, (i + 1) * hx], [y_top[i], y_top[i]], 'w-', linewidth=2)
    ax1.axhline(y=0, color='white', linewidth=1, linestyle='--', alpha=0.7)
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(-Dmax, Dmax)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('Champ de pression', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(im1, ax=ax1, label='Pression')
    
    # 2) Vitesse Ux
    ax2 = axes[0, 1]
    im2 = ax2.contourf(Xgrid.T, Ygrid.T, u_c_masked.T, levels=20, cmap='viridis')
    for i in range(Nx):
        ax2.plot([i * hx, (i + 1) * hx], [y_bottom[i], y_bottom[i]], 'w-', linewidth=2)
        ax2.plot([i * hx, (i + 1) * hx], [y_top[i], y_top[i]], 'w-', linewidth=2)
    ax2.axhline(y=0, color='white', linewidth=1, linestyle='--', alpha=0.7)
    ax2.set_xlim(0, Lx)
    ax2.set_ylim(-Dmax, Dmax)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title('Vitesse Ux', fontsize=13, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(im2, ax=ax2, label='u')
    
    # 3) Vitesse Uy
    ax3 = axes[1, 0]
    im3 = ax3.contourf(Xgrid.T, Ygrid.T, v_c_masked.T, levels=20, cmap='plasma')
    for i in range(Nx):
        ax3.plot([i * hx, (i + 1) * hx], [y_bottom[i], y_bottom[i]], 'w-', linewidth=2)
        ax3.plot([i * hx, (i + 1) * hx], [y_top[i], y_top[i]], 'w-', linewidth=2)
    ax3.axhline(y=0, color='white', linewidth=1, linestyle='--', alpha=0.7)
    ax3.set_xlim(0, Lx)
    ax3.set_ylim(-Dmax, Dmax)
    ax3.set_aspect('equal', adjustable='box')
    ax3.set_title('Vitesse Uy', fontsize=13, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    fig.colorbar(im3, ax=ax3, label='v')
    
    # 4) Magnitude de vitesse avec vecteurs
    ax4 = axes[1, 1]
    im4 = ax4.contourf(Xgrid.T, Ygrid.T, speed_masked.T, levels=20, cmap='jet', alpha=0.8)
    
    # Champ de vecteurs (sous-échantillonnage pour clarté)
    skip = max(1, Nx // 20)
    u_plot = np.where(mask, u_c, 0.0)
    v_plot = np.where(mask, v_c, 0.0)
    ax4.quiver(Xgrid[::skip, ::skip].T, Ygrid[::skip, ::skip].T,
               u_plot[::skip, ::skip].T, v_plot[::skip, ::skip].T,
               color='black', alpha=0.6, scale=None, width=0.003)
    
    for i in range(Nx):
        ax4.plot([i * hx, (i + 1) * hx], [y_bottom[i], y_bottom[i]], 'w-', linewidth=2)
        ax4.plot([i * hx, (i + 1) * hx], [y_top[i], y_top[i]], 'w-', linewidth=2)
    ax4.axhline(y=0, color='white', linewidth=1, linestyle='--', alpha=0.7)
    ax4.set_xlim(0, Lx)
    ax4.set_ylim(-Dmax, Dmax)
    ax4.set_aspect('equal', adjustable='box')
    ax4.set_title('Magnitude de vitesse + vecteurs', fontsize=13, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    fig.colorbar(im4, ax=ax4, label='|V|')
    
    plt.tight_layout()
    plt.show()


def plot_streamlines(Lx, Dmax, Nx, N, u, v, j_start, j_end, y_bottom, y_top):
    """Lignes de courant dans le domaine"""
    hx = float(Lx) / int(Nx)
    hy = 2.0 * float(Dmax) / int(N)
    
    # Centres
    Xc = (np.arange(Nx) + 0.5) * hx
    Yc = -Dmax + (np.arange(N) + 0.5) * hy
    Xgrid, Ygrid = np.meshgrid(Xc, Yc, indexing='ij')
    
    # Interpolation
    u_c = 0.5 * (u[0:Nx, :] + u[1:Nx + 1, :])
    v_c = 0.5 * (v[:, 0:N] + v[:, 1:N + 1])
    
    # Masque
    mask = np.zeros((Nx, N), dtype=bool)
    for i in range(Nx):
        mask[i, j_start[i]:j_end[i]] = True
    
    # Pour streamplot, mettre 0 hors domaine
    u_plot = np.where(mask, u_c, 0.0)
    v_plot = np.where(mask, v_c, 0.0)
    
    # Vitesse magnitude
    speed = np.sqrt(u_c**2 + v_c**2)
    speed_masked = np.where(mask, speed, np.nan)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Fond coloré par magnitude
    im = ax.contourf(Xgrid.T, Ygrid.T, speed_masked.T, levels=20, cmap='jet', alpha=0.7)
    
    # Lignes de courant
    ax.streamplot(Xgrid.T, Ygrid.T, u_plot.T, v_plot.T,
                  color='black', linewidth=1.2, density=1.8, arrowsize=1.5)
    
    # Contours du tube
    for i in range(Nx):
        ax.plot([i * hx, (i + 1) * hx], [y_bottom[i], y_bottom[i]], 'w-', linewidth=2.5)
        ax.plot([i * hx, (i + 1) * hx], [y_top[i], y_top[i]], 'w-', linewidth=2.5)
    ax.axhline(y=0, color='white', linewidth=1.5, linestyle='--', alpha=0.7, label='Axe de symétrie')
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(-Dmax, Dmax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Lignes de courant (streamlines)', fontsize=14, fontweight='bold')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(loc='upper right')
    fig.colorbar(im, ax=ax, label='Vitesse magnitude')
    
    plt.tight_layout()
    plt.show()


def plot_flow_rate(Lx, Nx, hy, u, j_start, j_end):
    """Débit volumique le long du tube"""
    hx = float(Lx) / int(Nx)
    x_sections = (np.arange(Nx + 1)) * hx
    
    Q = np.zeros(Nx + 1)
    for i in range(Nx + 1):
        if i < Nx:
            js = j_start[i]
            je = j_end[i]
            Q[i] = np.sum(u[i, js:je]) * hy
        else:
            # Dernière face (sortie)
            js = j_start[Nx - 1]
            je = j_end[Nx - 1]
            Q[i] = np.sum(u[Nx, js:je]) * hy
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_sections, Q, 'b-o', linewidth=2.5, markersize=6, label='Débit Q(x)')
    ax.axhline(y=np.mean(Q), color='r', linestyle='--', linewidth=2, 
               label=f'Moyenne = {np.mean(Q):.4f}')
    ax.set_xlabel('Position x', fontsize=12)
    ax.set_ylabel('Débit volumique Q', fontsize=12)
    ax.set_title('Débit le long du tube', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_grid_with_tube(Lx, Dmax, Nx, N, j_start, j_end, y_bottom, y_top, 
                        *, grid_color='0.85', tube_edge='tab:red'):
    """Dessine la grille globale centrée (-Dmax à +Dmax) et le tube symétrique"""
    hx = float(Lx) / int(Nx)
    hy = 2.0 * float(Dmax) / int(N)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Grille globale en lignes
    for i in range(int(Nx) + 1):
        x = i * hx
        ax.plot([x, x], [-Dmax, Dmax], color=grid_color, linewidth=0.8)
    for j in range(int(N) + 1):
        y = -Dmax + j * hy
        ax.plot([0, Lx], [y, y], color=grid_color, linewidth=0.8)

    # Axe x=0
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--', alpha=0.5, label='Axe de symétrie')

    # Cellules actives du tube (contours)
    for i in range(int(Nx)):
        js = j_start[i]
        je = j_end[i]
        for j in range(js, je):
            y_cell_bottom = -Dmax + j * hy
            ax.add_patch(
                plt.Rectangle((i * hx, y_cell_bottom), hx, hy, 
                            fill=False, edgecolor=tube_edge, linewidth=1.0)
            )
        # Lignes supérieure et inférieure du tube
        ax.plot([i * hx, (i + 1) * hx], [y_bottom[i], y_bottom[i]], 
                color=tube_edge, linewidth=1.5)
        ax.plot([i * hx, (i + 1) * hx], [y_top[i], y_top[i]], 
                color=tube_edge, linewidth=1.5)

    ax.set_xlim(0, Lx)
    ax.set_ylim(-Dmax, Dmax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Grille globale centrée (gris) et tube symétrique (rouge)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Exemple d'utilisation avec paramètres demandés
    Lx = 12.0
    Nx = 50
    mu = 0.01
    Pin, Pout = 4.0, 2.0

    # Profil de diamètre par colonne (symétrique autour de y=0)
    Ly_array = 2

    # Choix du Dmax (demi-hauteur max) et de N
    Dmax = 2.5
    N_target = 20

    # Résolution
    p, u, v, Ny_cols, hx, N, j_start, j_end, y_bottom, y_top = solve_stokes_sections(
        Lx, Ly_array, mu, Pin, Pout, Nx, Dmax, N_target
    )

    # 1) Plot de la grille globale et du tube superposé
    plot_grid_with_tube(Lx, Dmax, Nx, N, j_start, j_end, y_bottom, y_top)

    # 2) Visualisation 2D des champs (p, u, v, magnitude)
    hy = 2.0 * Dmax / N
    plot_fields_2d(Lx, Dmax, Nx, N, p, u, v, j_start, j_end, y_bottom, y_top)
    
    # 3) Lignes de courant
    plot_streamlines(Lx, Dmax, Nx, N, u, v, j_start, j_end, y_bottom, y_top)
    
    # 4) Débit volumique
    plot_flow_rate(Lx, Nx, hy, u, j_start, j_end)