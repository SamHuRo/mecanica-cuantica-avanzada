import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, factorial
from matplotlib.widgets import RadioButtons, Button

# Constantes
a0 = 1  # Radio de Bohr (unidades atómicas)

def radial_wavefunction(n, l, r):
    """
    Calcula la función de onda radial R_nl(r) para el átomo de hidrógeno.
    
    Parámetros:
    n: número cuántico principal (n >= 1)
    l: número cuántico azimutal (0 <= l < n)
    r: distancia radial (array o escalar)
    """
    rho = 2 * r / (n * a0)
    
    # Normalización
    normalization = np.sqrt(
        (2 / (n * a0))**3 * 
        factorial(n - l - 1) / 
        (2 * n * factorial(n + l))
    )
    
    # Polinomio de Laguerre asociado
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    
    # Función de onda radial
    R_nl = normalization * np.exp(-rho / 2) * rho**l * laguerre
    
    return R_nl

def radial_density(n, l, r):
    """
    Calcula la densidad de probabilidad radial r²|R_nl(r)|².
    """
    R = radial_wavefunction(n, l, r)
    return r**2 * R**2

def plot_hydrogen_orbitals(n, l):
    """
    Grafica la función de onda radial y la densidad de probabilidad.
    """
    # Rango de r
    r_max = n**2 * a0 * 3
    r = np.linspace(0.01, r_max, 1000)
    
    # Calcular funciones
    R = radial_wavefunction(n, l, r)
    P = radial_density(n, l, r)
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Átomo de Hidrógeno - Estado |{n}, {l}⟩', 
                 fontsize=16, fontweight='bold')
    
    # Gráfica 1: Función de onda radial
    ax1.plot(r, R, 'b-', linewidth=2, label=f'$R_{{{n}{l}}}(r)$')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('r (unidades de $a_0$)', fontsize=12)
    ax1.set_ylabel('$R_{nl}(r)$', fontsize=12)
    ax1.set_title('Función de Onda Radial', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Gráfica 2: Densidad de probabilidad radial
    ax2.plot(r, P, 'r-', linewidth=2, label=f'$r^2|R_{{{n}{l}}}(r)|^2$')
    ax2.fill_between(r, P, alpha=0.3, color='red')
    ax2.set_xlabel('r (unidades de $a_0$)', fontsize=12)
    ax2.set_ylabel('$P_{nl}(r)$', fontsize=12)
    ax2.set_title('Densidad de Probabilidad Radial', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # Información adicional
    labels = ['s', 'p', 'd', 'f', 'g', 'h', 'i']
    orbital = f'{n}{labels[l] if l < len(labels) else l}'
    nodes = n - l - 1
    energy = -13.6 / (n**2)
    
    info_text = f'Orbital: {orbital}\nNodos radiales: {nodes}\nEnergía: {energy:.3f} eV'
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def interactive_plot():
    """
    Crea una interfaz interactiva para explorar diferentes estados cuánticos.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(left=0.3, bottom=0.15)
    
    # Estado inicial
    n_current = 3
    l_current = 0
    
    # Función para actualizar gráficas
    def update_plot(n, l):
        r_max = n**2 * a0 * 3
        r = np.linspace(0.01, r_max, 1000)
        R = radial_wavefunction(n, l, r)
        P = radial_density(n, l, r)
        
        ax1.clear()
        ax2.clear()
        
        # Función de onda
        ax1.plot(r, R, 'b-', linewidth=2)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('r (unidades de $a_0$)', fontsize=11)
        ax1.set_ylabel('$R_{nl}(r)$', fontsize=11)
        ax1.set_title(f'Función de Onda Radial $R_{{{n}{l}}}(r)$', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Densidad de probabilidad
        ax2.plot(r, P, 'r-', linewidth=2)
        ax2.fill_between(r, P, alpha=0.3, color='red')
        ax2.set_xlabel('r (unidades de $a_0$)', fontsize=11)
        ax2.set_ylabel('$P_{nl}(r)$', fontsize=11)
        ax2.set_title(f'Densidad de Probabilidad $r^2|R_{{{n}{l}}}(r)|^2$', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Información
        labels = ['s', 'p', 'd', 'f', 'g', 'h']
        orbital = f'{n}{labels[l] if l < len(labels) else l}'
        nodes = n - l - 1
        energy = -13.6 / (n**2)
        
        fig.suptitle(f'Átomo de Hidrógeno - Orbital {orbital} | '
                    f'Nodos: {nodes} | Energía: {energy:.3f} eV',
                    fontsize=13, fontweight='bold')
        
        plt.draw()
    
    # Crear botones para n
    ax_n = plt.axes([0.05, 0.4, 0.15, 0.3])
    radio_n = RadioButtons(ax_n, ('n = 1', 'n = 2', 'n = 3', 'n = 4', 'n = 5'))
    
    # Crear botones para l
    ax_l = plt.axes([0.05, 0.05, 0.15, 0.25])
    radio_l = RadioButtons(ax_l, ('l = 0 (s)', 'l = 1 (p)', 'l = 2 (d)'))
    
    def update_n(label):
        nonlocal n_current, l_current
        n_current = int(label.split('=')[1])
        # Actualizar opciones de l
        l_options = [f'l = {i} ({["s","p","d","f","g","h"][i]})' 
                    for i in range(min(n_current, 6))]
        radio_l.labels = [ax_l.text(0, 0, '') for _ in l_options]
        for i, label_text in enumerate(l_options):
            radio_l.labels[i].set_text(label_text)
        l_current = min(l_current, n_current - 1)
        update_plot(n_current, l_current)
    
    def update_l(label):
        nonlocal l_current
        l_current = int(label.split('=')[1].split()[0])
        if l_current < n_current:
            update_plot(n_current, l_current)
    
    radio_n.on_clicked(update_n)
    radio_l.on_clicked(update_l)
    
    # Gráfica inicial
    update_plot(n_current, l_current)
    
    plt.show()

# Función principal
if __name__ == "__main__":
    print("=== Visualización del Átomo de Hidrógeno ===\n")
    print("Opciones:")
    print("1. Gráficas estáticas para un estado específico")
    print("2. Interfaz interactiva")
    print("3. Comparar varios orbitales")
    
    opcion = input("\nSeleccione una opción (1-3): ")
    
    if opcion == "1":
        n = int(input("Ingrese n (número cuántico principal, n >= 1): "))
        l = int(input(f"Ingrese l (número cuántico azimutal, 0 <= l < {n}): "))
        if 0 <= l < n:
            plot_hydrogen_orbitals(n, l)
        else:
            print("Error: l debe estar entre 0 y n-1")
    
    elif opcion == "2":
        print("\nAbriendo interfaz interactiva...")
        interactive_plot()
    
    elif opcion == "3":
        # Comparar varios orbitales
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Comparación de Orbitales del Átomo de Hidrógeno', 
                    fontsize=16, fontweight='bold')
        
        orbitals = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]
        labels = ['1s', '2s', '2p', '3s', '3p', '3d']
        
        for idx, ((n, l), label) in enumerate(zip(orbitals, labels)):
            ax = axes[idx // 3, idx % 3]
            r_max = n**2 * a0 * 3
            r = np.linspace(0.01, r_max, 500)
            P = radial_density(n, l, r)
            
            ax.plot(r, P, linewidth=2)
            ax.fill_between(r, P, alpha=0.3)
            ax.set_title(f'Orbital {label}', fontweight='bold')
            ax.set_xlabel('r ($a_0$)', fontsize=9)
            ax.set_ylabel('$P_{nl}(r)$', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    else:
        print("Opción no válida")