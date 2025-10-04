import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def propagador(x, t, x_prima, t_prima, m, hbar):
    """
    Calcula el propagador cuántico para una partícula libre.
    
    Parámetros:
    -----------
    x : float
        Posición final
    t : float
        Tiempo final
    x_prima : float
        Posición inicial
    t_prima : float
        Tiempo inicial (t_0)
    m : float
        Masa de la partícula
    hbar : float
        Constante de Planck reducida (ℏ)
    
    Retorna:
    --------
    complex : Valor del propagador u(x, t, x', t')
    """
    
    # Diferencia temporal
    dt = t - t_prima
    
    if dt <= 0:
        raise ValueError("t debe ser mayor que t_prima")
    
    # Diferencia espacial
    dx = x - x_prima
    
    # Factor de prefactor
    prefactor = (m / (2 * np.pi * hbar * 1j * dt))**(1/2)
    
    # Exponente complejo
    exponente = 1j * m * dx**2 / (2 * hbar * dt)
    
    # Propagador completo
    u = prefactor * np.exp(exponente)
    
    return u


def psi_inicial(x_prima, x0=0, sigma=1, k0=0):
    """
    Paquete de ondas gaussiano inicial.
    
    Parámetros:
    -----------
    x_prima : float o array
        Posición
    x0 : float
        Posición central del paquete
    sigma : float
        Ancho del paquete
    k0 : float
        Momento inicial (en unidades de hbar)
    
    Retorna:
    --------
    complex : ψ(x', 0)
    """
    normalizacion = 1/(np.sqrt(2 * np.pi) * sigma)
    gaussian = np.exp(-(x_prima - x0)**2 / (2 * sigma**2))
    fase = np.exp(1j * k0 * x_prima/hbar)
    
    return normalizacion * gaussian * fase


def evolucion_temporal(x, t, m, hbar, x0=0, sigma=1, k0=0, x_limits=(-50, 50)):
    """
    Calcula ψ(x, t) usando la integral del propagador.
    
    ψ(x, t) = ∫ u(x, t, x', 0) ψ(x', 0) dx'
    
    Parámetros:
    -----------
    x : float
        Posición donde evaluar ψ(x, t)
    t : float
        Tiempo
    m : float
        Masa de la partícula
    hbar : float
        Constante de Planck reducida
    x0 : float
        Posición central inicial del paquete
    sigma : float
        Ancho inicial del paquete
    k0 : float
        Momento inicial
    x_limits : tuple
        Límites de integración
    
    Retorna:
    --------
    complex : ψ(x, t)
    """
    
    def integrando_real(x_prima):
        prop = propagador(x, t, x_prima, 0, m, hbar)
        psi_0 = psi_inicial(x_prima, x0, sigma, k0)
        resultado = prop * psi_0
        return np.real(resultado)
    
    def integrando_imag(x_prima):
        prop = propagador(x, t, x_prima, 0, m, hbar)
        psi_0 = psi_inicial(x_prima, x0, sigma, k0)
        resultado = prop * psi_0
        return np.imag(resultado)
    
    # Integración numérica de partes real e imaginaria
    real_part, _ = quad(integrando_real, x_limits[0], x_limits[1], limit=100)
    imag_part, _ = quad(integrando_imag, x_limits[0], x_limits[1], limit=100)
    
    return real_part + 1j * imag_part


# Ejemplo de uso: Evolución de un paquete gaussiano
if __name__ == "__main__":
    # Constantes (unidades naturales para simplificar)
    hbar = 1.0
    m = 1.0
    
    # Parámetros del paquete inicial
    x0 = -10      # Centro inicial
    sigma = 1   # Ancho del paquete
    k0 = 1.0      # Momento inicial
    
    # Posiciones y tiempos para evaluar
    x_array = np.linspace(-30, 30, 200)
    tiempos = [0, 5, 10, 20]
    
    # Graficar la evolución
    plt.figure(figsize=(12, 8))
    
    for i, t in enumerate(tiempos):
        plt.subplot(2, 2, i+1)
        
        # Calcular ψ(x, t) para cada posición
        psi_t = []
        for x in x_array:
            if t == 0:
                psi = psi_inicial(x, x0, sigma, k0)
            else:
                psi = evolucion_temporal(x, t, m, hbar, x0, sigma, k0)
            psi_t.append(psi)
        
        psi_t = np.array(psi_t)
        densidad_prob = np.abs(psi_t)**2
        
        # Graficar densidad de probabilidad
        plt.plot(x_array, densidad_prob, 'b-', linewidth=2)
        plt.fill_between(x_array, densidad_prob, alpha=0.3)
        plt.xlabel('Posición x')
        plt.ylabel('|ψ(x,t)|²')
        plt.title(f't = {t}')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(densidad_prob) * 1.1)
    
    plt.tight_layout()
    plt.savefig('evolucion_paquete.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=" * 60)
    print("EVOLUCIÓN TEMPORAL DE UN PAQUETE DE ONDAS GAUSSIANO")
    print("=" * 60)
    print("Parametros iniciales:")
    print(f"  - Posicion central: x0 = {x0}")
    print(f"  - Ancho del paquete: sigma = {sigma}")
    print(f"  - Momento inicial: k0 = {k0}")
    print(f"  - Masa: m = {m}")
    print(f"  - hbar = {hbar}")
    print(f"\nSe han calculado {len(tiempos)} instantes de tiempo")
    print("Grafico guardado como 'evolucion_paquete.png'")