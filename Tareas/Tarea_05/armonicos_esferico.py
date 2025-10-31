import sympy as sp
from sympy import symbols, simplify, integrate, pi, sin, conjugate 
import numpy as np
import matplotlib.pyplot as plt

# Definir símbolos
theta, phi = symbols('theta phi', real=True, positive=True)

def armonico_esferico_simbolico(l, m):  # noqa: E741
    """
    Calcula el armónico esférico Y_l^m de forma simbólica usando SymPy.
    
    Parámetros:
    -----------
    l : int
        Número cuántico principal (l >= 0)
    m : int
        Número cuántico magnético (-l <= m <= l)
    
    Retorna:
    --------
    sympy expression
        Expresión simbólica del armónico esférico Y_l^m(theta, phi)
    """
    from sympy.functions.special.spherical_harmonics import Ynm
    
    # Calcular armónico esférico
    Y = Ynm(l, m, theta, phi)
    
    return Y.expand(func=True)

def densidad_armonico_esferico(l, m, simplificar=True):  # noqa: E741
    """
    Calcula la densidad de probabilidad |Y_l^m|² de forma simbólica.
    
    Parámetros:
    -----------
    l : int
        Número cuántico principal (l >= 0)
    m : int
        Número cuántico magnético (-l <= m <= l)
    simplificar : bool
        Si es True, simplifica la expresión resultante
    
    Retorna:
    --------
    dict
        Diccionario con:
        - 'armonico': expresión del armónico esférico Y_l^m
        - 'conjugado': conjugado complejo de Y_l^m
        - 'densidad': |Y_l^m|² simplificado
    """
    
    print(f"Calculando armonico esferico Y_{l}^{m}...")
    Y = armonico_esferico_simbolico(l, m)
    
    print("Calculando conjugado complejo...")
    Y_conj = conjugate(Y)
    
    print(f"Calculando densidad |Y_{l}^{m}|²...")
    densidad = Y * Y_conj
    
    if simplificar:
        print("Simplificando expresion...")
        densidad = simplify(densidad)
    
    return {
        'armonico': Y,
        'conjugado': Y_conj,
        'densidad': densidad
    }

def mostrar_resultado(l, m, resultado):  # noqa: E741
    """
    Muestra los resultados de forma formateada.
    """
    print("\n" + "="*70)
    print(f"ARMONICO ESFERICO Y_{l}^{m}")
    print("="*70)
    
    print(f"\nY_{l}^{m} =")
    sp.pprint(resultado['armonico'])
    
    print(f"\n\nConjugado Y_{l}^{m}* =")
    sp.pprint(resultado['conjugado'])
    
    print(f"\n\nDensidad |Y_{l}^{m}|² =")
    sp.pprint(resultado['densidad'])
    print("\n" + "="*70)

def graficar_densidad_simbolica(l, m, resolucion=100):  # noqa: E741
    """
    Grafica la densidad calculada simbólicamente.
    
    Parámetros:
    -----------
    l : int
        Número cuántico principal
    m : int
        Número cuántico magnético
    resolucion : int
        Resolución de la malla
    """
    # Calcular densidad simbólica
    resultado = densidad_armonico_esferico(l, m)
    densidad_expr = resultado['densidad']
    
    # Convertir a función numérica
    densidad_func = sp.lambdify((theta, phi), densidad_expr, 'numpy')
    
    # Crear malla
    theta_vals = np.linspace(0, np.pi, resolucion)
    phi_vals = np.linspace(0, 2*np.pi, resolucion)
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
    
    # Evaluar densidad
    try:
        densidad_vals = densidad_func(theta_grid, phi_grid)
        densidad_vals = np.real(densidad_vals)  # Tomar parte real (debería ser real)
    except Exception as e:
        print(f"Error al evaluar: {e}")
        return
    
    # Convertir a coordenadas cartesianas
    r = densidad_vals
    x = r * np.sin(theta_grid) * np.cos(phi_grid)
    y = r * np.sin(theta_grid) * np.sin(phi_grid)
    z = r * np.cos(theta_grid)
    
    # Crear figura
    fig = plt.figure(figsize=(10, 6))
    
    # Gráfico 3D
    ax1 = fig.add_subplot(111, projection='3d')
    surf = ax1.plot_surface(x, y, z, facecolors=plt.cm.viridis(densidad_vals/np.max(densidad_vals)),  # noqa: F841
                            rstride=2, cstride=2, alpha=0.9, antialiased=True)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'|Y_{l}^{m}|² - Densidad de Probabilidad (Simbolico)')
    ax1.set_box_aspect([1,1,1])
    
    # Gráfico 2D
    # ax2 = fig.add_subplot(122)
    # im = ax2.contourf(phi_grid, theta_grid, densidad_vals, levels=50, cmap='viridis')
    # ax2.set_xlabel('φ (azimut)')
    # ax2.set_ylabel('θ (polar)')
    # ax2.set_title(f'Mapa de calor: |Y_{l}^{m}|²')
    # plt.colorbar(im, ax=ax2, label='Densidad')
    
    plt.tight_layout()
    plt.show()

def verificar_normalizacion(l, m):  # noqa: E741
    """
    Verifica que el armónico esférico esté normalizado integrando |Y|² sobre la esfera.
    La integral debería dar 1.
    """
    print(f"\nVerificando normalizacion de Y_{l}^{m}...")
    resultado = densidad_armonico_esferico(l, m)
    densidad = resultado['densidad']
    
    # Integrar sobre la esfera: ∫∫ |Y|² sin(θ) dθ dφ
    print("Integrando ...")
    integral = integrate(densidad * sin(theta), (theta, 0, pi), (phi, 0, 2*pi))
    integral_simplificado = simplify(integral)
    
    print(f"Resultado de la integral: {integral_simplificado}")
    
    if integral_simplificado == 1:
        print(" El armonico esférico está correctamente normalizado!")
    else:
        print(f"La normalización es: {integral_simplificado} (deberia ser 1)")
    
    return integral_simplificado

# Ejemplos de uso
if __name__ == "__main__":
    # Definición número cuantico l
    l = 2  # noqa: E741

    # Definición del número cuantico m
    m = np.arange(0, l, 1)

    for i in m:
        print(f"\nDensidada del armonico: Y_{l}^{i}")
        resultado = densidad_armonico_esferico(l, i)
        verificar_normalizacion(l, i)
        print(f"\n*** GRAFICANDO: Y_{l}^{i} ***")
        graficar_densidad_simbolica(l, i, resolucion=80)
