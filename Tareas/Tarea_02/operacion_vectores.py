import numpy as np
import matplotlib.pyplot as plt

# Definir dos vectores de ejemplo
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

print("Vectores originales:")
print(f"a = {vector_a}")
print(f"b = {vector_b}")
print("-" * 50)

# =============================================================================
# 1. PRODUCTO EXTERNO (OUTER PRODUCT) - Más común
# =============================================================================
print("1. PRODUCTO EXTERNO (Outer Product)")
print("   Crea matriz M donde M[i,j] = a[i] * b[j]")

# Método 1: usando np.outer()
matriz_outer = np.outer(vector_a, vector_b)
print(f"\nnp.outer(a, b):\n{matriz_outer}")

# Método 2: usando broadcasting
matriz_broadcast = vector_a[:, np.newaxis] * vector_b[np.newaxis, :]
print(f"\nUsando broadcasting:\n{matriz_broadcast}")

# Método 3: usando meshgrid
A, B = np.meshgrid(vector_b, vector_a)
matriz_meshgrid = A * B
print(f"\nUsando meshgrid:\n{matriz_meshgrid}")

print("-" * 50)

# =============================================================================
# 2. CONCATENACIÓN DE VECTORES
# =============================================================================
print("2. CONCATENACIÓN DE VECTORES")

# Como filas
matriz_filas = np.vstack([vector_a, vector_b])
print(f"\nComo filas - np.vstack([a, b]):\n{matriz_filas}")

# Como columnas
matriz_columnas = np.column_stack([vector_a, vector_b])
print(f"\nComo columnas - np.column_stack([a, b]):\n{matriz_columnas}")

# Usando hstack para columnas
matriz_hstack = np.hstack([vector_a.reshape(-1, 1), vector_b.reshape(-1, 1)])
print(f"\nUsando hstack:\n{matriz_hstack}")

print("-" * 50)

# =============================================================================
# 3. MATRIZ DE CORRELACIÓN/COVARIANZA
# =============================================================================
print("3. MATRIZ DE CORRELACIÓN")

# Matriz de covarianza
datos = np.vstack([vector_a, vector_b])
matriz_cov = np.cov(datos)
print(f"\nMatriz de covarianza:\n{matriz_cov}")

# Matriz de correlación
matriz_corr = np.corrcoef(datos)
print(f"\nMatriz de correlación:\n{matriz_corr}")

print("-" * 50)

# =============================================================================
# 4. OPERACIONES ESPECÍFICAS PARA FÍSICA/INGENIERÍA
# =============================================================================
print("4. APLICACIONES EN FÍSICA/INGENIERÍA")

# Para el problema de cristales ópticos - tensor dieléctrico
print("\nEjemplo: Tensor dieléctrico para cristal uniáxico")
k_vector = np.array([0.5, 0.3, 0.8])  # Vector de onda normalizado
n_o, n_e = 1.5, 1.7  # Índices de refracción

# Tensor dieléctrico diagonal
epsilon = np.diag([n_o**2, n_o**2, n_e**2])
print(f"Tensor dieléctrico ε:\n{epsilon}")

# Matriz M = k_i * k_j (producto externo del vector k)
M_kk = np.outer(k_vector, k_vector)
print(f"\nMatriz k_i * k_j:\n{M_kk}")

# Matriz identidad escalada
I_scaled = np.eye(3) * np.dot(k_vector, k_vector)
print(f"\nMatriz k² * I:\n{I_scaled}")

print("-" * 50)

# =============================================================================
# 5. CASOS ESPECIALES
# =============================================================================
print("5. CASOS ESPECIALES")

# Vectores de diferentes dimensiones
v1 = np.array([1, 2])
v2 = np.array([3, 4, 5])

# Producto externo con vectores de diferente tamaño
matriz_diff = np.outer(v1, v2)
print(f"\nProducto externo de vectores diferentes:\nv1={v1}, v2={v2}")
print(f"Resultado (2x3):\n{matriz_diff}")

# Matriz antisimétrica (para productos cruzados)
def matriz_antisimetrica(vector):
    """Crea matriz antisimétrica para producto cruz"""
    v = vector
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

v_cross = np.array([1, 2, 3])
matriz_antisim = matriz_antisimetrica(v_cross)
print(f"\nMatriz antisimétrica para v = {v_cross}:")
print(matriz_antisim)

print("-" * 50)

# =============================================================================
# 6. EJEMPLO PRÁCTICO: SUPERFICIE K
# =============================================================================
print("6. EJEMPLO PRÁCTICO: SUPERFICIE K PARA CRISTAL")

def crear_matriz_superficie_k(kx, ky, kz, epsilon_tensor):
    """
    Crea la matriz para la ecuación de superficie k:
    [k_i*k_j - k²*δ_ij + ω²μ₀ε_ij] = 0
    """
    k_vector = np.array([kx, ky, kz])
    k_squared = np.dot(k_vector, k_vector)
    
    # Matriz k_i * k_j
    kk_matrix = np.outer(k_vector, k_vector)
    
    # Matriz -k² * δ_ij
    identity_scaled = -k_squared * np.eye(3)
    
    # Matriz total (asumiendo ω²μ₀ = 1 para simplicidad)
    matriz_total = kk_matrix + identity_scaled + epsilon_tensor
    
    return matriz_total

# Ejemplo para cristal uniáxico
kx, ky, kz = 0.5, 0.3, 0.8
epsilon_uniaxial = np.diag([1.5**2, 1.5**2, 1.7**2])

matriz_k = crear_matriz_superficie_k(kx, ky, kz, epsilon_uniaxial)
print(f"Matriz para superficie k (cristal uniáxico):")
print(f"k = ({kx}, {ky}, {kz})")
print(matriz_k)

# Determinante (debe ser ≈ 0 para soluciones no triviales)
det = np.linalg.det(matriz_k)
print(f"\nDeterminante: {det:.6f}")

print("-" * 50)

# =============================================================================
# 7. VISUALIZACIÓN
# =============================================================================
print("7. VISUALIZACIÓN DE MATRICES")

# Crear figura con subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Datos para visualizar
matrices = [
    (matriz_outer, "Producto Externo"),
    (matriz_filas, "Vectores como Filas"), 
    (matriz_columnas, "Vectores como Columnas"),
    (matriz_cov, "Matriz Covarianza"),
    (M_kk, "Tensor k⊗k"),
    (epsilon_uniaxial, "Tensor Dieléctrico")
]

for i, (matriz, titulo) in enumerate(matrices):
    ax = axes[i//3, i%3]
    im = ax.imshow(matriz, cmap='RdBu_r', aspect='equal')
    ax.set_title(titulo)
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # Añadir valores en cada celda
    for (j, k), val in np.ndenumerate(matriz):
        ax.text(k, j, f'{val:.2f}', ha='center', va='center',
                color='white' if abs(val) > np.max(np.abs(matriz))/2 else 'black')

plt.tight_layout()
plt.show()

# =============================================================================
# 8. FUNCIONES ÚTILES
# =============================================================================
print("\n8. FUNCIONES ÚTILES PARA CREAR MATRICES")

def ejemplos_creacion_matrices():
    """Ejemplos adicionales de creación de matrices"""
    
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    print("Métodos adicionales:")
    
    # Usando einsum (muy eficiente)
    matriz_einsum = np.einsum('i,j->ij', a, b)
    print(f"\nnp.einsum('i,j->ij', a, b):\n{matriz_einsum}")
    
    # Matriz diagonal con vectores
    matriz_diag_a = np.diag(a)
    print(f"\nnp.diag(a):\n{matriz_diag_a}")
    
    # Matriz de Hankel (para señales)
    from scipy.linalg import hankel
    matriz_hankel = hankel(a, b)
    print(f"\nMatriz de Hankel:\n{matriz_hankel}")
    
    # Matriz de Toeplitz (para convolución)
    from scipy.linalg import toeplitz
    matriz_toeplitz = toeplitz(a, b)
    print(f"\nMatriz de Toeplitz:\n{matriz_toeplitz}")

ejemplos_creacion_matrices()

print("\n" + "="*70)
print("RESUMEN DE MÉTODOS:")
print("="*70)
print("• np.outer(a, b)           → Producto externo (más común)")
print("• np.vstack([a, b])        → Vectores como filas")
print("• np.column_stack([a, b])  → Vectores como columnas")
print("• a[:, None] * b[None, :]  → Broadcasting manual")
print("• np.einsum('i,j->ij',a,b) → Einstein summation (eficiente)")
print("• np.meshgrid()            → Para crear grillas")
print("="*70)
