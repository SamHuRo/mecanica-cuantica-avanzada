import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Definir la grilla esférica
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Coordenadas cartesianas base
x_coord = np.sin(theta) * np.cos(phi)
y_coord = np.sin(theta) * np.sin(phi)
z_coord = np.cos(theta)

# Función para convertir coordenadas esféricas a cartesianas para graficar
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

# Armónicos esféricos para l=1 (orbitales p)
# Combinaciones lineales que dan las funciones direccionales x, y, z

# p_z = Y_1^0 (proporcional a z)
Y_1_0 = np.sqrt(3/(4*np.pi)) * z_coord
p_z = np.abs(Y_1_0)

# p_x = combinación de Y_1^1 y Y_1^{-1} (proporcional a x)
Y_1_1 = -np.sqrt(3/(8*np.pi)) * np.sin(theta) * np.exp(1j*phi)
Y_1_m1 = np.sqrt(3/(8*np.pi)) * np.sin(theta) * np.exp(-1j*phi)
p_x = np.abs((Y_1_m1 - Y_1_1) / np.sqrt(2))

# p_y = combinación de Y_1^1 y Y_1^{-1} (proporcional a y)
p_y = np.abs((Y_1_m1 + Y_1_1) / (1j*np.sqrt(2)))

# Armónicos esféricos para l=2 (orbitales d)
# d_z² (3z²-r²) proporcional a Y_2^0
Y_2_0 = np.sqrt(5/(16*np.pi)) * (3*z_coord**2 - 1)
d_z2 = np.abs(Y_2_0)

# d_xz proporcional a combinación de Y_2^1 y Y_2^{-1}
Y_2_1 = -np.sqrt(15/(8*np.pi)) * np.sin(theta) * np.cos(theta) * np.exp(1j*phi)
Y_2_m1 = np.sqrt(15/(8*np.pi)) * np.sin(theta) * np.cos(theta) * np.exp(-1j*phi)
d_xz = np.abs((Y_2_m1 - Y_2_1) / np.sqrt(2))

# d_yz proporcional a combinación de Y_2^1 y Y_2^{-1}
d_yz = np.abs((Y_2_m1 + Y_2_1) / (1j*np.sqrt(2)))

# d_xy proporcional a combinación de Y_2^2 y Y_2^{-2}
Y_2_2 = np.sqrt(15/(32*np.pi)) * np.sin(theta)**2 * np.exp(2j*phi)
Y_2_m2 = np.sqrt(15/(32*np.pi)) * np.sin(theta)**2 * np.exp(-2j*phi)
d_xy = np.abs((Y_2_m2 + Y_2_2) / (1j*np.sqrt(2)))

# d_x²-y² proporcional a combinación de Y_2^2 y Y_2^{-2}
d_x2y2 = np.abs((Y_2_m2 - Y_2_2) / np.sqrt(2))

# Crear figura para l=1
fig1 = plt.figure(figsize=(15, 5))
fig1.suptitle('Armónicos Esféricos l=1: Orbitales p (Funciones Direccionales)', fontsize=14, fontweight='bold')

orbitals_l1 = [
    (p_x, 'p_x (proporcional a x)'),
    (p_y, 'p_y (proporcional a y)'),
    (p_z, 'p_z (proporcional a z)')
]

for idx, (orbital, title) in enumerate(orbitals_l1, 1):
    ax = fig1.add_subplot(1, 3, idx, projection='3d')
    
    # Usar el valor absoluto del armónico como radio
    x, y, z = spherical_to_cartesian(orbital, theta, phi)
    
    # Colorear según el valor
    colors = orbital
    surf = ax.plot_surface(x, y, z, facecolors=cm.seismic(colors/colors.max()),
                          alpha=0.8, antialiased=True, shade=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    
    # Establecer límites iguales
    max_range = 0.6
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

plt.tight_layout()

# Crear figura para l=2
fig2 = plt.figure(figsize=(18, 10))
fig2.suptitle('Armónicos Esféricos l=2: Orbitales d (Funciones Direccionales)', fontsize=14, fontweight='bold')

orbitals_l2 = [
    (d_z2, 'd_z² (3z²-r²)'),
    (d_xz, 'd_xz'),
    (d_yz, 'd_yz'),
    (d_xy, 'd_xy'),
    (d_x2y2, 'd_x²-y²')
]

for idx, (orbital, title) in enumerate(orbitals_l2, 1):
    ax = fig2.add_subplot(2, 3, idx, projection='3d')
    
    # Usar el valor absoluto del armónico como radio
    x, y, z = spherical_to_cartesian(orbital, theta, phi)
    
    # Colorear según el valor
    colors = orbital
    surf = ax.plot_surface(x, y, z, facecolors=cm.seismic(colors/colors.max()),
                          alpha=0.8, antialiased=True, shade=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    
    # Establecer límites iguales
    max_range = 0.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

plt.tight_layout()
plt.show()

print("Figuras generadas:")
print("\nPara l=1 (orbitales p):")
print("- p_x: combinación lineal que da dependencia con x")
print("- p_y: combinación lineal que da dependencia con y")
print("- p_z: dependencia directa con z")
print("\nPara l=2 (orbitales d):")
print("- d_z²: dependencia con 3z²-r²")
print("- d_xz, d_yz, d_xy: dependencias con productos de coordenadas")
print("- d_x²-y²: dependencia con x²-y²")