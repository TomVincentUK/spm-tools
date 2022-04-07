import numpy as np

# Base coords used to generate images
x, d_x = np.linspace(-1, 1, 512, retstep=True)
y, d_y = np.linspace(-1, 1, 512, retstep=True)
X, Y = np.meshgrid(x, y)
extent = x[0] - d_x / 2, x[-1] + d_x / 2, y[0] - d_y / 2, y[-1] + d_y / 2
coords = dict(X=X, Y=Y, extent=extent)
R = np.linalg.norm([X, Y], axis=0)
THETA = np.arctan2(Y, X)

# Chequerboard patterns
square_size = 1 / 4
check = dict(
    Z=np.sign(np.sin(X * np.pi / square_size) * np.sin(Y * np.pi / square_size)),
    **coords
)
check_diag = dict(
    Z=np.sign(
        np.sin((X + Y) * np.pi / (np.sqrt(2) * square_size))
        * np.sin((X - Y) * np.pi / (np.sqrt(2) * square_size))
    ),
    **coords
)
anisotropic = dict(
    Z=(X % square_size) / square_size + np.floor(Y / square_size), **coords
)

# Radially symmetric functions
radius = 1 / 3
pillar = dict(Z=(R < radius).astype(float), **coords)
bump = dict(Z=(np.exp(-((R / radius) ** 2) / 2)), **coords)
