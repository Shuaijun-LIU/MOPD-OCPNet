import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


n_particles = 300
n_steps = 25
lats = np.zeros((n_particles, n_steps))
lons = np.zeros((n_particles, n_steps))

lats[:, 0] = 75 + np.random.randn(n_particles) * 0.5
lons[:, 0] = -30 + np.random.randn(n_particles) * 0.5

for t in range(1, n_steps):
    lats[:, t] = lats[:, t-1] + np.random.randn(n_particles) * 0.8
    lons[:, t] = lons[:, t-1] + np.random.randn(n_particles) * 0.8

cmap = plt.cm.jet
norm = plt.Normalize(vmin=0, vmax=n_steps)

fig, ax = plt.subplots(1, 1, figsize=(8, 8),
                       subplot_kw={'projection': ccrs.NorthPolarStereo()})

ax.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, color='lightgray')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

for i in range(n_particles):
    for t in range(1, n_steps):
        ax.plot([lons[i, t-1], lons[i, t]],
                [lats[i, t-1], lats[i, t]],
                transform=ccrs.PlateCarree(),
                color=cmap(norm(t)),
                linewidth=0.7)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.6, pad=0.05)
cbar.set_label("Time elapsed (steps / months)")

plt.title("Pollutant dispersion simulation", fontsize=14)
plt.show()