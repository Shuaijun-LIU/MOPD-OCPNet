import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import cartopy.crs as ccrs
from cartopy import feature as cfeature
from shapely.vectorized import contains
from scipy.ndimage import gaussian_filter
import math
from typing import List, Optional

# -----------------------------
# Utilities: land/ocean mask, faux land elevation, colormaps
# -----------------------------
def compute_ocean_mask(lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    """Return True for ocean, False for land."""
    land_feature = cfeature.NaturalEarthFeature('physical', 'land', '10m')
    land_geoms = list(land_feature.geometries())
    ocean_mask = np.ones(lon_grid.shape, dtype=bool)
    for geom in land_geoms:
        ocean_mask &= ~contains(geom, lon_grid, lat_grid)
    return ocean_mask

def build_fake_land_elevation(lon_grid: np.ndarray, lat_grid: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """Generate a ‘satellite-style’ land texture using smoothed noise (land only)."""
    elev_noise = np.random.rand(*lon_grid.shape)
    land_elev = gaussian_filter(elev_noise, sigma=3) * 1000
    land_elev[ocean_mask] = np.nan
    return land_elev

def get_land_cmap() -> mcolors.Colormap:
    land_colors = [(0.0, "#2e4536"), (0.5, "#4e5e3c"), (1.0, "#a69176")]
    return mcolors.LinearSegmentedColormap.from_list("satellite_land", land_colors)

def get_pollutant_cmap() -> mcolors.Colormap:
    pollutant_colors = [
        (0.0, "#b3e6ff"), (0.2, "#9ad1f0"), (0.4, "#80bccc"),
        (0.6, "#f0e68c"), (0.8, "#ff9999"), (1.0, "#ff0000")
    ]
    return mcolors.LinearSegmentedColormap.from_list("pollutant_custom", pollutant_colors)

# -----------------------------
# Core class: single-axis animation (+ optional grid panels)
# -----------------------------
class PollutantDiffusionPlotter:
    def __init__(
        self,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        pollutant_stack: List[np.ndarray],
        days: List[int],
        pollutant_name: str = "Pollutant Dispersion",
        threshold: float = 0.2,
        contour_levels: Optional[np.ndarray] = None,
    ):
        self.lon_grid = lon_grid
        self.lat_grid = lat_grid
        self.pollutant_stack = pollutant_stack
        self.days = days
        self.pollutant_name = pollutant_name
        self.threshold = threshold
        self.contour_levels = contour_levels if contour_levels is not None else np.linspace(0.2, 1.0, 9)

        # Precompute static layers
        self.ocean_mask = compute_ocean_mask(lon_grid, lat_grid)
        self.land_elev = build_fake_land_elevation(lon_grid, lat_grid, self.ocean_mask)
        self.land_cmap = get_land_cmap()
        self.pollutant_cmap = get_pollutant_cmap()

        # Matplotlib state
        self._fig = None
        self._ax = None
        self._pc_land = None
        self._cf = None
        self._cl = None

    # ---------- Single-axis animation ----------
    def _init_ax(self, figsize=(7, 6)):
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)

        ax.set_extent([
            np.nanmin(self.lon_grid), np.nanmax(self.lon_grid),
            np.nanmin(self.lat_grid), np.nanmax(self.lat_grid)
        ], crs=proj)
        ax.set_facecolor('#000c3f')
        gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # Land base
        self._pc_land = ax.pcolormesh(
            self.lon_grid, self.lat_grid, self.land_elev,
            cmap=self.land_cmap, shading='auto', transform=proj, zorder=1
        )

        ax.coastlines(resolution='10m', color='black', linewidth=0.3, zorder=3)
        ax.set_title(self.pollutant_name, fontsize=14)
        self._fig, self._ax = fig, ax

    def _render_one_frame(self, idx: int):
        """Render frame idx; clear previous filled/line contours."""
        data = np.copy(self.pollutant_stack[idx])
        data[data < self.threshold] = np.nan
        data[~self.ocean_mask] = np.nan

        if self._cf is not None:
            for c in self._cf.collections:
                c.remove()
        if self._cl is not None:
            for c in self._cl.collections:
                c.remove()

        self._cf = self._ax.contourf(
            self.lon_grid, self.lat_grid, data,
            levels=self.contour_levels, cmap=self.pollutant_cmap, alpha=1.0,
            transform=ccrs.PlateCarree(), zorder=2
        )
        self._cl = self._ax.contour(
            self.lon_grid, self.lat_grid, data,
            levels=self.contour_levels, colors='#aa66f5', linewidths=0.6,
            transform=ccrs.PlateCarree(), zorder=3
        )
        try:
            self._ax.clabel(self._cl, inline=True, fmt="%.2f", fontsize=8, colors="#001f7f")
        except Exception:
            pass

        self._ax.set_title(f"{self.pollutant_name} — Day {self.days[idx]}", fontsize=14)
        return self._cf.collections + self._cl.collections

    def animate(self, interval_ms: int = 700, figsize=(7, 6), blit: bool = False) -> animation.FuncAnimation:
        """Create and return the animation object (does not save)."""
        if self._fig is None:
            self._init_ax(figsize=figsize)

        def _init():
            return self._render_one_frame(0)

        def _update(i):
            return self._render_one_frame(i)

        anim = animation.FuncAnimation(
            self._fig,
            _update,
            init_func=_init,
            frames=len(self.pollutant_stack),
            interval=interval_ms,
            blit=blit
        )

        # Single shared colorbar
        cbar_ax = self._fig.add_axes([0.90, 0.20, 0.02, 0.6])
        cb = self._fig.colorbar(self._cf, cax=cbar_ax)
        cb.set_label('Pollutant Concentration')

        self._fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.92)
        return anim

    def save_gif(self, anim: animation.FuncAnimation, outfile: str = "pollutant_diffusion.gif", fps: int = 1):
        """Save GIF via PillowWriter."""
        writer = animation.PillowWriter(fps=fps)
        anim.save(outfile, writer=writer, dpi=200)
        print(f"Saved GIF -> {outfile}")

    def save_mp4(self, anim: animation.FuncAnimation, outfile: str = "pollutant_diffusion.mp4", fps: int = 2):
        """Save MP4 via FFMpegWriter (requires ffmpeg installed)."""
        Writer = animation.FFMpegWriter
        writer = Writer(fps=fps, metadata=dict(artist='plotter'), bitrate=1800)
        anim.save(outfile, writer=writer, dpi=200)
        print(f"Saved MP4 -> {outfile}")

    # ---------- Optional: grid panels export ----------
    def save_grid_panels(self, outfile_prefix="pollutant_panels", ncols=3, dpi=300):
        """Export all timesteps as a grid of small multiples (PNG/PDF/EPS)."""
        n = len(self.pollutant_stack)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(4 * ncols, 5 * nrows),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        axes = np.array(axes).flatten()
        fig.suptitle(self.pollutant_name, fontsize=16)

        for i in range(n):
            ax = axes[i]
            ax.set_extent([self.lon_grid.min(), self.lon_grid.max(),
                           self.lat_grid.min(), self.lat_grid.max()], crs=ccrs.PlateCarree())
            ax.set_facecolor('#000c3f')
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False

            # Land base
            ax.pcolormesh(self.lon_grid, self.lat_grid, self.land_elev,
                          cmap=self.land_cmap, shading='auto', transform=ccrs.PlateCarree())

            data = np.copy(self.pollutant_stack[i])
            data[data < self.threshold] = np.nan
            data[~self.ocean_mask] = np.nan

            cf = ax.contourf(self.lon_grid, self.lat_grid, data,
                             levels=self.contour_levels, cmap=self.pollutant_cmap, alpha=1,
                             transform=ccrs.PlateCarree())
            cl = ax.contour(self.lon_grid, self.lat_grid, data,
                            levels=self.contour_levels, colors='#aa66f5', linewidths=0.5,
                            transform=ccrs.PlateCarree())
            try:
                ax.clabel(cl, inline=True, fmt="%.2f", fontsize=8, colors="#001f7f")
            except Exception:
                pass
            ax.coastlines(resolution='10m', color='black', linewidth=0.2)
            ax.set_title(f"Day {self.days[i]}", fontsize=12)

        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        cb = fig.colorbar(cf, cax=cbar_ax)
        cb.set_label('Pollutant Concentration')
        fig.subplots_adjust(left=0.05, right=0.88, bottom=0.05, top=0.9)

        fig.savefig(f"{outfile_prefix}.png", dpi=dpi)
        fig.savefig(f"{outfile_prefix}.pdf", dpi=dpi)
        fig.savefig(f"{outfile_prefix}.eps", dpi=dpi)
        print(f"Saved panels -> {outfile_prefix}.png / .pdf / .eps")
        plt.close(fig)

# -----------------------------
# Example data generator
# -----------------------------
def simulate_gaussian_fields(lon_grid: np.ndarray, lat_grid: np.ndarray, days: List[int]) -> List[np.ndarray]:
    """Random Gaussian blobs with small noise, normalized to [0,1]."""
    stack = []
    for _ in days:
        center_lon = 135.0 + np.random.uniform(-1, 1)
        center_lat = 35.0 + np.random.uniform(-1, 1)
        scale = np.random.uniform(1.0, 3.0)
        dist_lon = lon_grid - center_lon
        dist_lat = lat_grid - center_lat
        field = np.exp(-((dist_lon**2 + dist_lat**2) / (2 * scale**2)))
        field += 0.1 * np.random.rand(*lon_grid.shape)
        field /= np.nanmax(field)
        stack.append(field)
    return stack

# -----------------------------
# Demo runner
# -----------------------------
def run_demo_animation():
    # Domain & grid
    lon_min, lon_max = 130, 145
    lat_min, lat_max = 30, 45
    resolution = 0.1
    lons = np.arange(lon_min, lon_max + resolution, resolution)
    lats = np.arange(lat_min, lat_max + resolution, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Timeline
    days = [5, 20, 35, 55, 75, 110]
    pollutant_stack = simulate_gaussian_fields(lon_grid, lat_grid, days)

    plotter = PollutantDiffusionPlotter(
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        pollutant_stack=pollutant_stack,
        days=days,
        pollutant_name="Simulated Pollutant Dispersion",
        threshold=0.2,
        contour_levels=np.linspace(0.2, 1.0, 9)
    )

    # Create animation (display) + export
    anim = plotter.animate(interval_ms=700, figsize=(7, 6), blit=False)
    plotter.save_gif(anim, outfile="pollutant_diffusion.gif", fps=1)
    plotter.save_mp4(anim, outfile="pollutant_diffusion.mp4", fps=2)

    # Optional: small-multiples figure
    # plotter.save_grid_panels(outfile_prefix="pollutant_panels", ncols=3, dpi=300)

    plt.show()

if __name__ == "__main__":
    run_demo_animation()