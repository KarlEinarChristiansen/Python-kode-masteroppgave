from pvlib.iam import martin_ruiz, martin_ruiz_diffuse
import numpy as np
import matplotlib.pyplot as plt

#Inndata
a_r = 0.16

#Beregninger
aoi = np.arange(0, 91)
tilts = np.arange(0, 91)

iam_beam = martin_ruiz(aoi, a_r=a_r)
iam_sky, iam_gnd = zip(*[martin_ruiz_diffuse(t, a_r=a_r) for t in tilts])

#Plot
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
})

c = '#2c3e50'
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

ax1.plot(aoi, iam_beam, color=c)
ax1.set_xlabel(r'Innfallsvinkel, AOI ($\degree$)')
ax1.set_ylabel('IAM')
ax1.set_title(r'$IAM_{beam}$')
ax1.set_ylim([0, 1.05])
ax1.set_xlim([0, 90])

ax2.plot(tilts, iam_sky, color=c)
ax2.set_xlabel(r'Helningsvinkel, $\beta$ ($\degree$)')
ax2.set_ylabel('IAM')
ax2.set_title(r'$IAM_{sky}$')
ax2.set_ylim([0.90, 1.0])
ax2.set_xlim([0, 90])

ax3.plot(tilts, iam_gnd, color=c)
ax3.set_xlabel(r'Helningsvinkel, $\beta$ ($\degree$)')
ax3.set_ylabel('IAM')
ax3.set_title(r'$IAM_{ground}$')
ax3.set_ylim([0, 1.05])
ax3.set_xlim([0, 90])

plt.tight_layout()
plt.savefig('IAM_alle.png', dpi=300, bbox_inches='tight')
plt.show()