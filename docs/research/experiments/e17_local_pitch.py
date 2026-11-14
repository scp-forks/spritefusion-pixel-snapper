"""E17: is the lattice the same everywhere? Run the E15 pitch spectrum per tile (4 x 4).
Per tile: pitch = largest effective pitch scoring within 85% of that tile's best, and the best score.
A single-lattice image should print the same pitch in every textured tile; a mixed one will not.
Usage: e17_local_pitch.py [image names...]"""
import sys, numpy as np
from lab import *
from e15_pitch_spectrum import spectrum
names = sys.argv[1:] or ['colliseum', 'wolf', 'ogtrail', 'isometric-city-with-tiny-cars']
ps = np.arange(2.0, 16.01, 0.25); T = 4
for name in names:
    olab = srgb_to_oklab(load(R + name + '.png')); h, w = olab.shape[:2]
    print(f'\n{name}: per-tile "pitch/score" ({T}x{T} tiles of {w//T}x{h//T}); -- = tile too flat to say')
    for ty in range(T):
        row = []
        for tx in range(T):
            t = olab[ty * h // T:(ty + 1) * h // T, tx * w // T:(tx + 1) * w // T]
            if local_range(t).mean() < 0.004: row.append('   --    '); continue
            sp = spectrum(t, ps); best = sp[:, 2].max()
            row.append(f'{sp[sp[:, 2] >= 0.85 * best, 0].max():5.1f}/{best:.2f}')
        print('   ' + '  '.join(row), flush=True)
