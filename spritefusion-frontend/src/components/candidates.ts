import type { ResultSet } from '../utils/result-cache';
import { CLEAR_LATTICE_SCORE, NO_LATTICE_SCORE } from '../settings';

export interface CandidatesCallbacks {
  /** The user picked the result at this index of the current set. */
  onSelect: (index: number) => void;
  /** The user asked for a grid this many cells wide; 0 clears the request. */
  onTargetWidth: (width: number) => void;
}

function latticeText(score: number): { text: string; level: string } {
  if (score >= CLEAR_LATTICE_SCORE) {
    return { text: 'Clear pixel grid found', level: 'clear' };
  }
  if (score >= NO_LATTICE_SCORE) {
    return {
      text: 'Weak or mixed pixel grid: compare the sizes',
      level: 'weak',
    };
  }
  return {
    text: 'No pixel grid found: these sizes are plain reductions',
    level: 'none',
  };
}

function sizeNote(gridScale: number, isLegacy: boolean, typed: boolean): string {
  if (typed) return 'your size';
  if (isLegacy || Math.abs(gridScale - 1) < 1e-9) return 'detected';
  return `${Number(gridScale.toFixed(1))}x finer`;
}

/**
 * The audition strip: every size the engine produced for the current settings, as
 * thumbnails. The engine cannot know which size is right; the user picks.
 */
export function createCandidates(
  container: HTMLElement,
  callbacks: CandidatesCallbacks,
) {
  container.innerHTML = `
    <div class="candidates-head">
      <span class="candidates-title">Sizes</span>
      <span class="candidates-lattice"></span>
      <form class="candidates-width" novalidate>
        <label>
          Add width
          <input type="number" class="candidates-width-input" min="1" max="5000" step="1"
            inputmode="numeric" placeholder="e.g. 128"
            title="Ask for a grid this many cells wide. On an image with a pixel grid the result follows the grid and may land a few cells off." />
        </label>
        <button type="submit" class="candidates-width-btn">Add</button>
      </form>
    </div>
    <div class="candidates-strip" role="listbox" aria-label="Result sizes"></div>
  `;

  const strip = container.querySelector<HTMLDivElement>('.candidates-strip')!;
  const lattice = container.querySelector<HTMLSpanElement>('.candidates-lattice')!;
  const form = container.querySelector<HTMLFormElement>('.candidates-width')!;
  const input = container.querySelector<HTMLInputElement>('.candidates-width-input')!;
  const addBtn = container.querySelector<HTMLButtonElement>('.candidates-width-btn')!;

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const raw = input.value.trim();
    if (raw === '') {
      callbacks.onTargetWidth(0);
      return;
    }
    const width = Number(raw);
    if (!Number.isInteger(width) || width < 1 || width > 5000) {
      input.setCustomValidity('Enter a whole number of cells, 1 to 5000');
      input.reportValidity();
      return;
    }
    input.setCustomValidity('');
    callbacks.onTargetWidth(width);
  });
  input.addEventListener('input', () => input.setCustomValidity(''));

  strip.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('.candidate');
    if (btn) callbacks.onSelect(Number(btn.dataset.index));
  });

  return {
    /** Show `set`, with the result at `selected` marked. Hidden when there is no choice. */
    update(set: ResultSet, selected: number) {
      const hasGrid = set.results.some((r) => r.cols > 0);
      container.classList.toggle('hidden', !hasGrid);
      if (!hasGrid) return;

      const { text, level } = latticeText(set.latticeScore);
      lattice.textContent = text;
      lattice.dataset.level = level;
      lattice.title = `Lattice score ${set.latticeScore.toFixed(2)}: how much better the fitted grid explains the image than grids that cannot be aligned with it.`;

      // The typed width is the rung nearest to it, if the engine added one.
      const target = set.settings.targetWidth;
      let typedIndex = -1;
      if (target > 0) {
        let best = Infinity;
        set.results.forEach((r, i) => {
          const d = Math.abs(r.cols - target);
          if (d < best) {
            best = d;
            typedIndex = i;
          }
        });
      }
      if (document.activeElement !== input) {
        input.value = target > 0 ? String(target) : '';
      }

      strip.replaceChildren(
        ...set.results.map((r, i) => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'candidate';
          btn.dataset.index = String(i);
          btn.setAttribute('role', 'option');
          btn.setAttribute('aria-selected', String(i === selected));
          if (i === selected) btn.classList.add('selected');

          const img = document.createElement('img');
          img.src = r.url;
          img.alt = '';
          img.className = 'candidate-thumb';

          const size = document.createElement('span');
          size.className = 'candidate-size';
          size.textContent = `${r.cols} x ${r.rows}`;

          const note = document.createElement('span');
          note.className = 'candidate-note';
          note.textContent = sizeNote(r.gridScale, r.isLegacy, i === typedIndex);

          btn.append(img, size, note);
          return btn;
        }),
      );
    },

    select(selected: number) {
      strip.querySelectorAll<HTMLButtonElement>('.candidate').forEach((btn, i) => {
        btn.classList.toggle('selected', i === selected);
        btn.setAttribute('aria-selected', String(i === selected));
      });
    },

    setProcessing(processing: boolean) {
      input.disabled = processing;
      addBtn.disabled = processing;
    },
  };
}
