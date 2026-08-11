/* ════════════════════════════════════════════════════════════════════════
   ATLAS — the interface

   Speed here is mostly about what does NOT happen. There is no framework, no
   build step and no virtual DOM: a search response becomes a document fragment
   in one pass and is swapped in once. Four things do the actual work:

     · a client-side result cache, so re-running a query or coming back to one
       repaints from memory with no request at all
     · prefetch on hover and on focus, so the file is usually already resident
       by the time the click lands
     · posters loaded through an IntersectionObserver, so a 500-row library
       costs the bandwidth of a screenful
     · the player is a persistent element that is never torn down — switching
       videos re-points a `src`, which keeps the decoder warm

   Nothing in this file knows the names of database tables. Sections, columns
   and filters all come from what the API reports about the data it found.
   ════════════════════════════════════════════════════════════════════════ */

'use strict';

/* ── small helpers ─────────────────────────────────────────────────────── */
const $  = (id) => document.getElementById(id);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

function h(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function')
      node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : String(v));
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.appendChild(typeof kid === 'string'
      ? document.createTextNode(kid) : kid);
  }
  return node;
}

/* Query words marked inside a passage. Built as DOM nodes rather than an HTML
   string on purpose: passage text comes from transcripts and OCR and contains
   whatever the speaker said, so it must never be parsed as markup. */
function marked(text, query) {
  const frag = document.createDocumentFragment();
  const raw = String(text === null || text === undefined ? '' : text);
  const words = Array.from(new Set(
    (query || '').toLowerCase().match(/[a-z0-9']{3,}/g) || []));
  if (!words.length || !raw) {
    frag.appendChild(document.createTextNode(raw));
    return frag;
  }
  const re = new RegExp('(' + words
    .map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi');
  let last = 0, m;
  while ((m = re.exec(raw)) !== null) {
    if (m.index > last) frag.appendChild(document.createTextNode(raw.slice(last, m.index)));
    frag.appendChild(h('mark', { text: m[0] }));
    last = m.index + m[0].length;
    if (!m[0].length) re.lastIndex++;
  }
  if (last < raw.length) frag.appendChild(document.createTextNode(raw.slice(last)));
  return frag;
}

function timecode(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return '—';
  const s = Math.max(0, Math.round(Number(sec)));
  const m = Math.floor(s / 60), r = s % 60;
  if (m >= 60) {
    const hh = Math.floor(m / 60);
    return `${hh}:${String(m % 60).padStart(2, '0')}:${String(r).padStart(2, '0')}`;
  }
  return `${m}:${String(r).padStart(2, '0')}`;
}

const fmtInt = (n) => (Number(n) || 0).toLocaleString('en-US');

function fmtBytes(b) {
  b = Number(b) || 0;
  if (b >= 1073741824) return (b / 1073741824).toFixed(2) + ' GB';
  if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
  if (b >= 1024) return (b / 1024).toFixed(0) + ' KB';
  return b + ' B';
}

function fmtWhen(v) {
  if (!v) return '';
  let t = Number(v);
  if (!isFinite(t)) { const d = new Date(v); return isNaN(d) ? '' : d.toLocaleDateString(); }
  if (t > 1e11) t = t / 1000;                 // milliseconds, not seconds
  const d = new Date(t * 1000);
  return isNaN(d) ? '' : d.toLocaleDateString('en-US',
    { year: 'numeric', month: 'short', day: 'numeric' });
}

const SOURCE_COLOR = {
  narrative: 'var(--s-narrative)', speech: 'var(--s-speech)',
  visual: 'var(--s-visual)', ocr: 'var(--s-ocr)',
  caption: 'var(--s-caption)', meta: 'var(--s-meta)',
};
const SOURCE_LABEL = {
  narrative: 'narrative', speech: 'speech', visual: 'objects seen',
  ocr: 'on-screen text', caption: 'caption', meta: 'metadata',
};
const color = (src) => SOURCE_COLOR[src] || 'var(--s-meta)';

let toastTimer = 0;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 3200);
}

/* ── transport ─────────────────────────────────────────────────────────── */

/* Where this Atlas is mounted.
 *
 * Atlas runs two ways and the interface must not care which: standalone on its
 * own port (`atlas_boot.py`, everything at `/`), or mounted inside the main
 * VIOS server at `/atlas` so the whole system is one tunnel and one URL.
 *
 * Derived from this script's own resolved URL rather than from
 * `location.pathname`, because the pathname is not reliable here — the tab
 * router writes to `location.hash`, deep links carry a hash, and a trailing
 * slash may or may not be present. The script tag's `src` is resolved to an
 * absolute URL by the browser and is exactly one of `/atlas.js` or
 * `/atlas/atlas.js`, which makes this a fact rather than an inference.
 *
 * Yields '' standalone and '/atlas' mounted, so `BASE + '/api/…'` is right in
 * both and no caller has to know.
 */
const BASE = (() => {
  const tag = document.currentScript
    || [...document.querySelectorAll('script')].find(s => /atlas\.js(\?|$)/.test(s.src));
  try { return new URL(tag.src).pathname.replace(/\/atlas\.js.*$/, ''); }
  catch { return ''; }
})();

/* Absolute-from-root paths, rewritten to sit under BASE. Every call site keeps
 * writing '/api/…' and this is the single place that knows about mounting. */
const U = (path) => (path.startsWith('/') ? BASE + path : path);

async function api(path, opts) {
  const res = await fetch(U(path), opts);
  const body = await res.text();
  let data;
  try { data = body ? JSON.parse(body) : {}; }
  catch { throw new Error(`${res.status}: ${body.slice(0, 160)}`); }
  if (!res.ok && data.note) throw new Error(data.note);
  return data;
}

/* ── state ─────────────────────────────────────────────────────────────── */
const S = {
  tab: 'home',
  query: '',
  results: [],
  resultTotal: 0,
  resultMeta: null,
  sourceFilter: new Set(),
  searchCache: new Map(),      // query|offset|filter → response
  // How the results are shaped, ordered and narrowed. `view` and `density` are
  // pure presentation and never refetch; `sort` and `narrow` are questions for
  // the server, because ordering and filtering happen over the whole matched
  // pool rather than the page the browser happens to hold.
  view: 'list',
  density: 3,
  sort: 'relevance',
  narrow: { creator: '', category: '', min_dur: '', max_dur: '', min_hits: '' },
  facets: null,
  status: null,
  video: null,                 // the open video's search-result shape
  record: null,                // /api/video payload for the open video
  lib: { offset: 0, rows: [], total: 0, creator: '', category: '', inside: 0, q: '' },
  home: { latest: null, counted: {} },   // the landing page's own fetch + count-up
  browse: { table: '', offset: 0, q: '' },
  prefetched: new Set(),
  suggestIndex: -1,
  suggestItems: [],
};

const SEARCH_LIMIT = 24;
const LIB_LIMIT = 40;

/* ════════════════════════════════════════════════════════════════════════
   ROUTING
   ════════════════════════════════════════════════════════════════════════ */
/* Every section that owns a URL. A tab missing from this list is unreachable
 * by link and silently falls back to the landing page, which is how the Maps
 * tab could be opened by click but never by refresh. */
const TABS = ['home', 'search', 'library', 'graph', 'maps', 'data', 'sources'];

function readHash() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [tab, qs] = raw.split('?');
  return {
    tab: TABS.includes(tab) ? tab : 'home',
    params: new URLSearchParams(qs || ''),
  };
}

function writeHash(tab, params) {
  const qs = params && params.toString();
  const next = `#/${tab}${qs ? '?' + qs : ''}`;
  if (location.hash !== next) history.replaceState(null, '', next);
}

function showTab(tab, { push = true } = {}) {
  S.tab = tab;
  previewStop();   // a preview left running under a hidden tab keeps streaming
  $$('.tabs button').forEach(b =>
    b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
  $$('.view').forEach(v => { v.hidden = v.dataset.view !== tab; });
  $('left').scrollTop = 0;
  if (push) {
    const p = new URLSearchParams();
    if (tab === 'search' && S.query) p.set('q', S.query);
    if (S.video) p.set('v', S.video.video_key);
    writeHash(tab, p);
  }
  if (tab === 'library' && !S.lib.rows.length) loadLibrary(true);
  // The canvas has no size while its view is hidden, so the graph can only be
  // measured after the switch — hence the boot call here rather than at start.
  if (tab === 'graph') graphBoot(false);
  if (tab === 'maps') mapsBoot(false);
  if (tab === 'data' && !$('schema').childElementCount) loadSchema();
  if (tab === 'sources') loadSources();
  if (tab === 'home') homeBoot();
}

/* ════════════════════════════════════════════════════════════════════════
   SEARCH
   ════════════════════════════════════════════════════════════════════════ */
/* Everything the server saw goes in the key. `view` and `density` deliberately
 * do not: they reshape rows the browser already has, so including them would
 * throw away a good response to answer a question it already answers. */
function cacheKey(q, offset) {
  const f = Array.from(S.sourceFilter).sort().join(',');
  const n = S.narrow;
  return [q, offset, f, S.sort, n.creator, n.category,
          n.min_dur, n.max_dur, n.min_hits].join('|');
}

/* The narrowing params, as the server wants them — empty strings dropped so a
 * cleared field is absent rather than sent as an explicit "no length". */
function narrowParams(p) {
  const n = S.narrow;
  if (n.creator) p.set('creator', n.creator);
  if (n.category) p.set('category', n.category);
  for (const k of ['min_dur', 'max_dur', 'min_hits']) {
    const v = String(n[k] ?? '').trim();
    if (v !== '' && isFinite(Number(v))) p.set(k, v);
  }
  return p;
}

function narrowActive() {
  const n = S.narrow;
  return Boolean(n.creator || n.category ||
    String(n.min_dur).trim() || String(n.max_dur).trim() || String(n.min_hits).trim());
}

async function runSearch(query, { append = false } = {}) {
  query = (query || '').trim();
  if (!query) { showOpening(); return; }
  S.query = query;
  $('q').value = query;
  const offset = append ? S.results.length : 0;

  const key = cacheKey(query, offset);
  const cached = S.searchCache.get(key);
  if (cached) { applySearch(cached, append); return; }

  if (!append) paintSearchSkeleton();
  const p = new URLSearchParams({
    q: query, limit: String(SEARCH_LIMIT), offset: String(offset),
  });
  if (S.sourceFilter.size) p.set('source', Array.from(S.sourceFilter).join(','));
  if (S.sort && S.sort !== 'relevance') p.set('sort', S.sort);
  narrowParams(p);

  try {
    const data = await api('/api/search?' + p.toString());
    S.searchCache.set(key, data);
    if (S.searchCache.size > 60) S.searchCache.delete(S.searchCache.keys().next().value);
    applySearch(data, append);
  } catch (e) {
    $('results').hidden = true;
    $('opening').hidden = true;
    showEmpty('Search failed', String(e.message || e));
  }
}

function applySearch(data, append) {
  S.resultMeta = data;
  S.results = append ? S.results.concat(data.results || []) : (data.results || []);
  S.resultTotal = data.total || 0;

  $('opening').hidden = true;
  $('emptySearch').hidden = true;

  if (!S.results.length) {
    // "The query found nothing" and "your filters removed everything" need
    // different answers, and only the second one is undone by a button.
    if (data.matched && narrowActive()) {
      $('results').hidden = false;
      renderCount(data);
      renderSourceFilters(data);
      renderNarrow(data);
      $('cards').textContent = '';
      $('more').hidden = true;
      $('cards').appendChild(h('li', { class: 'narrowed-out' },
        h('p', { text: `“${S.query}” matched ${fmtInt(data.matched)} video` +
                       `${data.matched === 1 ? '' : 's'}, but none of them get past the filters.` }),
        h('button', { class: 'btn', onclick: clearNarrow }, 'Clear the filters')));
      return;
    }
    $('results').hidden = true;
    showEmpty('Nothing matched “' + S.query + '”',
      data.dense
        ? 'Try fewer words, or describe what is happening rather than naming it. Search covers narratives, speech, objects and on-screen text.'
        : 'The meaning index is still building — right now search only matches words that literally appear. Semantic matches will start working on their own.');
    return;
  }

  $('results').hidden = false;
  renderCount(data);
  renderSourceFilters(data);
  renderNarrow(data);
  renderCards(S.results, append);

  const shown = S.results.length;
  $('more').hidden = shown >= S.resultTotal;
  $('moreBtn').textContent = `Load ${Math.min(SEARCH_LIMIT, S.resultTotal - shown)} more`;

  const p = new URLSearchParams({ q: S.query });
  if (S.video) p.set('v', S.video.video_key);
  writeHash('search', p);
}

function renderCount(data) {
  const bits = [
    `<b>${fmtInt(data.total)}</b> video${data.total === 1 ? '' : 's'}`,
    `<span class="lat">${data.cached ? 'cached' : data.took_ms + ' ms'}</span>`,
  ];
  // `matched` is the pool before filtering. Saying "18 of 340" keeps the blame
  // where it belongs — on the filters, not on the query.
  if (data.matched && data.total < data.matched)
    bits.splice(1, 0, `<span class="of">of ${fmtInt(data.matched)} matched</span>`);
  if (data.mode === 'hybrid') bits.push('meaning + words');
  else if (data.mode === 'lexical') bits.push('words only');
  else if (data.mode === 'dense') bits.push('meaning only');
  if (data.sort && data.sort !== 'relevance')
    bits.push(`<span class="by">by ${SORT_LABEL[data.sort] || data.sort}</span>`);
  $('resultsCount').innerHTML = bits.join(' <span class="sep">·</span> ');
  // What these results have in common is often the more interesting question,
  // and it is one click away rather than a separate search.
  $('resultsCount').appendChild(h('button', {
    class: 'plot-link', onclick: graphFromResults,
    title: 'Plot these results and what they share on the graph',
  }, 'see the graph'));
}

const SORT_LABEL = {
  relevance: 'best match', matches: 'most moments', recent: 'newest',
  oldest: 'oldest', longest: 'longest', shortest: 'shortest', liked: 'most liked',
};

/* ── narrowing ─────────────────────────────────────────────────────────────
 * The creator and category chips are built from the response, not from the
 * archive-wide facets the library uses: offering "Ana — 40" when this query
 * reached none of her reels is an invitation to an empty page. */
function renderNarrow(data) {
  const facets = data.facets || {};
  const rows = [
    ['narrowCreators', 'creator', facets.creators],
    ['narrowCategories', 'category', facets.categories],
  ];

  for (const [id, field, list] of rows) {
    const box = $(id);
    box.textContent = '';
    const active = S.narrow[field];
    const items = (list || []).slice();
    // A chosen value that this response no longer lists still needs its own
    // chip, or the only way to switch it off is Clear all.
    if (active && !items.some(i => i.value === active))
      items.unshift({ value: active, count: data.total || 0 });
    if (!items.length) { box.hidden = true; continue; }
    box.hidden = false;
    box.appendChild(h('span', { class: 'narrow-label',
                                text: field === 'creator' ? 'creator' : 'category' }));
    for (const item of items) {
      const on = active === item.value;
      box.appendChild(h('button', {
        class: 'facet', 'aria-pressed': String(on),
        onclick: () => {
          S.narrow[field] = on ? '' : item.value;
          runSearch(S.query);
        },
      }, item.value, h('span', { class: 'n', text: fmtInt(item.count) })));
    }
  }

  for (const [id, key] of [['narrowMinDur', 'min_dur'], ['narrowMaxDur', 'max_dur'],
                           ['narrowMinHits', 'min_hits']]) {
    const el = $(id);
    if (el && document.activeElement !== el) el.value = S.narrow[key] ?? '';
  }

  const on = narrowActive();
  $('searchNarrowBtn').setAttribute('aria-pressed', String(on));
  $('searchNarrowBtn').textContent = on ? 'narrowing' : 'narrow';
  // Opening the panel is a choice; an active filter forces it open so nothing
  // is hiding results invisibly.
  if (on) setNarrowOpen(true);

  const note = $('narrowNote');
  note.textContent = on && data.matched
    ? `${fmtInt(data.total)} of ${fmtInt(data.matched)} matched videos shown. ` +
      'Filtering happens after ranking, so every result keeps the score it would have had unfiltered.'
    : '';
}

function setNarrowOpen(open) {
  $('searchNarrow').hidden = !open;
  $('searchNarrowBtn').setAttribute('aria-expanded', String(open));
}

function clearNarrow() {
  S.narrow = { creator: '', category: '', min_dur: '', max_dur: '', min_hits: '' };
  for (const id of ['narrowMinDur', 'narrowMaxDur', 'narrowMinHits'])
    if ($(id)) $(id).value = '';
  runSearch(S.query);
}

function renderSourceFilters(data) {
  const present = new Map();
  for (const r of S.results)
    for (const m of r.moments || [])
      present.set(m.source, (present.get(m.source) || 0) + 1);

  const box = $('sourceFilters');
  box.textContent = '';
  if (present.size < 2 && !S.sourceFilter.size) return;

  const order = ['narrative', 'speech', 'visual', 'ocr', 'caption', 'meta'];
  const keys = Array.from(present.keys())
    .sort((a, b) => order.indexOf(a) - order.indexOf(b));

  for (const src of keys) {
    const on = S.sourceFilter.has(src);
    box.appendChild(h('button', {
      class: 'chip-filter', 'aria-pressed': String(on),
      style: on ? `color:${color(src)}` : '',
      title: `Only results found in ${SOURCE_LABEL[src] || src}`,
      onclick: () => {
        if (on) S.sourceFilter.delete(src); else S.sourceFilter.add(src);
        runSearch(S.query);
      },
    }, h('i', { class: 'dot', style: `background:${color(src)}` }),
       SOURCE_LABEL[src] || src));
  }
  if (S.sourceFilter.size) {
    box.appendChild(h('button', {
      class: 'chip-filter', onclick: () => { S.sourceFilter.clear(); runSearch(S.query); },
    }, 'clear'));
  }
}

/* the moment ribbon — the one element the whole interface is built around */
function ribbon(video, { large = false, onSeek = null } = {}) {
  const span = Number(video.duration) || Math.max(
    2, ...(video.moments || []).map(m => Number(m.t_end || m.t_start || 0) + 2));
  const bar = h('div', {
    class: 'ribbon' + (large ? ' ribbon-lg' : ''),
    title: span ? `${timecode(span)} of video` : '',
  });

  for (const m of video.moments || []) {
    if (m.t_start === null || m.t_start === undefined) continue;
    const start = Math.max(0, Number(m.t_start));
    const end = Number(m.t_end);
    const dur = isFinite(end) && end > start ? end - start : 1.4;
    bar.appendChild(h('i', {
      class: 'seg',
      style: `left:${(start / span * 100).toFixed(3)}%;` +
             `width:${Math.max(1.2, dur / span * 100).toFixed(3)}%;` +
             `background:${color(m.source)}`,
      title: `${timecode(start)} · ${SOURCE_LABEL[m.source] || m.source}`,
    }));
  }

  // Moments with no timestamp (a caption, a creator name) belong to the whole
  // reel, so they are drawn as a faint full-width wash rather than dropped.
  const untimed = (video.moments || []).filter(
    m => m.t_start === null || m.t_start === undefined);
  if (untimed.length && !bar.childElementCount) {
    bar.appendChild(h('i', {
      class: 'seg',
      style: `left:0;width:100%;opacity:.30;background:${color(untimed[0].source)}`,
      title: 'matches the whole video',
    }));
  }

  if (onSeek) {
    bar.addEventListener('click', (ev) => {
      const box = bar.getBoundingClientRect();
      onSeek(Math.max(0, Math.min(1, (ev.clientX - box.left) / box.width)) * span);
    });
  }
  return bar;
}

function posterImg(video, at, cls) {
  const key = video.video_key;
  const t = (at === null || at === undefined) ? '' : `?t=${Math.max(0, at).toFixed(1)}`;
  const img = h('img', { alt: '', loading: 'lazy', 'data-src': U(`/api/poster/${key}${t}`) });
  const wrap = h('div', { class: cls },
    h('span', { class: 'noshot', text: key }), img);
  img.addEventListener('error', () => { img.remove(); });
  posterWatcher.observe(img);
  return wrap;
}

// Posters are ffmpeg calls on the server, so they are only requested for
// thumbnails that actually reach the viewport.
const posterWatcher = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    const img = e.target;
    posterWatcher.unobserve(img);
    if (img.dataset.src) { img.src = img.dataset.src; delete img.dataset.src; }
  }
}, { rootMargin: '400px 0px' });

/* ── hover preview ─────────────────────────────────────────────────────────
 * Hovering a result plays the found moment, not the start of the reel: two
 * seconds from `/api/clip`, looped, silent. There is exactly one <video> for
 * the whole page and it is moved into whichever thumbnail is under the
 * pointer — a grid of forty tiles must not become forty media downloads.
 * ------------------------------------------------------------------------- */
const PV = {
  el: null,
  host: null,        // the .card-shot / .tile-shot currently holding it
  timer: 0,
  key: '',
  none: new Set(),   // keys the server has already answered 204 for
};

function previewEl() {
  if (PV.el) return PV.el;
  const v = document.createElement('video');
  v.className = 'hover-clip';
  v.muted = true; v.loop = true; v.playsInline = true;
  v.preload = 'auto';
  v.setAttribute('aria-hidden', 'true');
  // A 204 or an undecodable clip must leave the poster exactly as it was,
  // so a video with no clip index simply behaves like one without a preview.
  v.addEventListener('error', () => { PV.none.add(PV.key); previewStop(); });
  PV.el = v;
  return v;
}

function previewStop() {
  clearTimeout(PV.timer);
  PV.timer = 0;
  const v = PV.el;
  if (!v) return;
  v.pause();
  v.removeAttribute('src');
  v.load();                       // releases the connection, not just the frame
  if (v.parentNode) v.parentNode.removeChild(v);
  if (PV.host) PV.host.classList.remove('previewing');
  PV.host = null;
  PV.key = '';
}

function previewStart(host, key, t) {
  if (PV.none.has(key)) return;
  const v = previewEl();
  PV.host = host;
  PV.key = key;
  host.classList.add('previewing');
  host.appendChild(v);
  v.src = U(`/api/clip/${key}?t=${Math.max(0, t || 0).toFixed(1)}`);
  v.play().catch(() => { /* autoplay refused, or the clip never arrived */ });
}

/* Attach to a thumbnail. The delay is what keeps a pointer travelling across
 * a grid from firing a request per tile it crosses. */
function previewOn(host, key, t, delay = 320) {
  if (!window.matchMedia || !window.matchMedia('(hover: hover)').matches) return host;
  host.addEventListener('pointerenter', () => {
    clearTimeout(PV.timer);
    PV.timer = setTimeout(() => previewStart(host, key, t), delay);
  });
  host.addEventListener('pointerleave', () => {
    if (PV.host === host) previewStop(); else clearTimeout(PV.timer);
  });
  return host;
}

/* ── layout ────────────────────────────────────────────────────────────────
 * Rows and grid are one DOM under two stylesheets, not two render paths. The
 * card already holds everything either layout needs, so switching is a class
 * change — instant, and it never asks the server for rows it already has. */
const SEARCH_DENSITY = { 1: 2, 2: 3, 3: 0, 4: 5, 5: 7 };   // 0 = the CSS default

function setSearchView(view, { remember = true } = {}) {
  S.view = view === 'grid' ? 'grid' : 'list';
  $('cards').dataset.view = S.view;
  $$('.rc-view').forEach(b => b.classList.toggle('on', b.dataset.view === S.view));
  // Density only means anything when there is more than one result per row.
  $('searchDensityWrap').hidden = S.view !== 'grid';
  applySearchDensity(S.density);
  previewStop();   // a preview mid-play under a relaid-out card keeps streaming
  if (remember) { try { localStorage.setItem('atlas.searchView', S.view); } catch { /* private mode */ } }
}

function applySearchDensity(step, { remember = false } = {}) {
  S.density = step;
  const n = SEARCH_DENSITY[step] || 0;
  const list = $('cards');
  if (n && S.view === 'grid') list.style.setProperty('--card-cols', String(n));
  else list.style.removeProperty('--card-cols');
  if (remember) { try { localStorage.setItem('atlas.searchDensity', String(step)); } catch { /* private mode */ } }
}

function renderCards(results, append) {
  const list = $('cards');
  list.dataset.view = S.view;
  if (!append) { previewStop(); list.textContent = ''; }
  const frag = document.createDocumentFragment();

  for (const r of results.slice(append ? list.childElementCount : 0)) {
    const best = r.best || (r.moments || [])[0] || {};
    const at = (best.t_start === null || best.t_start === undefined)
      ? null : Number(best.t_start);

    const shot = posterImg(r, at, 'card-shot');
    if (r.duration) shot.appendChild(h('span', { class: 'dur', text: timecode(r.duration) }));
    if (r.has_file) shot.appendChild(h('i', { class: 'cached', title: 'already on this machine' }));
    // Hovering plays the moment that made this a result — not the reel's
    // opening frame, which is the one part of a video nothing ever matched on.
    if (at !== null) previewOn(shot, r.video_key, at);

    const line = h('div', { class: 'card-line' });
    if (r.creator) line.appendChild(h('span', { class: 'who', text: r.creator }));
    if (r.category) line.append(h('span', { class: 'sep', text: '·' }),
                                document.createTextNode(r.category));
    line.append(h('span', { class: 'sep', text: '·' }),
                document.createTextNode(
                  `${r.hit_count} match${r.hit_count === 1 ? '' : 'es'} of ${fmtInt(r.moment_count)}`));
    if (r.created_at) line.append(h('span', { class: 'sep', text: '·' }),
                                  document.createTextNode(fmtWhen(r.created_at)));

    const hits = h('div', { class: 'card-hits' });
    const top = (r.moments || []).slice()
      .sort((a, b) => (b.score || 0) - (a.score || 0)).slice(0, 3);
    for (const m of top) {
      hits.appendChild(h('div', {
        class: 'hit',
        onclick: (ev) => { ev.stopPropagation(); openVideo(r, m.t_start); },
      },
        h('span', { class: 't', text: m.t_start === null || m.t_start === undefined
          ? '—' : timecode(m.t_start) }),
        h('span', { class: 'rail', style: `background:${color(m.source)}` }),
        h('span', { class: 'txt' }, marked(m.text, S.query))));
    }

    const card = h('li', {
      class: 'card', 'data-key': r.video_key, tabindex: '0',
      onclick: () => openVideo(r, at),
      onkeydown: (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openVideo(r, at); } },
      onpointerenter: () => prefetch([r.video_key]),
      onfocus: () => prefetch([r.video_key]),
    },
      h('div', { class: 'card-rank', text: String(r.rank).padStart(2, '0') }),
      shot,
      h('div', { class: 'card-body' },
        h('div', { class: 'card-title', text: r.title }),
        line,
        ribbon(r, { onSeek: (t) => openVideo(r, t) }),
        hits,
        r.hit_count > 3 ? h('div', { class: 'card-more',
          text: `+${r.hit_count - 3} more moment${r.hit_count - 3 === 1 ? '' : 's'}` }) : null));

    frag.appendChild(card);
  }
  list.appendChild(frag);
  markActiveCard();
}

function paintSearchSkeleton() {
  $('opening').hidden = true;
  $('emptySearch').hidden = true;
  $('results').hidden = false;
  $('resultsCount').innerHTML = '<span class="lat">searching…</span>';
  $('sourceFilters').textContent = '';
  $('more').hidden = true;
  const list = $('cards');
  list.textContent = '';
  for (let i = 0; i < 4; i++) {
    list.appendChild(h('li', { class: 'card' },
      h('div', {}), h('div', { class: 'skeleton', style: 'aspect-ratio:9/13' }),
      h('div', { class: 'card-body' },
        h('div', { class: 'skeleton', style: 'height:18px;width:65%' }),
        h('div', { class: 'skeleton', style: 'height:12px;width:35%' }),
        h('div', { class: 'skeleton', style: 'height:16px' }),
        h('div', { class: 'skeleton', style: 'height:44px' }))));
  }
}

function showEmpty(title, body) {
  const box = $('emptySearch');
  box.textContent = '';
  box.appendChild(h('h3', { text: title }));
  box.appendChild(h('p', { text: body }));
  box.hidden = false;
}

function showOpening() {
  S.query = '';
  S.results = [];
  $('results').hidden = true;
  $('emptySearch').hidden = true;
  // The narrowing panel belongs to a result set. With no results it is a set of
  // controls governing nothing, so it goes away with them.
  setNarrowOpen(false);
  $('opening').hidden = false;
  writeHash('search', new URLSearchParams(S.video ? { v: S.video.video_key } : {}));
}

/* ── type-ahead ───────────────────────────────────────────────────────── */
let suggestTimer = 0;
function scheduleSuggest(value) {
  clearTimeout(suggestTimer);
  if (value.trim().length < 2) { closeSuggest(); return; }
  suggestTimer = setTimeout(async () => {
    try {
      const data = await api('/api/suggest?q=' + encodeURIComponent(value.trim()));
      renderSuggest(data.suggestions || []);
    } catch { closeSuggest(); }
  }, 110);
}

function renderSuggest(items) {
  const box = $('suggest');
  box.textContent = '';
  S.suggestItems = items;
  S.suggestIndex = -1;
  if (!items.length) { box.hidden = true; return; }
  items.forEach((s, i) => {
    box.appendChild(h('button', {
      type: 'button', role: 'option', 'aria-selected': 'false', 'data-i': i,
      onclick: () => { closeSuggest(); runSearch(s.text); },
    }, h('span', { text: s.text }),
       h('span', { class: 'kind', text: s.kind === 'name' ? 'in library' : 'term' })));
  });
  box.hidden = false;
}

function closeSuggest() {
  $('suggest').hidden = true;
  S.suggestItems = [];
  S.suggestIndex = -1;
}

function moveSuggest(delta) {
  const buttons = $$('#suggest button');
  if (!buttons.length) return;
  S.suggestIndex = (S.suggestIndex + delta + buttons.length + 1) % (buttons.length + 1);
  buttons.forEach((b, i) => b.setAttribute('aria-selected', String(i === S.suggestIndex)));
  if (S.suggestIndex >= 0 && S.suggestIndex < buttons.length)
    $('q').value = S.suggestItems[S.suggestIndex].text;
}

/* ════════════════════════════════════════════════════════════════════════
   PLAYBACK
   ════════════════════════════════════════════════════════════════════════ */
function prefetch(keys) {
  const fresh = keys.filter(k => k && !S.prefetched.has(k));
  if (!fresh.length) return;
  fresh.forEach(k => S.prefetched.add(k));
  fetch(U('/api/prefetch?keys=' + encodeURIComponent(fresh.join(','))),
        { method: 'POST' }).catch(() => {});
}

let statePoll = 0;
let retryTimer = 0;
let busyGuard = 0;
/* The ordered timestamps of the open video's passages, and where the player
 * currently sits in that list. Navigation is by moment, not by scrubbing —
 * the whole point of the index is that the interesting instants are known. */
let momTimes = [];
let momAt = -1;
let momLoop = null;

function openVideo(video, at) {
  const key = video.video_key;
  const same = S.video && S.video.video_key === key;
  S.video = video;

  $('playerIdle').hidden = true;
  $('playerLive').hidden = false;
  $('player').dataset.open = 'true';
  markActiveCard();

  const p = new URLSearchParams();
  if (S.tab === 'search' && S.query) p.set('q', S.query);
  p.set('v', key);
  writeHash(S.tab, p);

  renderPlayerMeta(video);
  renderMoments(video, at);
  showPanel('moments');
  $('panel-similar').textContent = '';
  $('panel-similar').dataset.key = '';

  const vid = $('video');
  if (!same) {
    clearTimeout(retryTimer);
    busy(true, 'opening');
    // A media fragment makes the browser request the byte range around that
    // timestamp first, so a click on a moment 40 s in does not download the
    // 40 s before it.
    const frag = at ? `#t=${Math.max(0, at).toFixed(2)}` : '';
    vid.src = U(`/api/play/${encodeURIComponent(key)}${frag}`);
    vid.load();
    vid.play().catch(() => {});
    pollMediaState(key);
  } else if (at !== null && at !== undefined && isFinite(at)) {
    seekTo(at);
  }
  loadRecord(key);
}

/* Open a reel when all you have is its key — from a deep link, or from a cell
   in a table that turned out to be about a video. One fetch, so the player
   arrives with its moments and its record rather than a bare title. */
async function openVideoKey(key, at) {
  try {
    const data = await api(`/api/video/${encodeURIComponent(key)}`);
    const m = data.meta || {};
    openVideo({
      video_key: data.video_key, title: m.title || m.caption || data.video_key,
      caption: m.caption, creator: m.creator, category: m.category,
      duration: m.duration, width: m.width, height: m.height, likes: m.likes,
      created_at: m.created_at, msg_id: m.msg_id,
      moment_count: m.moment_count, hit_count: 0, moments: data.moments || [],
    }, at === undefined ? null : at);
  } catch (e) { toast('Could not open ' + key + ': ' + e.message); }
}

function seekTo(t) {
  const vid = $('video');
  const go = () => { try { vid.currentTime = Math.max(0, t); vid.play().catch(() => {}); } catch {} };
  if (vid.readyState >= 1) go();
  else vid.addEventListener('loadedmetadata', go, { once: true });
}

/* The overlay used to be cleared only by `loadeddata` or `playing`. When
 * neither ever fires — a codec the browser will not decode, a range request
 * the channel never answers — the word "opening" stayed on screen forever and
 * looked like the whole application had hung. So every busy state now carries
 * its own deadline and, when it expires, says what it is actually waiting for
 * and offers the two things a person can do about it. */
function busy(on, text, pct, opts) {
  const box = $('screenBusy');
  box.hidden = !on;
  if (text) $('busyText').textContent = text;
  $('busyBar').style.width = (pct === undefined ? (on ? 8 : 100) : pct) + '%';

  const act = $('busyAct');
  act.hidden = true;
  act.onclick = null;
  clearTimeout(busyGuard);
  if (!on) return;

  if (opts && opts.action) {
    act.hidden = false;
    act.textContent = opts.action.label;
    act.onclick = opts.action.run;
    return;
  }
  busyGuard = setTimeout(() => {
    if ($('screenBusy').hidden || !S.video) return;
    const vid = $('video');
    // Bytes are decoding: the overlay is simply stale, so drop it.
    if (vid.readyState >= 2 || (!vid.paused && vid.currentTime > 0)) {
      busy(false);
      return;
    }
    busy(true, 'still waiting on this file — it may be large, or the channel ' +
      'may not have answered', 0, { action: {
        label: 'Try again',
        run: () => {
          busy(true, 'opening');
          vid.load();
          vid.play().catch(() => {});
          if (S.video) pollMediaState(S.video.video_key);
        },
      } });
  }, (opts && opts.after) || 12000);
}

function pollMediaState(key) {
  clearInterval(statePoll);
  statePoll = setInterval(async () => {
    if (!S.video || S.video.video_key !== key) { clearInterval(statePoll); return; }
    try {
      const st = await api(`/api/media/${encodeURIComponent(key)}/state`);
      if (st.where === 'local' || st.where === 'cache' || st.status === 'ready') {
        clearInterval(statePoll);
        return;
      }
      // Streaming means bytes are already reaching the player. The video's own
      // events clear the overlay; showing a progress bar over a playing video
      // would be a lie about what it is waiting for.
      if (st.status === 'streaming') return;
      if (st.status === 'error') {
        clearInterval(statePoll);
        busy(true, st.note || 'could not fetch this video from the channel', 0,
          { action: { label: 'Try again', run: () => {
              const vid = $('video');
              busy(true, 'opening');
              vid.load(); vid.play().catch(() => {});
              pollMediaState(key);
            } } });
        return;
      }
      const pct = st.percent || (st.total ? (st.got / st.total) * 100 : 0);
      busy(true, st.status === 'downloading'
        ? `fetching from the channel — ${fmtBytes(st.got)}${st.total ? ' of ' + fmtBytes(st.total) : ''}`
        : 'queued behind another download', Math.max(4, pct));
    } catch { /* keep polling; a 503 here is expected while it downloads */ }
  }, 900);
}

function renderPlayerMeta(video) {
  const box = $('playerMeta');
  box.textContent = '';
  box.appendChild(h('div', { class: 'pm-title', text: video.title || video.video_key }));

  const line = h('div', { class: 'pm-line' });
  const bits = [];
  if (video.creator) bits.push(video.creator);
  if (video.duration) bits.push(timecode(video.duration));
  if (video.width && video.height) bits.push(`${video.width}×${video.height}`);
  if (video.likes) bits.push(`${fmtInt(video.likes)} likes`);
  if (video.created_at) bits.push(fmtWhen(video.created_at));
  bits.push(`msg ${video.msg_id || video.video_key}`);
  bits.forEach((b, i) => {
    if (i) line.appendChild(h('span', { class: 'sep', text: '·' }));
    line.appendChild(document.createTextNode(b));
  });
  box.appendChild(line);

  if (video.caption)
    box.appendChild(h('div', { class: 'pm-caption', text: video.caption }));

  const kinds = new Map();
  for (const m of video.moments || []) kinds.set(m.source, (kinds.get(m.source) || 0) + 1);
  if (kinds.size) {
    const tags = h('div', { class: 'pm-tags' });
    for (const [src, n] of kinds)
      tags.appendChild(h('span', { class: 'tag' },
        h('i', { class: 'dot', style: `background:${color(src)}` }),
        `${SOURCE_LABEL[src] || src} ${n}`));
    box.appendChild(tags);
  }
}

function renderMoments(video, at) {
  const block = $('playerRibbon').parentElement;
  const fresh = ribbon(video, { large: true, onSeek: seekTo });
  fresh.id = 'playerRibbon';
  fresh.setAttribute('role', 'slider');
  fresh.setAttribute('tabindex', '0');
  fresh.setAttribute('aria-label', 'Matching moments');
  fresh.appendChild(h('i', { class: 'playhead', id: 'playhead' }));
  block.replaceChild(fresh, $('playerRibbon'));
  $('tEnd').textContent = timecode(video.duration || 0);
  $('tNow').textContent = timecode(at || 0);

  const panel = $('panel-moments');
  panel.textContent = '';
  const list = (video.moments || []).slice().sort((a, b) =>
    (a.t_start === null ? -1 : a.t_start) - (b.t_start === null ? -1 : b.t_start));
  // Keep the ordered timestamps for moment navigation. The loop needs a live
  // video to know when to stop, so it is an interval, not a one-shot seek.
  momTimes = list.map(m => m.t_start).filter(t => t !== null && t !== undefined);
  momAt = -1;
  stopMomLoop();
  momStepTo(at !== null && at !== undefined ? at : (momTimes[0] ?? -1));
  if (!list.length) {
    panel.appendChild(h('p', { class: 'hint',
      text: 'No indexed passages for this video yet.' }));
    return;
  }
  for (const m of list) {
    panel.appendChild(h('div', {
      class: 'mrow', 'data-t': m.t_start === null ? '' : m.t_start,
      onclick: () => { if (m.t_start !== null && m.t_start !== undefined) { momStepTo(m.t_start); seekTo(m.t_start); } },
    },
      h('span', { class: 't', text: m.t_start === null || m.t_start === undefined
        ? '—' : timecode(m.t_start) }),
      h('span', { class: 'rail', style: `background:${color(m.source)}` }),
      h('span', { class: 'txt' }, marked(m.text, S.query))));
  }
}

/* ── moment navigation ────────────────────────────────────────────────── */
function momIndex(t) {
  const dur = (S.video && S.video.duration) || 0;
  return momTimes.findIndex(x => Math.abs(x - t) < Math.max(0.6, dur * 0.005));
}

function momStepTo(t) {
  momAt = momIndex(t);
  $('momAt').textContent = momAt < 0 ? timecode(t || 0)
    : `${momAt + 1} / ${momTimes.length}`;
  const pos = momAt < 0 ? '' : 'current';
  $$('#panel-moments .mrow').forEach(row => {
    row.dataset.step = (momAt >= 0 && Number(row.dataset.t) === momTimes[momAt])
      ? pos : '';
  });
  return momAt;
}

function momGo(delta) {
  if (!momTimes.length) return;
  const now = $('video').currentTime;
  let i = momIndex(now);
  if (i < 0) i = now <= (momTimes[0] || 0) ? 0 : momTimes.length;
  i += delta;
  i = Math.max(0, Math.min(momTimes.length - 1, i));
  if (momStepTo(momTimes[i]) >= 0) seekTo(momTimes[i]);
}

/* Loop the passage the player is sitting on, so a moment can be watched twice
 * without touching the scrubber. The window runs to the next indexed moment,
 * capped — a 40-second gap between passages is not a loop anyone wants. */
function momLoopToggle() {
  if (momLoop) { stopMomLoop(); return; }
  const i = momIndex($('video').currentTime);
  const t = momTimes[i];
  if (t === undefined) return;
  const next = momTimes[i + 1];
  const span = Math.min(6, Math.max(1.5, (next === undefined ? 3 : next - t)));
  $('momLoop').dataset.pressed = 'true';
  $('momLoop').setAttribute('aria-pressed', 'true');
  momLoop = setInterval(() => {
    const vid = $('video');
    if (vid.currentTime < t - 0.4 || vid.currentTime > t + span) {
      vid.currentTime = t;
      vid.play().catch(() => {});
    }
  }, 150);
}

function stopMomLoop() {
  clearInterval(momLoop);
  momLoop = null;
  const b = $('momLoop');
  if (b) { delete b.dataset.pressed; b.setAttribute('aria-pressed', 'false'); }
}

/* ── how big the player is ──
   The rail is 452 px wide, which is right for glancing at a result while you
   keep reading the list and wrong for actually watching a reel. Three sizes,
   because the two extremes both have a real use: rail to browse, wide to
   watch, theatre to study one video with everything else out of the way. */
const PLAYER_SIZES = ['rail', 'wide', 'theatre'];

function playerSize(next) {
  const el = $('player');
  const now = el.dataset.size || 'rail';
  const size = next || PLAYER_SIZES[(PLAYER_SIZES.indexOf(now) + 1) % 3];
  el.dataset.size = size;
  document.body.dataset.playerSize = size;
  try { localStorage.setItem('atlas.playerSize', size); } catch {}
  const b = $('screenSize');
  if (b) b.title = `Player size: ${size} — press f to change`;
  // The maps and graph canvases size themselves off their viewport, so a
  // change to the player's width has to be followed by a re-measure or they
  // draw at the old size until the next window resize.
  if (S.tab === 'maps') { mapsResize(); mapsDraw(); }
  if (S.tab === 'graph' && typeof gresize === 'function') gresize();
}

function momNavWire() {
  $('momPrev').addEventListener('click', () => momGo(-1));
  $('momNext').addEventListener('click', () => momGo(1));
  $('momLoop').addEventListener('click', momLoopToggle);
  $('screenSize').addEventListener('click', () => playerSize());
  let saved = '';
  try { saved = localStorage.getItem('atlas.playerSize') || ''; } catch {}
  playerSize(PLAYER_SIZES.includes(saved) ? saved : 'rail');

  // Keyboard, but only when the person is not typing into something. Every
  // key here is a single letter for the same reason a video editor's are.
  window.addEventListener('keydown', (ev) => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const el = document.activeElement;
    if (el && (el.tagName === 'INPUT' || el.tagName === 'SELECT' ||
               el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
    if (!S.video && ev.key !== 'Escape') return;
    const vid = $('video');
    switch (ev.key) {
      case 'n': momGo(1); break;
      case 'p': momGo(-1); break;
      // j / k / l are the scrub triple every editor already knows, so loop
      // takes 'o' rather than stealing 'l'.
      case 'o': momLoopToggle(); break;
      case 'f': playerSize(); break;
      // The cell inspector opens over the player and closes first, so Escape
      // does not dismiss both layers with one press.
      case 'Escape':
        if (!$('cellSlab').hidden) return;
        if (S.video) closePlayer(); else return;
        break;
      case ' ':
      case 'k': if (vid.paused) vid.play().catch(() => {}); else vid.pause(); break;
      case 'j': vid.currentTime = Math.max(0, vid.currentTime - 5); break;
      case 'l': vid.currentTime = vid.currentTime + 5; break;
      case 'm': vid.muted = !vid.muted; break;
      default: return;
    }
    ev.preventDefault();
  });
}

function markActiveCard() {
  const key = S.video && S.video.video_key;
  $$('#cards .card').forEach(c =>
    c.dataset.active = String(c.dataset.key === key));
}

function closePlayer() {
  clearInterval(statePoll);
  clearTimeout(retryTimer);
  clearTimeout(busyGuard);
  stopMomLoop();
  momTimes = [];
  momAt = -1;
  S.video = null;
  const vid = $('video');
  vid.pause();
  vid.removeAttribute('src');
  vid.load();
  $('playerLive').hidden = true;
  $('playerIdle').hidden = false;
  $('player').dataset.open = 'false';
  markActiveCard();
}

/* ── the database record for the open video ───────────────────────────── */
async function loadRecord(key) {
  const panel = $('panel-record');
  panel.textContent = '';
  panel.appendChild(h('p', { class: 'hint', text: 'reading the record…' }));
  try {
    const data = await api(`/api/video/${encodeURIComponent(key)}`);
    if (!S.video || S.video.video_key !== key) return;
    S.record = data;
    renderRecord(data);
    // The moment list from search only carries the matches; the record has
    // every passage, which is the more useful thing once a video is open.
    if ((data.moments || []).length > (S.video.moments || []).length) {
      renderMoments({ ...S.video, moments: data.moments }, null);
    }
  } catch (e) {
    panel.textContent = '';
    panel.appendChild(h('p', { class: 'hint', text: 'Could not read it: ' + e.message }));
  }
}

function renderRecord(data) {
  const panel = $('panel-record');
  panel.textContent = '';

  const meta = data.meta || {};
  if (Object.keys(meta).length) {
    panel.appendChild(h('div', { class: 'rec-h', text: 'Summary' }));
    panel.appendChild(kvTable(meta));
  }

  if (data.playback) {
    panel.appendChild(h('div', { class: 'rec-h', text: 'Playback' }));
    panel.appendChild(kvTable({
      location: { local: 'on this machine', cache: 'downloaded here',
                  remote: 'in the channel, not yet fetched',
                  missing: 'no message id' }[data.playback.where] || data.playback.where,
      size: data.playback.size ? fmtBytes(data.playback.size) : '—',
      telegram_message: data.playback.msg_id || '—',
    }));
  }

  // Every table in the bundle that has a row for this video, whatever those
  // tables are. Nothing here is hard-coded, so a new table just appears.
  for (const rel of data.related || []) {
    panel.appendChild(h('div', { class: 'rec-h' },
      rel.table,
      h('span', { class: 'rec-count', text: `${rel.rows.length} row${rel.rows.length === 1 ? '' : 's'}` })));
    if (rel.rows.length === 1) {
      panel.appendChild(kvTable(rel.rows[0]));
    } else {
      const wrap = h('div', { class: 'table-wrap', style: 'max-height:300px' });
      // Passing the table name makes these cells askable too: the same
      // drill-down the Data tab offers, without leaving the player.
      wrap.appendChild(rowTable(rel.columns,
        rel.rows.map(r => rel.columns.map(c => r[c])), null, rel.table));
      panel.appendChild(wrap);
    }
  }

  if (!Object.keys(meta).length && !(data.related || []).length)
    panel.appendChild(h('p', { class: 'hint',
      text: 'No database rows carry this video key.' }));
}

function kvTable(obj) {
  const t = h('table', { class: 'rec-table' });
  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined || v === '') continue;
    let shown = v;
    if (typeof v === 'object') shown = JSON.stringify(v);
    if (k === 'created_at' || k === 'imported_at') shown = fmtWhen(v) || shown;
    t.appendChild(h('tr', {}, h('th', { text: k }),
                            h('td', { text: String(shown) })));
  }
  return t;
}

/* `src` names the table these rows came from, and `rowids` their identity in
   it. Given both, every cell becomes a question you can ask — see openCell. */
function rowTable(columns, rows, types, src, rowids) {
  const t = h('table', { class: 'grid-table' });
  const head = h('tr', {});
  columns.forEach(c => head.appendChild(h('th', { text: c })));
  t.appendChild(head);
  rows.forEach((row, ri) => {
    const tr = h('tr', {});
    row.forEach((cell, i) => {
      const ask = src
        ? { class: 'cell cell-ask', role: 'button', tabindex: '0',
            title: `${src}.${columns[i]} — what is this?`,
            onclick: () => openCell(src, columns[i],
                                    rowids ? rowids[ri] : null, cell),
            // role="button" is a promise the keyboard has to be able to keep.
            onkeydown: (ev) => {
              if (ev.key !== 'Enter' && ev.key !== ' ') return;
              ev.preventDefault();
              openCell(src, columns[i], rowids ? rowids[ri] : null, cell);
            } }
        : { class: 'cell' };
      if (cell === null || cell === undefined) {
        tr.appendChild(h('td', {}, h('span', {
          ...ask, class: (src ? 'null cell-ask' : 'null'), text: 'null' })));
        return;
      }
      const numeric = typeof cell === 'number' ||
        (types && /INT|REAL|NUM|FLOAT|DOUBLE/i.test(types[i] || ''));
      const text = typeof cell === 'object' ? JSON.stringify(cell) : String(cell);
      tr.appendChild(h('td', { class: numeric ? 'num' : '' },
        h('span', { ...ask, text })));
    });
    t.appendChild(tr);
  });
  return t;
}

/* ── the cell inspector ───────────────────────────────────────────────────
   A number in a grid is not an answer, it is a lookup nobody has done yet.
   This does the lookup: what the column is for, whether search reads it,
   what the value points at, how many rows share it, which other tables say
   the same thing — and the reel it all belongs to, ready to play. */
let cellSeq = 0;

async function openCell(table, column, rowid, fallback) {
  const slab = $('cellSlab');
  const body = $('cellBody');
  slab.hidden = false;
  body.textContent = '';
  body.appendChild(h('p', { class: 'hint', text: 'reading this cell…' }));

  // Clicks come faster than the round trip, and the slowest answer must not
  // be the one left on screen.
  const token = ++cellSeq;

  const p = new URLSearchParams({ table, column });
  if (rowid !== null && rowid !== undefined) p.set('rowid', String(rowid));
  if (fallback !== null && fallback !== undefined && typeof fallback !== 'object')
    p.set('value', String(fallback));

  let d;
  try {
    d = await api('/api/cell?' + p.toString());
  } catch (e) {
    if (token !== cellSeq) return;
    body.textContent = '';
    body.appendChild(h('p', { class: 'hint', text: 'Could not read it: ' + e.message }));
    return;
  }
  if (token !== cellSeq) return;

  body.textContent = '';
  body.appendChild(h('div', { class: 'cs-crumb' },
    h('span', { text: d.table }), h('i', { text: '·' }),
    h('b', { text: d.column })));

  // A feed in the Sources tab names a column with no particular row behind it,
  // so there is no value to print — only what the column is for.
  const noValue = d.value === null || d.value === undefined;
  if (noValue && rowid === null)
    body.appendChild(h('p', { class: 'hint',
      text: 'The whole column, not one value — open a row in Data to ask about a single cell.' }));
  else
    body.appendChild(h('pre', { class: 'cs-value',
      text: noValue ? 'null'
        : typeof d.value === 'object' ? JSON.stringify(d.value, null, 2)
        : String(d.value) }));
  $('cellTitle').textContent = noValue && rowid === null ? 'What this column is'
                                                         : 'What this is';

  // What it is. The role is inferred from the schema, and it is the same
  // inference search itself uses — so this is the truth, not a description.
  const facts = h('div', { class: 'cs-facts' });
  const fact = (k, v) => { facts.append(h('span', { text: k }), h('b', { text: v })); };
  fact('role', ROLE_MEANS[d.role] || d.role);
  fact('declared type', d.type + (d.pk ? ' · primary key' : ''));
  fact('read by search', d.indexed
    ? `yes — weighted as ${SOURCE_LABEL[d.source] || d.source}`
    : 'no — reference only, never matched');
  if (d.same_value !== null && d.same_value !== undefined)
    fact('rows with this value', d.same_value === 1
      ? 'just this one — unique here'
      : `${fmtInt(d.same_value)} in ${d.table}`);
  body.appendChild(facts);

  if (d.refers_to) {
    body.appendChild(h('h4', { class: 'cs-h',
      text: `points at ${d.refers_to.table}.${d.refers_to.on}` }));
    const grid = h('div', { class: 'cs-kv' });
    for (const [k, v] of Object.entries(d.refers_to.row))
      grid.append(h('span', { text: k }),
                  h('b', { text: v === null ? 'null' : String(v) }));
    body.appendChild(grid);
  }

  if ((d.elsewhere || []).length) {
    body.appendChild(h('h4', { class: 'cs-h', text: 'the same value elsewhere' }));
    const list = h('div', { class: 'cs-else' });
    for (const e of d.elsewhere)
      list.appendChild(h('button', {
        class: 'linky',
        text: `${e.table}.${e.column} — ${fmtInt(e.rows)} row${e.rows === 1 ? '' : 's'}`,
        // Land on the rows that actually carry this value, then ask the same
        // question of the other table's column.
        onclick: () => {
          showTab('data');
          openTable(e.table, 0, String(d.value));
          openCell(e.table, e.column, null, d.value);
        },
      }));
    body.appendChild(list);
  }

  if (d.video && d.video.video_key) {
    body.appendChild(h('h4', { class: 'cs-h', text: 'the reel this row is about' }));
    const at = cellTime(d);
    const card = h('div', { class: 'cs-reel' });
    card.appendChild(posterImg(d.video, at, 'cs-poster'));
    card.appendChild(h('div', {},
      h('div', { class: 'cs-reel-title', text: d.video.title || d.video.video_key }),
      h('button', {
        class: 'btn btn-tiny',
        text: at === null ? 'Play it' : `Play from ${timecode(at)}`,
        // If the row named a time, open at it — a claim row is about a moment,
        // not about a whole reel.
        onclick: () => openVideoKey(d.video.video_key, at),
      })));
    body.appendChild(card);
  }

  if (d.row && Object.keys(d.row).length) {
    const det = h('details', { class: 'cs-row' });
    det.appendChild(h('summary', { text: 'the whole row, as stored' }));
    const grid = h('div', { class: 'cs-kv' });
    for (const [k, v] of Object.entries(d.row))
      grid.append(
        h('span', { text: k }),
        h('b', {
          class: k === d.column ? 'on' : '',
          text: v === null ? 'null' : typeof v === 'object' ? JSON.stringify(v) : String(v),
        }));
    det.appendChild(grid);
    body.appendChild(det);
  }
}

/* The second this row is about, when it is about one. `time_column` is the
   column reflect.py inferred as the start; a video-level row has no such
   column and the reel opens at its beginning. */
function cellTime(d) {
  const t = d.time_column ? d.row[d.time_column] : null;
  const n = Number(t);
  return t === null || t === undefined || !isFinite(n) ? null : n;
}

/* Plain words for the roles reflect.py infers, because "start" on its own
   does not tell you that this is the column search reads a timestamp from. */
const ROLE_MEANS = {
  key: 'the key — this is how the row joins to a video',
  start: 'when it begins — search reads this as the moment',
  end: 'when it ends',
  content: 'text search reads and matches',
  field: 'a field carried along, not matched',
};

/* ── similar ──────────────────────────────────────────────────────────── */
async function loadSimilar(key) {
  const panel = $('panel-similar');
  if (panel.dataset.key === key) return;
  panel.dataset.key = key;
  panel.textContent = '';
  panel.appendChild(h('p', { class: 'hint', text: 'comparing…' }));
  try {
    const data = await api(`/api/similar/${encodeURIComponent(key)}`);
    panel.textContent = '';
    if (!(data.results || []).length) {
      panel.appendChild(h('p', { class: 'hint',
        text: 'Nothing close enough yet — this needs the meaning index, which builds in the background.' }));
      return;
    }
    const grid = h('div', { class: 'simgrid' });
    for (const r of data.results) {
      const tile = h('div', {
        class: 'tile',
        onclick: () => openVideo({ ...r, moments: [] }, null),
        onpointerenter: () => prefetch([r.video_key]),
      });
      const shot = posterImg(r, null, 'tile-shot');
      if (r.duration) shot.appendChild(h('span', { class: 'dur', text: timecode(r.duration) }));
      tile.append(shot,
        h('div', { class: 'tile-title', text: r.title }),
        h('div', { class: 'tile-line',
          text: `${Math.round(r.similarity * 100)}% alike` }));
      grid.appendChild(tile);
    }
    panel.appendChild(grid);
  } catch (e) {
    panel.textContent = '';
    panel.appendChild(h('p', { class: 'hint', text: e.message }));
  }
}

function showPanel(name) {
  $$('.strip button').forEach(b => b.classList.toggle('on', b.dataset.panel === name));
  ['moments', 'record', 'similar'].forEach(p => { $('panel-' + p).hidden = p !== name; });
  if (name === 'similar' && S.video) loadSimilar(S.video.video_key);
}

/* ════════════════════════════════════════════════════════════════════════
   LIBRARY
   ════════════════════════════════════════════════════════════════════════ */
async function loadLibrary(reset) {
  if (reset) { S.lib.offset = 0; S.lib.rows = []; }
  S.lib.q = $('libQ').value.trim();
  const p = new URLSearchParams({
    limit: String(LIB_LIMIT), offset: String(S.lib.offset),
    sort: $('libSort').value, has: $('libHas').value, q: S.lib.q,
  });
  if (S.lib.creator) p.set('creator', S.lib.creator);
  if (S.lib.category) p.set('category', S.lib.category);

  if (reset) libNote('searching…');
  try {
    const data = await api('/api/library?' + p.toString());
    S.lib.rows = S.lib.rows.concat(data.results || []);
    S.lib.total = data.total || 0;
    S.lib.inside = data.inside || 0;
    S.lib.offset = S.lib.rows.length;
    renderLibrary(reset);
    libNote();
  } catch (e) { libNote(''); toast('Library failed: ' + e.message); }

  if (!S.facets) loadFacets();
}

/* What the library is currently showing, and — when a query is running — how
 * many of those videos were found by what is *inside* them rather than by
 * their title. Without that line, a result whose title says nothing about the
 * query looks like a bug instead of the feature it is. */
function libNote(override) {
  const el = $('libNote');
  if (override !== undefined) { el.textContent = override; return; }
  const bits = [`${fmtInt(S.lib.total)} video${S.lib.total === 1 ? '' : 's'}`];
  if (S.lib.q) {
    bits.push(`matching “${S.lib.q}”`);
    if (S.lib.inside)
      bits.push(`· ${fmtInt(S.lib.inside)} found by what is spoken, seen or written inside them`);
  }
  const filters = [S.lib.creator, S.lib.category].filter(Boolean);
  if (filters.length) bits.push(`· ${filters.join(' · ')}`);
  el.textContent = bits.join(' ');
}

/* How many tiles fit on a row. Kept in a custom property so the grid rule
 * stays one line of CSS, and remembered, because a density is a preference
 * rather than a per-visit decision. */
const LIB_DENSITY = { 1: 1, 2: 2, 3: 0, 4: 5, 5: 7 };   // 0 = the CSS default
function libDensity(step) {
  const n = LIB_DENSITY[step] || 0;
  const grid = $('libGrid');
  if (n) grid.style.setProperty('--tile-cols', String(n));
  else grid.style.removeProperty('--tile-cols');
  try { localStorage.setItem('atlas.libDensity', String(step)); } catch { /* private mode */ }
}

function renderLibrary(reset) {
  const grid = $('libGrid');
  if (reset) { previewStop(); grid.textContent = ''; }
  const frag = document.createDocumentFragment();

  for (const r of S.lib.rows.slice(grid.childElementCount)) {
    const tile = h('div', {
      class: 'tile', tabindex: '0',
      onclick: () => openLibraryVideo(r),
      onkeydown: (ev) => { if (ev.key === 'Enter') openLibraryVideo(r); },
      onpointerenter: () => prefetch([r.video_key]),
    });
    const shot = posterImg(r, null, 'tile-shot');
    if (r.duration) shot.appendChild(h('span', { class: 'dur', text: timecode(r.duration) }));
    // No query means no found moment to show, so the browse grid previews just
    // past the opening — far enough in to be the video rather than its title card.
    previewOn(shot, r.video_key, r.duration ? Math.min(r.duration * 0.1, 6) : 0);

    // A miniature of the ribbon: the mix of evidence this video carries,
    // without needing a query to have been run.
    const counts = r.sources || {};
    const totalMoments = Object.values(counts).reduce((a, b) => a + b, 0);
    if (totalMoments) {
      const rail = h('div', { class: 'tile-rail' });
      for (const [src, n] of Object.entries(counts))
        rail.appendChild(h('i', { style: `flex:${n};background:${color(src)}` }));
      shot.appendChild(rail);
    }

    if (r.matched === 'inside')
      tile.appendChild(h('span', { class: 'tile-why', text: 'found inside',
        title: 'The words are spoken, seen or written in this video, not in its title.' }));

    tile.append(shot,
      h('div', { class: 'tile-title', text: r.title || r.caption || r.video_key }),
      h('div', { class: 'tile-line',
        text: [r.creator, `${fmtInt(r.moment_count || 0)} passages`]
          .filter(Boolean).join(' · ') }));
    frag.appendChild(tile);
  }
  grid.appendChild(frag);

  $('libMore').hidden = S.lib.rows.length >= S.lib.total;
  $('libMoreBtn').textContent =
    `Load ${Math.min(LIB_LIMIT, S.lib.total - S.lib.rows.length)} more of ${fmtInt(S.lib.total)}`;
}

function openLibraryVideo(row) {
  openVideo({
    video_key: row.video_key, title: row.title || row.caption || row.video_key,
    caption: row.caption, creator: row.creator, category: row.category,
    duration: row.duration, width: row.width, height: row.height,
    likes: row.likes, created_at: row.created_at, msg_id: row.msg_id,
    moment_count: row.moment_count, hit_count: 0, moments: [],
  }, null);
}

async function loadFacets() {
  try {
    S.facets = await api('/api/facets');
    renderFacets();
    renderOpening();
    renderHome();
  } catch { /* the library still works without them */ }
}

function renderFacets() {
  const box = $('libFacets');
  box.textContent = '';
  if (!S.facets) return;

  const add = (list, field) => {
    if (!list || !list.length) return;
    for (const item of list.slice(0, 12)) {
      const on = S.lib[field] === item.value;
      box.appendChild(h('button', {
        class: 'facet', 'aria-pressed': String(on),
        onclick: () => { S.lib[field] = on ? '' : item.value; renderFacets(); loadLibrary(true); },
      }, item.value, h('span', { class: 'n', text: fmtInt(item.count) })));
    }
  };
  add(S.facets.creators, 'creator');
  add(S.facets.categories, 'category');
}

function renderOpening() {
  const f = S.facets;
  const st = S.status && S.status.search;
  const stats = $('openingStats');
  stats.textContent = '';
  const pairs = [
    ['videos', st ? st.videos : (f && f.totals.videos) || 0],
    ['indexed passages', st ? st.moments : (f && f.totals.moments) || 0],
    ['playable now', st ? st.playable : 0],
  ];
  if (st && st.dense_count) pairs.push(['meaning vectors', st.dense_count]);
  for (const [label, n] of pairs)
    stats.appendChild(h('div', {},
      h('span', { class: 'stat-n', text: fmtInt(n) }),
      h('span', { class: 'stat-l', text: label })));

  // Openers built from what is actually in the corpus, so every one of them
  // returns something.
  const tries = $('openingTries');
  tries.textContent = '';
  const picks = [];
  if (f) {
    for (const c of (f.categories || []).slice(0, 3)) picks.push(c.value);
    for (const c of (f.creators || []).slice(0, 2)) picks.push(c.value);
  }
  if (!picks.length) return;
  tries.appendChild(h('div', { class: 'tries-label', text: 'Start from what is in here' }));
  for (const p of picks)
    tries.appendChild(h('button', { class: 'try', onclick: () => runSearch(p) }, p));
}

/* ════════════════════════════════════════════════════════════════════════
   HOME
   ════════════════════════════════════════════════════════════════════════
   The landing page reports the system rather than describing it. Every figure
   comes from /api/status, which is already polled for the top-bar pulse, so
   opening this tab costs one library call and nothing else. A stage that has
   not run says so — an archive half-way through indexing should not advertise a
   total it does not have. */
const REDUCED = () => window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function homeBoot() {
  renderHome();
  // Only once: the six newest videos do not change while the tab is open, and
  // re-fetching them on every visit would flicker posters that are already
  // decoded.
  if (S.home.latest === null) loadHomeLatest();
}

/* A number that counts up reads as a measurement being taken; a number that
   jumps reads as a page loading. It runs once per tile per session, so a status
   poll refreshing the value does not restart the animation from zero. */
function countTo(el, value) {
  const target = Number(value) || 0;
  const key = el.dataset.k || '';
  const seen = S.home.counted[key];
  if (seen === target) return;
  S.home.counted[key] = target;
  if (REDUCED() || seen !== undefined || target <= 0) {
    el.textContent = fmtInt(target);
    return;
  }
  const started = performance.now(), span = 620;
  const step = (now) => {
    const t = Math.min(1, (now - started) / span);
    // Ease out: the last digits settle rather than snapping.
    const v = Math.round(target * (1 - Math.pow(1 - t, 3)));
    el.textContent = fmtInt(v);
    if (t < 1) requestAnimationFrame(step);
    else el.textContent = fmtInt(target);
  };
  requestAnimationFrame(step);
}

function homeHours(seconds) {
  const s = Number(seconds) || 0;
  if (s < 3600) return { n: Math.round(s / 60), unit: 'minutes of footage' };
  return { n: Math.round(s / 3600), unit: 'hours of footage' };
}

function renderHome() {
  if ($('view-home').hidden && S.tab !== 'home') return;
  renderHomeMetrics();
  renderHomePipe();
  renderHomeInside();
}

function renderHomeMetrics() {
  const box = $('homeMetrics');
  const st = S.status || {};
  const se = st.search || {};
  const f = S.facets;
  const hours = homeHours(se.seconds);

  const tiles = [
    ['videos', se.videos || (f && f.totals.videos) || 0, 'read end to end'],
    ['moments', se.moments || (f && f.totals.moments) || 0, 'passages indexed to the second'],
    ['hours', hours.n, hours.unit],
    ['vectors', se.dense_count || 0,
     se.dense_model ? `meaning vectors · ${se.dense_model}` : 'meaning vectors'],
    ['creators', se.creators || (f && f.creators ? f.creators.length : 0), 'creators represented'],
    ['links', (st.graph && st.graph.edges) || 0, 'relationships derived'],
  ];

  // Rebuilt in place: the tiles exist after the first paint, so a poll updates
  // the numbers without replacing the nodes the count-up is animating.
  if (box.childElementCount !== tiles.length) {
    box.textContent = '';
    for (const [k, , label] of tiles)
      box.appendChild(h('div', { class: 'metric', 'data-k': k },
        h('span', { class: 'metric-n', 'data-k': k, text: '0' }),
        h('span', { class: 'metric-l', text: label })));
  }
  tiles.forEach(([k, n, label], i) => {
    const tile = box.children[i];
    if (!tile) return;
    countTo(tile.querySelector('.metric-n'), n);
    tile.querySelector('.metric-l').textContent = label;
  });
}

/* Each row is one stage, in the order the data actually moves through them, so
   reading top to bottom is reading the pipeline. */
function renderHomePipe() {
  const st = S.status;
  const box = $('homePipe');
  box.textContent = '';
  if (!st) {
    box.appendChild(h('li', { class: 'pipe-row', 'data-state': 'wait' },
      h('span', { class: 'pipe-k', text: 'server' }),
      h('span', { class: 'pipe-v', text: 'still coming up' })));
    return;
  }

  const ing = st.ingest || {}, idx = st.index || {}, mp = st.map || {};
  const g = st.graph || {}, ca = st.cache || {};
  const rows = [];

  rows.push(['channel', ing.running
    ? ['work', ing.bytes_total
        ? `importing ${Math.round(100 * ing.bytes_done / ing.bytes_total)}%`
        : (ing.scan_total ? `scanning ${fmtInt(ing.scanned)} of ${fmtInt(ing.scan_total)}`
                          : (ing.current || 'scanning'))]
    : (st.bundles ? ['ok', `${fmtInt(st.bundles)} bundle${st.bundles === 1 ? '' : 's'} imported`]
                  : ['wait', ing.error || 'nothing imported yet'])]);

  rows.push(['index', idx.running
    ? ['work', idx.embed_total
        ? `embedding ${fmtInt(idx.embedded)} of ${fmtInt(idx.embed_total)}`
        : (idx.detail || idx.phase)]
    : (idx.error ? ['bad', idx.error]
      : idx.dense_ready ? ['ok', 'keywords and meaning both ready']
      : idx.lexical_ready ? ['wait', 'keywords ready · meaning search off']
      : ['wait', 'not indexed yet'])]);

  rows.push(['graph', g.nodes
    ? ['ok', `${fmtInt(g.nodes)} nodes · ${fmtInt(g.edges)} links`]
    : ['wait', 'not derived yet']]);

  rows.push(['maps', mp.running
    ? ['work', mp.detail || mp.phase]
    : (mp.phase === 'unavailable' ? ['wait', mp.detail || 'projection libraries unavailable']
      : mp.error ? ['bad', mp.error]
      : mp.points ? ['ok', `${fmtInt(mp.points)} points · ${fmtInt(mp.clusters)} clusters${mp.method ? ' · ' + mp.method : ''}`]
      : ['wait', 'not projected yet'])]);

  rows.push(['playback', (st.search && st.search.playable)
    ? ['ok', `${fmtInt(st.search.playable)} playable now · ${fmtBytes(ca.bytes || 0)} cached`
             + (ca.limit_gb ? ` of ${ca.limit_gb} GB` : '')]
    : ['wait', 'nothing cached — the first play streams from the channel']]);

  const tg = st.telegram || {};
  if (!tg.configured)
    rows.push(['channel access', ['bad',
      'credentials missing' + ((tg.missing || []).length ? ': ' + tg.missing.join(', ') : '')]]);

  for (const [k, [state, v]] of rows)
    box.appendChild(h('li', { class: 'pipe-row', 'data-state': state },
      h('span', { class: 'pipe-k', text: k }),
      h('span', { class: 'pipe-v', text: v })));
}

function renderHomeInside() {
  const box = $('homeInside');
  const f = S.facets;
  box.textContent = '';
  if (!f || (!(f.categories || []).length && !(f.creators || []).length)) {
    box.appendChild(h('p', { class: 'hint',
      text: 'nothing indexed yet — this fills in as the archive imports' }));
    return;
  }
  const add = (items, kind) => {
    for (const it of (items || []).slice(0, 8))
      box.appendChild(h('button', {
        class: 'home-chip', 'data-kind': kind,
        title: `Search the archive for “${it.value}”`,
        onclick: () => { showTab('search'); $('q').value = it.value; runSearch(it.value); },
      }, it.value, h('span', { class: 'n', text: fmtInt(it.count) })));
  };
  add(f.categories, 'category');
  add(f.creators, 'creator');

  // Openers built from the corpus, on the hero, for the same reason: an example
  // query that returns nothing teaches the wrong thing about the archive.
  const tries = $('homeTries');
  tries.textContent = '';
  const picks = [...(f.categories || []).slice(0, 2).map(c => c.value),
                 ...(f.creators || []).slice(0, 2).map(c => c.value)];
  if (!picks.length) return;
  tries.appendChild(h('span', { class: 'tries-label', text: 'try' }));
  for (const p of picks)
    tries.appendChild(h('button', {
      class: 'try',
      onclick: () => { showTab('search'); $('q').value = p; runSearch(p); },
    }, p));
}

async function loadHomeLatest() {
  const box = $('homeLatest');
  box.textContent = '';
  box.appendChild(h('p', { class: 'hint', text: 'reading the library…' }));
  try {
    const data = await api('/api/library?limit=6&offset=0&sort=recent');
    S.home.latest = data.results || [];
  } catch {
    S.home.latest = [];
  }
  box.textContent = '';
  if (!S.home.latest.length) {
    box.appendChild(h('p', { class: 'hint',
      text: 'no videos yet — import a bundle from the channel and they appear here' }));
    return;
  }
  // Warm the transfers for what is on screen: the tiles are the most likely
  // first click on the page.
  prefetch(S.home.latest.map(v => v.video_key));
  for (const v of S.home.latest) {
    const shot = posterImg(v, 0, 'ht-shot');
    previewOn(shot, v.video_key, 0);
    box.appendChild(h('button', {
      class: 'htile', onclick: () => openVideo(v, null),
      title: v.title || v.video_key,
    },
      shot,
      h('span', { class: 'ht-title', text: v.title || v.caption || v.video_key }),
      h('span', { class: 'ht-meta' },
        h('span', { text: v.creator || 'unattributed' }),
        h('span', { class: 'ht-dot', text: '·' }),
        h('span', { text: v.duration ? timecode(v.duration) : '—' }),
        h('span', { class: 'ht-dot', text: '·' }),
        h('span', { text: `${fmtInt(v.moment_count || 0)} moments` }))));
  }
}

/* ════════════════════════════════════════════════════════════════════════
   DATA
   ════════════════════════════════════════════════════════════════════════ */
async function loadSchema() {
  const box = $('schema');
  box.textContent = '';
  box.appendChild(h('p', { class: 'hint', text: 'reading the schema…' }));
  try {
    const data = await api('/api/schema');
    box.textContent = '';
    const indexed = data.tables.filter(t => t.indexed).length;
    $('dataNote').textContent =
      `${data.tables.length} tables, ${indexed} feeding search. ` +
      `Roles are inferred from the data itself — schema ${data.fingerprint.slice(0, 8)}.`;

    for (const t of data.tables.sort((a, b) => b.rows - a.rows)) {
      const card = h('div', { class: 'tcard', onclick: () => openTable(t.name, 0, '') },
        h('div', { class: 'tcard-head' },
          h('span', { class: 'tcard-name', text: t.name }),
          h('span', { class: 'tcard-flag', 'data-on': String(t.indexed),
                      text: t.indexed ? 'searchable' : 'reference' }),
          h('span', { class: 'tcard-rows', text: fmtInt(t.rows) + ' rows' })),
        h('div', { class: 'cols' },
          t.columns.map(c => h('span', {
            class: 'col', 'data-role': c.role,
            title: c.source ? `${c.type} · indexed as ${SOURCE_LABEL[c.source] || c.source}`
                            : c.type,
            text: c.name,
          }))));
      box.appendChild(card);
    }
  } catch (e) {
    box.textContent = '';
    box.appendChild(h('p', { class: 'hint', text: 'Could not read it: ' + e.message }));
  }
}

/* `filter` is optional and, when given, replaces the in-table search — a
   different table almost never wants the previous one's filter, and jumping
   to a value wants exactly that value. Omitting it keeps whatever is there,
   which is what paging and the search box itself need. */
async function openTable(name, offset, filter) {
  if (filter !== undefined) {
    S.browse.q = filter;
    $('browserQ').value = filter;
  }
  S.browse.table = name;
  S.browse.offset = offset || 0;
  $('browser').hidden = false;
  $('browserTitle').textContent = name;
  const p = new URLSearchParams({
    limit: '50', offset: String(S.browse.offset), q: S.browse.q,
  });
  try {
    const data = await api(`/api/table/${encodeURIComponent(name)}?` + p.toString());
    const table = rowTable(data.columns, data.rows, data.types, name, data.rowids);
    table.id = 'browserTable';
    table.className = 'grid-table';
    $('browserTable').replaceWith(table);

    const pager = $('browserPager');
    pager.textContent = '';
    const from = data.total ? data.offset + 1 : 0;
    pager.append(
      h('button', {
        class: 'btn btn-quiet', disabled: data.offset === 0,
        onclick: () => openTable(name, Math.max(0, data.offset - 50)),
      }, '← previous'),
      h('span', { text: `${fmtInt(from)}–${fmtInt(data.offset + data.rows.length)} of ${fmtInt(data.total)}` }),
      h('button', {
        class: 'btn btn-quiet',
        disabled: data.offset + data.rows.length >= data.total,
        onclick: () => openTable(name, data.offset + 50),
      }, 'next →'));
    $('browser').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) { toast('Could not read ' + name + ': ' + e.message); }
}

/* ════════════════════════════════════════════════════════════════════════
   SOURCES
   ════════════════════════════════════════════════════════════════════════ */
async function loadSources() {
  try {
    const [ch, b, lg] = await Promise.all([
      api('/api/channel').catch(e => ({ ok: false, error: e.message })),
      api('/api/bundles').catch(() => ({ bundles: [], sources: [] })),
      api('/api/log?limit=140').catch(() => ({ lines: [] })),
    ]);
    renderChannel(ch);
    renderBundles(b.bundles || []);
    renderFeeds(b.sources || []);
    $('log').textContent = (lg.lines || []).join('\n') || 'nothing logged yet';
    $('log').scrollTop = $('log').scrollHeight;
  } catch (e) { toast(e.message); }
}

function renderChannel(ch) {
  const box = $('channelCard');
  box.textContent = '';
  const st = S.status || {};
  const kv = (k, v) => box.appendChild(h('div', { class: 'kv' },
    h('div', { class: 'k', text: k }), h('div', { class: 'v', text: String(v) })));

  kv('channel', ch.channel || (st.telegram && st.telegram.channel) || '—');
  kv('reachable', ch.ok ? 'yes' : (ch.error || 'no'));
  if (ch.bot) kv('bot', '@' + ch.bot);
  kv('large-file transport', ch.mtproto ? 'MTProto (2 GB parts)' : 'Bot API only (20 MB parts)');
  if (ch.pinned_message_id) kv('pinned manifest', '#' + ch.pinned_message_id);
  if (st.cache) kv('video cache', `${st.cache.files} files · ${st.cache.gb} of ${st.cache.limit_gb} GB`);
  if (st.search) kv('meaning index', st.search.dense_ready
    ? `${fmtInt(st.search.dense_count)} vectors · ${st.search.dense_model || ''}`
    : 'building');
  if (ch.missing && ch.missing.length) kv('missing credentials', ch.missing.join(', '));
}

function renderBundles(rows) {
  const box = $('bundles');
  box.textContent = '';
  if (!rows.length) {
    box.appendChild(h('p', { class: 'hint',
      text: 'No bundles imported yet. Atlas scans the channel on start; use “Rescan channel” to look again.' }));
    return;
  }
  for (const b of rows) {
    const counts = h('div', { class: 'counts' });
    for (const [k, v] of Object.entries(b.counts || {}).slice(0, 8))
      counts.appendChild(h('span', { text: `${k} ${fmtInt(v)}` }));
    box.appendChild(h('div', { class: 'bundle' },
      h('div', {},
        h('div', { class: 'seq', text: b.seq }),
        h('div', { class: 'when', text: b.created_at || '' })),
      h('div', {}, counts,
        h('div', { class: 'when',
          text: `${b.parts || 0} part${b.parts === 1 ? '' : 's'} · ${fmtBytes(b.bytes)}` +
                (b.note ? ' · ' + b.note : '') })),
      h('span', { class: 'status', 'data-ok': String(b.status === 'ok'),
                  text: b.status || '?' })));
  }
}

function renderFeeds(sources) {
  const box = $('feeds');
  box.textContent = '';
  if (!sources.length) {
    box.appendChild(h('p', { class: 'hint', text: 'No text columns found yet.' }));
    return;
  }
  for (const s of sources)
    box.appendChild(h('button', {
      class: 'feed', style: `border-left-color:${color(s.source)}`,
      title: (s.via ? `joined through ${s.via}` : `keyed on ${s.key}`) +
             ' — click to see what this column is and browse it',
      // A feed is a column, so clicking one asks the same question a cell
      // does, minus a row: what is this, and does search read it?
      // Browsing means the Data tab, since that is where the table renders —
      // the slab floats over the stage and follows either way.
      onclick: () => {
        openCell(s.table, s.text, null, null);
        showTab('data');
        // A filter left over from some other table would silently hide most of
        // this one, so browsing a feed starts unfiltered.
        openTable(s.table, 0, '');
      },
    }, h('b', { text: s.table }), '.' + s.text));
}

/* ════════════════════════════════════════════════════════════════════════
   GRAPH

   A force-directed graph on a canvas, written here rather than pulled in,
   because the two things this one has to do are the two things a general
   library makes awkward: nodes that mean different things need to be drawn
   differently — a reel is a vertical frame, an entity is a disc — and a click
   has to reach the rest of the application, opening the same persistent
   player a search result opens.

   Three parts, in order: the layout (quadtree + springs), the paint, and the
   pointer. The layout runs on a clock that stops itself once the graph has
   settled, so an idle tab costs nothing.
   ════════════════════════════════════════════════════════════════════════ */
const G = {
  nodes: new Map(),           // id → node with position and velocity
  edges: new Map(),           // src|dst|rel → edge
  view: { x: 0, y: 0, k: 1 },
  sel: null, selEdge: null, hover: null, hoverEdge: null,
  drag: null, pan: null, moved: 0,
  alpha: 0, raf: 0, frozen: false,
  mode: 'data',
  off: new Set(),             // kinds switched off in the rail
  counts: null, loaded: false,
  posters: new Map(),         // video key → Image | 'no'
  posterOff: false,           // Frames toggle: read structure without imagery
  hits: [], hitIndex: -1,
  labelBoxes: [],
  traceFrom: null,            // first end of a "how are these two related" ask
  pathSet: null, pathEdges: null,
  colorBy: 'kind', sizeBy: 'degree',
  mini: null,                 // the minimap's world→map transform
  spread: 3, pull: 3, labels: 3,
  // Ranges for whatever continuous encoding is active, recomputed when the node
  // set changes. Without them a scale would rescale on every repaint and the
  // same node would change colour while nothing about it had.
  scale: null,
};

const GRAPH_TICK = {
  charge: -900,        // node-to-node repulsion
  theta: 0.9,          // Barnes-Hut opening angle
  spring: 0.055,       // edge stiffness
  gravity: 0.022,      // pull toward the middle, so nothing drifts away
  damp: 0.72,
  decay: 0.021,        // how fast the layout cools
  floor: 0.005,        // below this it is settled, stop the clock
};

const KIND_COLOR = {
  video: '#E6F0EE', dim: '#5EC8D8', tag: '#B9F18D',
  hashtag: '#8A9BA8', table: '#7E9A98', anchor: '#FFB020',
};
const KIND_LABEL = {
  video: 'video', dim: 'entity', tag: 'thing seen or said',
  hashtag: 'hashtag', table: 'table', anchor: 'join key',
};

/* A tag inherits the colour of the evidence it came from, so an object the
   vision model saw is the same cyan on this canvas as it is on a ribbon. */
const RAW_SOURCE_COLOR = {
  narrative: '#B9F18D', speech: '#FFB020', visual: '#5EC8D8',
  ocr: '#E8705C', caption: '#8A9BA8', meta: '#6E7F8C',
};

/* A cool-to-hot ramp for the continuous encodings. It stays inside the palette
   — dim through cyan to amber — so a heat scale still reads as Atlas and the
   amber end keeps meaning "this is the one you want". */
const GRAMP = ['#3C5F63', '#4E8890', '#5EC8D8', '#B9F18D', '#FFB020'];

function gramp(t) {
  if (!(t >= 0)) return GRAMP[0];
  const x = Math.max(0, Math.min(0.999, t)) * (GRAMP.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = GRAMP[i], b = GRAMP[Math.min(GRAMP.length - 1, i + 1)];
  const mix = (p) => {
    const av = parseInt(a.slice(p, p + 2), 16), bv = parseInt(b.slice(p, p + 2), 16);
    return Math.round(av + (bv - av) * f).toString(16).padStart(2, '0');
  };
  return '#' + mix(1) + mix(3) + mix(5);
}

/* What each continuous encoding reads off a node. Returning null means "this
   node has no value for this question" — it keeps the node visible in the
   neutral tone rather than pretending it sits at zero. */
function gvalue(node, what) {
  const m = node.meta || {};
  if (what === 'reach') return Number(node.deg) || 0;
  if (what === 'recency') {
    let t = Number(m.created_at);
    if (!isFinite(t) || !t) return null;
    if (t > 1e11) t = t / 1000;
    return t;
  }
  if (what === 'moments') return Number(m.moments) || (Number(node.weight) || 0);
  if (what === 'duration') {
    const d = Number(m.duration);
    return isFinite(d) && d > 0 ? d : null;
  }
  return null;
}

/* One pass over the live set fixes the domain of whichever scale is on. Log for
   counts, because one hub with 400 links would otherwise flatten everything
   else onto the same colour. */
function gscale() {
  const what = G.colorBy !== 'kind' && G.colorBy !== 'source' ? G.colorBy : null;
  const sizeWhat = G.sizeBy === 'degree' || G.sizeBy === 'flat' ? null : G.sizeBy;
  const dom = {};
  for (const key of [what, sizeWhat]) {
    if (!key) continue;
    let lo = Infinity, hi = -Infinity;
    for (const n of G.nodes.values()) {
      const v = gvalue(n, key);
      if (v === null) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    dom[key] = lo <= hi ? { lo, hi } : null;
  }
  G.scale = dom;
}

function gnorm(node, key) {
  if (!key) return null;
  const d = G.scale && G.scale[key];
  const v = gvalue(node, key);
  if (!d || v === null) return null;
  if (d.hi === d.lo) return 1;
  // Counts are log-scaled; a timestamp is already linear in the thing a reader
  // means by "more recent", so it is not.
  if (key === 'recency') return (v - d.lo) / (d.hi - d.lo);
  const l = Math.log2(Math.max(0, v) + 1);
  const llo = Math.log2(Math.max(0, d.lo) + 1);
  const lhi = Math.log2(Math.max(0, d.hi) + 1);
  return lhi === llo ? 1 : (l - llo) / (lhi - llo);
}

function gcolor(node) {
  if (!node) return '#6E7F8C';
  if (G.colorBy === 'reach' || G.colorBy === 'recency') {
    const t = gnorm(node, G.colorBy);
    return t === null ? '#3C5F63' : gramp(t);
  }
  if (G.colorBy === 'source') {
    // Every node the evidence hues apply to, not only tags: a video shows the
    // type of evidence that put it on the canvas.
    const s = (node.meta && node.meta.source) || null;
    return RAW_SOURCE_COLOR[s] || (node.kind === 'video' ? KIND_COLOR.video : '#6E7F8C');
  }
  if (node.kind === 'tag') {
    const s = node.meta && node.meta.source;
    return RAW_SOURCE_COLOR[s] || KIND_COLOR.tag;
  }
  return KIND_COLOR[node.kind] || '#6E7F8C';
}

function gradius(node) {
  if (G.sizeBy === 'flat') return node.kind === 'video' ? 11 : 9;
  if (G.sizeBy !== 'degree') {
    const t = gnorm(node, G.sizeBy);
    // No value for the chosen measure: sit at the small end rather than vanish.
    return 7 + (t === null ? 0 : t * 17);
  }
  // Degree decides size, on a log curve: a creator with 400 videos should read
  // as bigger than one with 4, not a hundred times bigger.
  const w = Math.max(1, Number(node.weight) || 1);
  if (node.kind === 'video') return 9 + Math.min(7, Math.log2(w + 1) * 1.6);
  return 7 + Math.min(19, Math.log2(w + 1) * 3.4);
}

/* Re-encode without refetching: gdegree already recomputes the domains and
   every radius in the right order, then the layout settles a little because the
   spring rest lengths depend on radius. */
function gre_encode() {
  gdegree();
  grenderLegend();
  gheat(0.28);
}

const gkey = (e) => `${e.src}|${e.dst}|${e.rel}`;

/* ── the model ──────────────────────────────────────────────────────────── */
function gmerge(payload, around) {
  const fresh = [];
  const list = payload.nodes || [];
  // New nodes land on a ring around whatever was expanded rather than at the
  // origin, so an expansion reads as unfolding instead of as an explosion.
  const spread = Math.max(70, 26 + list.length * 4);
  let i = 0;
  for (const raw of list) {
    let node = G.nodes.get(raw.id);
    if (node) {
      node.weight = raw.weight;
      node.label = raw.label;
      node.meta = raw.meta || node.meta;
      continue;
    }
    const angle = (i / Math.max(1, list.length)) * Math.PI * 2 + (i % 3) * 0.4;
    const base = around || { x: 0, y: 0 };
    node = {
      id: raw.id, kind: raw.kind, label: raw.label || raw.id,
      sub: raw.sub || '', weight: raw.weight || 0, meta: raw.meta || {},
      x: base.x + Math.cos(angle) * spread * (0.7 + (i % 5) * 0.09),
      y: base.y + Math.sin(angle) * spread * (0.7 + (i % 7) * 0.07),
      vx: 0, vy: 0, deg: 0, pin: false, expanded: false,
    };
    node.r = gradius(node);
    G.nodes.set(node.id, node);
    fresh.push(node);
    i++;
  }
  for (const raw of payload.edges || []) {
    const k = gkey(raw);
    if (G.edges.has(k)) continue;
    if (!G.nodes.has(raw.src) || !G.nodes.has(raw.dst)) continue;
    G.edges.set(k, {
      src: raw.src, dst: raw.dst, rel: raw.rel,
      weight: Number(raw.weight) || 1, ref: raw.ref || '',
    });
  }
  gdegree();
  return fresh;
}

function gdegree() {
  // Degrees first, then the scale domains, then the radii — in that order,
  // because a "by how connected" colour reads the degree it is about to scale,
  // and a "by indexed moments" radius reads the domain that scale computes.
  for (const n of G.nodes.values()) n.deg = 0;
  for (const e of G.edges.values()) {
    const a = G.nodes.get(e.src), b = G.nodes.get(e.dst);
    if (a) a.deg++;
    if (b) b.deg++;
  }
  gscale();
  for (const n of G.nodes.values()) n.r = gradius(n);
}

const gvisible = (n) => n && !G.off.has(n.kind);

function gliveNodes() {
  const out = [];
  for (const n of G.nodes.values()) if (gvisible(n)) out.push(n);
  return out;
}

function gliveEdges() {
  const out = [];
  for (const e of G.edges.values()) {
    const a = G.nodes.get(e.src), b = G.nodes.get(e.dst);
    if (gvisible(a) && gvisible(b)) out.push({ e, a, b });
  }
  return out;
}

/* ── layout: Barnes-Hut ─────────────────────────────────────────────────
   Repulsion between every pair is what spreads a graph out, and doing it
   honestly is O(n²) — 400 nodes is 160k distance calculations per frame,
   which drops the frame rate the moment anybody expands twice. A quadtree
   turns distant clusters into a single averaged mass, so the same pass costs
   O(n log n) and the layout stays smooth into the thousands. */
function gtree(nodes) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of nodes) {
    if (n.x < x0) x0 = n.x;
    if (n.y < y0) y0 = n.y;
    if (n.x > x1) x1 = n.x;
    if (n.y > y1) y1 = n.y;
  }
  if (!isFinite(x0)) return null;
  const size = Math.max(x1 - x0, y1 - y0, 1) * 1.05 + 2;
  const root = { x: x0 - 1, y: y0 - 1, s: size, cx: 0, cy: 0, m: 0, kids: null, leaf: null };
  for (const n of nodes) ginsert(root, n, 0);
  gfinish(root);
  return root;
}

const GTREE_DEPTH = 20;

function gchild(cell, n) {
  const half = cell.s / 2;
  const i = (n.x >= cell.x + half ? 1 : 0) + (n.y >= cell.y + half ? 2 : 0);
  if (!cell.kids) cell.kids = [null, null, null, null];
  if (!cell.kids[i]) {
    cell.kids[i] = {
      x: cell.x + (i & 1 ? half : 0), y: cell.y + (i & 2 ? half : 0),
      s: half, cx: 0, cy: 0, m: 0, kids: null, leaf: null,
    };
  }
  return cell.kids[i];
}

function ginsert(cell, n, depth) {
  for (;;) {
    cell.cx += n.x; cell.cy += n.y; cell.m++;
    if (!cell.kids && !cell.leaf) { cell.leaf = n; return; }
    // Two nodes at the same coordinates would subdivide forever. Past this
    // depth the cell just holds a list; the layout's separation pass will
    // have pulled them apart by the next frame anyway.
    if (depth >= GTREE_DEPTH) { (cell.extra || (cell.extra = [])).push(n); return; }
    if (cell.leaf) {
      const held = cell.leaf;
      cell.leaf = null;
      // Push the sitting tenant one level down before taking its place.
      // Bounded by GTREE_DEPTH, so this recursion cannot run away.
      ginsert(gchild(cell, held), held, depth + 1);
    }
    cell = gchild(cell, n);
    depth++;
  }
}

function gfinish(cell) {
  if (!cell) return;
  if (cell.m) { cell.cx /= cell.m; cell.cy /= cell.m; }
  if (cell.kids) for (const k of cell.kids) gfinish(k);
}

function grepel(root, n, strength) {
  const stack = [root];
  let fx = 0, fy = 0;
  while (stack.length) {
    const cell = stack.pop();
    if (!cell || !cell.m) continue;
    if (cell.leaf === n && cell.m === 1) continue;      // itself
    let dx = cell.cx - n.x, dy = cell.cy - n.y;
    let d2 = dx * dx + dy * dy;
    if (d2 < 1) {
      // Exactly coincident: nudge along a direction derived from the id, so
      // the jitter is the same every frame and the node does not shiver.
      dx = (n.id.length % 7) - 3.5;
      dy = (n.id.length % 5) - 2.5;
      d2 = dx * dx + dy * dy + 1;
    }
    // Small enough on screen, or a single node: treat as one mass. cell.m
    // already counts everything inside, including any coincident overflow.
    if (cell.leaf || cell.s * cell.s / d2 < GRAPH_TICK.theta * GRAPH_TICK.theta) {
      const d = Math.sqrt(d2);
      const f = strength * cell.m / (d2 * d);
      fx += dx * f; fy += dy * f;
      continue;
    }
    if (cell.kids) for (const k of cell.kids) if (k) stack.push(k);
  }
  n.vx += fx; n.vy += fy;
}

/* The two sliders that touch the simulation, as multipliers around the tuned
   default so the middle notch is exactly today's behaviour. */
const gspreadMul = () => [0.45, 0.7, 1, 1.55, 2.3][(G.spread | 0) - 1] || 1;
const gpullMul = () => [0.4, 0.68, 1, 1.5, 2.2][(G.pull | 0) - 1] || 1;

function gtick() {
  const nodes = gliveNodes();
  if (!nodes.length) return;
  const alpha = G.alpha;
  const root = gtree(nodes);
  const spread = gspreadMul();

  for (const n of nodes) {
    if (root) grepel(root, n, GRAPH_TICK.charge * spread * alpha);
    // Gravity resists spread, or turning it up walks the graph off screen
    // instead of opening it out.
    n.vx += -n.x * GRAPH_TICK.gravity * spread * alpha;
    n.vy += -n.y * GRAPH_TICK.gravity * spread * alpha;
  }

  for (const { a, b } of gliveEdges()) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.hypot(dx, dy) || 0.01;
    // Longer rest length for hubs so their satellites form a readable ring
    // instead of a solid disc of overlapping labels.
    const rest = 58 + a.r + b.r + Math.min(120, (a.deg + b.deg) * 0.9);
    const pull = (d - rest) * GRAPH_TICK.spring * gpullMul() * alpha;
    const ux = (dx / d) * pull, uy = (dy / d) * pull;
    // Each end moves in inverse proportion to its own degree — degree is the
    // node's mass here. Without this a creator with 400 videos is dragged
    // around by each of them in turn and the whole graph oscillates.
    const sa = 1 / Math.max(1, a.deg), sb = 1 / Math.max(1, b.deg);
    a.vx += ux * sa; a.vy += uy * sa;
    b.vx -= ux * sb; b.vy -= uy * sb;
  }

  for (const n of nodes) {
    if (n.pin || (G.drag && G.drag.node === n)) { n.vx = n.vy = 0; continue; }
    n.vx *= GRAPH_TICK.damp; n.vy *= GRAPH_TICK.damp;
    const cap = 34;
    n.vx = Math.max(-cap, Math.min(cap, n.vx));
    n.vy = Math.max(-cap, Math.min(cap, n.vy));
    n.x += n.vx; n.y += n.vy;
  }

  // A short separation pass. Repulsion alone leaves discs touching at rest,
  // and touching discs make the labels unreadable. It is O(n²), so it is the
  // first thing dropped as the graph grows — by then the repulsion is doing
  // the job well enough on its own.
  const passes = nodes.length > 900 ? 0 : (nodes.length > 420 ? 1 : 2);
  for (let pass = 0; pass < passes; pass++) {
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const want = a.r + b.r + 6;
        const d2 = dx * dx + dy * dy;
        if (d2 > want * want || d2 < 0.0001) continue;
        const d = Math.sqrt(d2);
        const push = (want - d) / d * 0.5;
        a.x -= dx * push; a.y -= dy * push;
        b.x += dx * push; b.y += dy * push;
      }
    }
  }

  G.alpha += (0 - G.alpha) * GRAPH_TICK.decay;
  if (G.alpha < GRAPH_TICK.floor) G.alpha = 0;
}

const REDUCED = window.matchMedia &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function gheat(to) {
  if (G.frozen) { gdraw(); return; }
  if (REDUCED) {
    // No animation wanted. Dragging still has to feel direct, so a drag just
    // repaints; anything else settles the layout in one go and shows the
    // finished arrangement rather than the journey to it.
    if (G.drag) { gdraw(); return; }
    G.alpha = Math.max(G.alpha, to === undefined ? 0.62 : to);
    for (let i = 0; i < 240 && G.alpha > GRAPH_TICK.floor; i++) gtick();
    G.alpha = 0;
    gdraw();
    return;
  }
  G.alpha = Math.max(G.alpha, to === undefined ? 0.62 : to);
  if (!G.raf) G.raf = requestAnimationFrame(gframe);
}

function gframe() {
  G.raf = 0;
  if (!G.frozen && G.alpha > 0) gtick();
  gdraw();
  if (!G.frozen && G.alpha > 0) G.raf = requestAnimationFrame(gframe);
}

/* ── paint ──────────────────────────────────────────────────────────────── */
let gctx = null, gcv = null, gsize = { w: 0, h: 0, dpr: 1 };

function gfit(padding) {
  const nodes = gliveNodes();
  if (!nodes.length) return;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of nodes) {
    x0 = Math.min(x0, n.x - n.r); y0 = Math.min(y0, n.y - n.r);
    x1 = Math.max(x1, n.x + n.r); y1 = Math.max(y1, n.y + n.r);
  }
  const pad = padding === undefined ? 90 : padding;
  const k = Math.min(
    (gsize.w - pad * 2) / Math.max(1, x1 - x0),
    (gsize.h - pad * 2) / Math.max(1, y1 - y0));
  G.view.k = Math.max(0.12, Math.min(2.2, k));
  G.view.x = gsize.w / 2 - ((x0 + x1) / 2) * G.view.k;
  G.view.y = gsize.h / 2 - ((y0 + y1) / 2) * G.view.k;
}

const gtoScreen = (x, y) => ({ x: x * G.view.k + G.view.x, y: y * G.view.k + G.view.y });
const gtoWorld = (x, y) => ({ x: (x - G.view.x) / G.view.k, y: (y - G.view.y) / G.view.k });

function gneighbourSet(id) {
  const set = new Set();
  if (!id) return set;
  for (const e of G.edges.values()) {
    if (e.src === id) set.add(e.dst);
    else if (e.dst === id) set.add(e.src);
  }
  return set;
}

function gdraw() {
  if (!gctx) return;
  const ctx = gctx, { w, h } = gsize;
  ctx.setTransform(gsize.dpr, 0, 0, gsize.dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const focus = G.sel || G.hover;
  const near = focus ? gneighbourSet(focus.id) : null;
  const k = G.view.k;

  // ── edges ──
  const edges = gliveEdges();
  ctx.lineCap = 'round';
  for (const { e, a, b } of edges) {
    const lit = !focus || focus.id === e.src || focus.id === e.dst;
    const hot = G.hoverEdge && gkey(G.hoverEdge) === gkey(e);
    const picked = (G.selEdge && gkey(G.selEdge) === gkey(e)) ||
                   (G.pathEdges && G.pathEdges.has(gkey(e)));
    const A = gtoScreen(a.x, a.y), B = gtoScreen(b.x, b.y);
    ctx.beginPath();
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
    if (picked || hot) {
      ctx.strokeStyle = '#FFB020';
      ctx.lineWidth = Math.max(1.6, 2.4 * Math.min(1.4, k));
      ctx.globalAlpha = 1;
    } else {
      // Coloured by whichever end carries the meaning: a link to an object
      // reads as that object's colour, not as a neutral grey.
      const teller = a.kind === 'video' ? b : (b.kind === 'video' ? a : b);
      ctx.strokeStyle = lit ? gcolor(teller) : '#24403F';
      ctx.lineWidth = Math.max(0.6, Math.min(2.6, 0.7 + Math.log2(e.weight + 1) * 0.5) * Math.min(1.3, k));
      ctx.globalAlpha = lit ? (focus ? 0.5 : 0.26) : 0.09;
    }
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // ── nodes ──
  const nodes = gliveNodes();
  // Painted small-first so hubs and the selection land on top.
  nodes.sort((p, q) => (p.r - q.r));
  G.labelBoxes = [];
  for (const n of nodes) {
    const p = gtoScreen(n.x, n.y);
    const r = n.r * k;
    if (p.x < -60 || p.y < -60 || p.x > w + 60 || p.y > h + 60) continue;
    const dim = focus && focus.id !== n.id && near && !near.has(n.id);
    const tone = gcolor(n);
    ctx.globalAlpha = dim ? 0.22 : 1;
    // A traced chain outranks the focus dimming — the whole point of asking
    // for a path is to see it against everything else.
    const onPath = G.pathSet && G.pathSet.has(n.id);
    if (onPath) ctx.globalAlpha = 1;

    if (n.kind === 'video') gpaintVideo(ctx, n, p, r);
    else gpaintEntity(ctx, n, p, r, tone);

    if ((G.sel && G.sel.id === n.id) || onPath) {
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r + 7, 0, Math.PI * 2);
      ctx.strokeStyle = '#FFB020';
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }
    if (!n.expanded && n.deg > 0 && !dim && r > 7) {
      // A quiet mark meaning "there is more behind this one".
      ctx.globalAlpha = dim ? 0.2 : 0.85;
      ctx.beginPath();
      ctx.arc(p.x + r * 0.72, p.y - r * 0.72, 2.6, 0, Math.PI * 2);
      ctx.fillStyle = '#FFB020';
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // ── labels last, and only where they fit ──
  const ranked = nodes.slice().sort((p, q) => {
    const pv = (G.sel && G.sel.id === p.id ? 1e9 : 0) + (G.hover && G.hover.id === p.id ? 1e8 : 0) + p.r;
    const qv = (G.sel && G.sel.id === q.id ? 1e9 : 0) + (G.hover && G.hover.id === q.id ? 1e8 : 0) + q.r;
    return qv - pv;
  });
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  // The labels slider is a radius threshold: at 1 only hubs are named, at 5
  // everything that fits is. Read once — it is the same for every node, and this
  // loop runs on every frame.
  const floor = [16, 12, 9, 6, 0][(G.labels | 0) - 1] ?? 9;
  for (const n of ranked) {
    const p = gtoScreen(n.x, n.y);
    const r = n.r * k;
    if (p.x < -40 || p.y < -40 || p.x > w + 40 || p.y > h + 40) continue;
    // The selection and the hover are always named, whatever the setting —
    // those two are answers to a direct question.
    const must = (G.sel && G.sel.id === n.id) || (G.hover && G.hover.id === n.id);
    if (!must && r < floor) continue;
    const dim = focus && focus.id !== n.id && near && !near.has(n.id);
    if (dim && !must) continue;

    const size = Math.max(10, Math.min(14, 9 + r * 0.22));
    ctx.font = `500 ${size}px 'Public Sans', sans-serif`;
    let text = n.label || '';
    if (text.length > 26) text = text.slice(0, 25) + '…';
    const tw = ctx.measureText(text).width;
    const box = { x: p.x - tw / 2 - 3, y: p.y + r + 5, w: tw + 6, h: size + 4 };
    if (!must && gcollides(box)) continue;
    G.labelBoxes.push(box);

    ctx.fillStyle = 'rgba(11,20,22,.76)';
    ctx.fillRect(box.x, box.y, box.w, box.h);
    ctx.fillStyle = must ? '#FFCF6E' : '#E6F0EE';
    ctx.fillText(text, p.x, box.y + 2);
  }
  ctx.globalAlpha = 1;
  gminidraw();
}

function gcollides(box) {
  for (const b of G.labelBoxes) {
    if (box.x < b.x + b.w && box.x + box.w > b.x &&
        box.y < b.y + b.h && box.y + box.h > b.y) return true;
  }
  return false;
}

/* ── the minimap ────────────────────────────────────────────────────────────
 * The whole graph at a glance, with the window drawn on it. Zooming into a
 * neighbourhood otherwise costs every landmark, and "fit" is a blunt way back
 * because it throws away the zoom you had chosen. */
let gmini = null, gminictx = null;

function gminidraw() {
  if (!gminictx) return;
  const cv = gmini;
  const w = cv.width, hh = cv.height;
  gminictx.clearRect(0, 0, w, hh);

  const nodes = gliveNodes();
  if (!nodes.length) return;

  let lo_x = Infinity, lo_y = Infinity, hi_x = -Infinity, hi_y = -Infinity;
  for (const n of nodes) {
    if (n.x < lo_x) lo_x = n.x;
    if (n.x > hi_x) hi_x = n.x;
    if (n.y < lo_y) lo_y = n.y;
    if (n.y > hi_y) hi_y = n.y;
  }
  // The viewport is included in the extent, so panning off the nodes still
  // shows the window rather than letting it slide out of the map.
  const tl = gtoWorld(0, 0), br = gtoWorld(gsize.w, gsize.h);
  lo_x = Math.min(lo_x, tl.x); lo_y = Math.min(lo_y, tl.y);
  hi_x = Math.max(hi_x, br.x); hi_y = Math.max(hi_y, br.y);

  const pad = 40;
  const sx = w / Math.max(1, (hi_x - lo_x) + pad * 2);
  const sy = hh / Math.max(1, (hi_y - lo_y) + pad * 2);
  const s = Math.min(sx, sy);
  const ox = (w - (hi_x - lo_x) * s) / 2 - lo_x * s;
  const oy = (hh - (hi_y - lo_y) * s) / 2 - lo_y * s;
  G.mini = { s, ox, oy };

  for (const n of nodes) {
    const x = n.x * s + ox, y = n.y * s + oy;
    gminictx.beginPath();
    gminictx.arc(x, y, Math.max(1.1, n.r * s * 0.9), 0, Math.PI * 2);
    gminictx.fillStyle = gcolor(n);
    gminictx.globalAlpha = (G.sel && G.sel.id === n.id) ? 1 : 0.62;
    gminictx.fill();
  }
  gminictx.globalAlpha = 1;

  gminictx.strokeStyle = '#FFB020';
  gminictx.lineWidth = 1.4;
  gminictx.strokeRect(tl.x * s + ox, tl.y * s + oy,
                      (br.x - tl.x) * s, (br.y - tl.y) * s);
}

/* Clicking the map means "take me there" — the world point under the cursor
   moves to the centre of the canvas, at whatever zoom is already set. */
function gminiseek(ev) {
  if (!G.mini || !gmini) return;
  const rect = gmini.getBoundingClientRect();
  const px = (ev.clientX - rect.left) / rect.width * gmini.width;
  const py = (ev.clientY - rect.top) / rect.height * gmini.height;
  const wx = (px - G.mini.ox) / G.mini.s;
  const wy = (py - G.mini.oy) / G.mini.s;
  G.view.x = gsize.w / 2 - wx * G.view.k;
  G.view.y = gsize.h / 2 - wy * G.view.k;
  gdraw();
}

function groundRect(ctx, x, y, w, hh, r) {
  const rr = Math.min(r, w / 2, hh / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + hh, rr);
  ctx.arcTo(x + w, y + hh, x, y + hh, rr);
  ctx.arcTo(x, y + hh, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/* A reel is shot vertically, so its node is a vertical frame. The shape alone
   says "this one plays" before any label is read. */
function gpaintVideo(ctx, n, p, r) {
  const w = r * 1.5, hh = r * 2.3;
  const x = p.x - w / 2, y = p.y - hh / 2;
  groundRect(ctx, x, y, w, hh, Math.max(2, r * 0.28));
  ctx.fillStyle = '#17292D';
  ctx.fill();

  const key = n.meta && n.meta.video_key;
  const img = key ? gposter(key, r) : null;
  if (img && img !== 'no' && img.complete && img.naturalWidth) {
    ctx.save();
    ctx.clip();
    const scale = Math.max(w / img.naturalWidth, hh / img.naturalHeight);
    const dw = img.naturalWidth * scale, dh = img.naturalHeight * scale;
    ctx.drawImage(img, p.x - dw / 2, p.y - dh / 2, dw, dh);
    ctx.restore();
    groundRect(ctx, x, y, w, hh, Math.max(2, r * 0.28));
  } else if (r > 6) {
    ctx.fillStyle = '#2F5450';
    ctx.beginPath();
    ctx.moveTo(p.x - r * 0.24, p.y - r * 0.34);
    ctx.lineTo(p.x + r * 0.34, p.y);
    ctx.lineTo(p.x - r * 0.24, p.y + r * 0.34);
    ctx.closePath();
    ctx.fill();
    groundRect(ctx, x, y, w, hh, Math.max(2, r * 0.28));
  }
  ctx.strokeStyle = '#E6F0EE';
  ctx.lineWidth = 1.2;
  ctx.stroke();
}

function gpaintEntity(ctx, n, p, r, tone) {
  ctx.beginPath();
  if (n.kind === 'hashtag' || n.kind === 'anchor') {
    // A diamond for author-supplied labels: they are claims about the video,
    // not observations of it, and the shape keeps that distinction visible.
    ctx.moveTo(p.x, p.y - r);
    ctx.lineTo(p.x + r, p.y);
    ctx.lineTo(p.x, p.y + r);
    ctx.lineTo(p.x - r, p.y);
    ctx.closePath();
  } else {
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  }
  ctx.fillStyle = tone + '2E';
  ctx.fill();
  ctx.strokeStyle = tone;
  ctx.lineWidth = n.kind === 'dim' ? 1.9 : 1.2;
  ctx.stroke();
  if (n.kind === 'dim' && r > 9) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, r * 0.36, 0, Math.PI * 2);
    ctx.fillStyle = tone;
    ctx.fill();
  }
}

/* The frame painted inside a reel's node.
 *
 * The threshold used to be r < 11, which sounds harmless and meant that on any
 * graph big enough to need fitting, gfit() left the zoom low enough that every
 * video node fell under it — so the posters never loaded at all and the graph
 * was a field of grey rectangles. It is now low enough that a fitted graph
 * still paints, and instead of a size test doing the rationing there is an
 * explicit limit on requests in flight: a hundred reels on screen must not
 * become a hundred simultaneous channel reads.
 */
const GPOSTER_INFLIGHT_MAX = 6;
let gposterBusy = 0;

function gposter(key, r) {
  if (G.posterOff) return null;
  const held = G.posters.get(key);
  if (held) return held;
  if (r < 5) return null;                    // smaller than the frame's border
  if (gposterBusy >= GPOSTER_INFLIGHT_MAX) return null;   // asked again next draw
  gposterBusy++;
  const img = new Image();
  img.decoding = 'async';
  const done = () => { gposterBusy = Math.max(0, gposterBusy - 1); };
  img.onload = () => { done(); gdraw(); };
  // A 204 means this reel has no poster yet. Remembering that as 'no' is what
  // stops the draw loop asking about it sixty times a second.
  img.onerror = () => { done(); G.posters.set(key, 'no'); };
  G.posters.set(key, img);
  img.src = U(`/api/poster/${encodeURIComponent(key)}`);
  return img;
}

/* ── hit testing ────────────────────────────────────────────────────────── */
function gnodeAt(sx, sy) {
  const world = gtoWorld(sx, sy);
  let best = null, bestD = Infinity;
  for (const n of gliveNodes()) {
    const dx = world.x - n.x, dy = world.y - n.y;
    const d = Math.hypot(dx, dy);
    const reach = n.r + 6 / G.view.k;
    if (d < reach && d < bestD) { best = n; bestD = d; }
  }
  return best;
}

function gedgeAt(sx, sy) {
  let best = null, bestD = 7;
  for (const { e, a, b } of gliveEdges()) {
    const A = gtoScreen(a.x, a.y), B = gtoScreen(b.x, b.y);
    const vx = B.x - A.x, vy = B.y - A.y;
    const len2 = vx * vx + vy * vy;
    if (!len2) continue;
    let t = ((sx - A.x) * vx + (sy - A.y) * vy) / len2;
    t = Math.max(0, Math.min(1, t));
    const d = Math.hypot(sx - (A.x + vx * t), sy - (A.y + vy * t));
    // Not a hit if it is really the node under the cursor.
    if (d < bestD && t > 0.14 && t < 0.86) { best = e; bestD = d; }
  }
  return best;
}

/* ── loading ────────────────────────────────────────────────────────────── */
async function graphBoot(force) {
  if (G.loaded && !force) { gresize(); gheat(0.24); return; }
  G.loaded = true;
  $('graphEmpty').hidden = true;
  try {
    const data = G.mode === 'schema'
      ? await api('/api/graph/schema')
      : await api('/api/graph?limit=18');
    G.nodes.clear(); G.edges.clear();
    G.sel = G.selEdge = G.hover = null;
    G.traceFrom = null;
    G.pathSet = G.pathEdges = null;
    gdetailClose();
    if (data.counts) G.counts = data.counts;
    gmerge(data, { x: 0, y: 0 });
    if (!G.nodes.size) {
      $('graphEmpty').hidden = false;
      $('graphEmpty').textContent = '';
      $('graphEmpty').appendChild(h('h3', { text: 'Nothing to plot yet' }));
      $('graphEmpty').appendChild(h('p', {
        text: data.note || 'Import a bundle from the channel and the graph '
          + 'builds itself from whatever tables arrive.',
      }));
    }
    grenderKinds();
    grenderHud();
    gresize();
    // Settle before the first paint so the opening view is a graph rather
    // than a cloud of dots that then flies apart.
    G.alpha = 1;
    for (let i = 0; i < 170; i++) gtick();
    gfit();
    gheat(0.3);
  } catch (e) {
    $('graphEmpty').hidden = false;
    $('graphEmpty').textContent = '';
    $('graphEmpty').appendChild(h('h3', { text: 'The graph is not ready' }));
    $('graphEmpty').appendChild(h('p', { text: e.message }));
  }
}

async function gexpand(node) {
  if (!node) return;
  try {
    const data = await api(
      `/api/graph/expand/${encodeURIComponent(node.id).replace(/%3A/g, ':')}?limit=48`);
    if (!data.ok) { toast(data.note || 'nothing to expand'); return; }
    const fresh = gmerge(data, node);
    node.expanded = true;
    grenderHud();
    if (!fresh.length) toast('Everything it connects to is already on screen.');
    else if (data.truncated)
      toast(`Added ${fresh.length}. ${fmtInt(data.truncated)} more are not shown.`);
    // Warm the videos that just appeared, so clicking one starts instantly.
    prefetch(fresh.filter(n => n.kind === 'video')
                  .map(n => n.meta && n.meta.video_key).filter(Boolean));
    gheat(0.75);
  } catch (e) { toast(e.message); }
}

/* ── the detail slab ────────────────────────────────────────────────────── */
function gdetailClose() {
  $('graphDetail').hidden = true;
  $('graphDetailBody').textContent = '';
}

function gkindDot(node) {
  return h('i', { class: 'gdot', style: `background:${gcolor(node)}` });
}

async function gselect(node) {
  G.sel = node;
  G.selEdge = null;
  // A trace survives clicking one of its own members — that is how you read
  // the chain — but any other selection means the question has moved on.
  if (G.pathSet && (!node || !G.pathSet.has(node.id)))
    G.pathSet = G.pathEdges = null;
  gdraw();
  if (!node) { gdetailClose(); return; }

  const body = $('graphDetailBody');
  $('graphDetail').hidden = false;
  body.textContent = '';
  body.appendChild(h('div', { class: 'gd-kind' }, gkindDot(node),
    (KIND_LABEL[node.kind] || node.kind) + (node.sub ? ' · ' + node.sub : '')));
  body.appendChild(h('div', { class: 'gd-title', text: node.label }));

  if (node.kind === 'video') { await gvideoDetail(node, body); return; }

  const line = h('div', { class: 'gd-line' });
  line.appendChild(document.createTextNode(
    `${fmtInt(Math.round(node.weight))} connection${node.weight === 1 ? '' : 's'}`));
  line.appendChild(h('span', { class: 'sep', text: '·' }));
  line.appendChild(document.createTextNode(`${node.deg} on screen`));
  body.appendChild(line);

  const acts = h('div', { class: 'gd-acts' });
  // In schema mode there is nothing to expand — the whole schema is already
  // on screen, and the nodes are tables rather than rows.
  if (G.mode !== 'schema')
    acts.appendChild(h('button', {
      class: 'btn', onclick: () => gexpand(node),
    }, node.expanded ? 'Expand again' : 'Expand'));
  acts.appendChild(h('button', {
    class: 'btn btn-quiet', onclick: () => gisolate(node),
  }, 'Focus on this'));
  gtraceButton(node, acts);
  body.appendChild(acts);

  if (node.kind === 'table') { gschemaDetail(node, body); return; }
  if (node.kind === 'anchor') {
    body.appendChild(h('p', {
      class: 'gd-note',
      text: 'Every table that carries a video key joins here. This is the '
          + 'column Atlas uses to tie a row to a reel, and a table with no '
          + 'line to it cannot be searched.',
    }));
    return;
  }

  gneighbourChips(node, body);

  body.appendChild(h('div', { class: 'gd-h', text: 'loading' }));
  let data;
  try {
    data = await api(
      `/api/graph/node/${encodeURIComponent(node.id).replace(/%3A/g, ':')}?rows=40`);
  } catch (e) {
    body.lastChild.remove();
    body.appendChild(h('p', { class: 'gd-note', text: e.message }));
    return;
  }
  if (G.sel !== node) return;                 // the person moved on
  body.lastChild.remove();

  for (const rec of data.records || []) {
    body.appendChild(h('div', { class: 'gd-h', text: `row in ${rec.table}` }));
    if (rec.rows.length === 1) body.appendChild(kvTable(rec.rows[0]));
    else {
      const cols = Object.keys(rec.rows[0] || {});
      const wrap = h('div', { class: 'gd-rows' });
      wrap.appendChild(rowTable(cols, rec.rows.map(r => cols.map(c => r[c])),
        null, rec.table));
      body.appendChild(wrap);
    }
  }

  const vids = data.videos || [];
  if (vids.length) {
    body.appendChild(h('div', { class: 'gd-h', text: `${fmtInt(vids.length)} video${vids.length === 1 ? '' : 's'}` }));
    body.appendChild(gvideoList(vids));
    prefetch(vids.slice(0, 8).map(v => v.video_key));
  }
}

/* What is next to this node, on screen, grouped by what the link is called.
   The chips are the keyboard-and-small-screen path through the graph: the
   canvas is a good way to see structure and a poor way to walk it precisely. */
function gneighbourChips(node, body) {
  const groups = new Map();
  for (const e of G.edges.values()) {
    let other = null;
    if (e.src === node.id) other = G.nodes.get(e.dst);
    else if (e.dst === node.id) other = G.nodes.get(e.src);
    if (!other) continue;
    if (!groups.has(e.rel)) groups.set(e.rel, []);
    groups.get(e.rel).push(other);
  }
  if (!groups.size) return;

  body.appendChild(h('div', { class: 'gd-h', text: 'connected to' }));
  const order = Array.from(groups.entries()).sort((a, b) => b[1].length - a[1].length);
  for (const [rel, list] of order) {
    const wrap = h('div', { class: 'gd-rel' });
    wrap.appendChild(h('div', { class: 'gd-rel-name' },
      rel, ' ', h('span', { class: 'n', text: `(${fmtInt(list.length)})` })));
    const chips = h('div', { class: 'gd-chips' });
    list.sort((a, b) => b.weight - a.weight);
    for (const other of list.slice(0, 40)) {
      chips.appendChild(h('button', {
        class: 'gchip', onclick: () => gfocus(other),
        title: other.sub || KIND_LABEL[other.kind] || other.kind,
      },
        h('i', { class: 'gdot', style: `background:${gcolor(other)}` }),
        other.label));
    }
    if (list.length > 40)
      chips.appendChild(h('span', { class: 'gd-note', text: `+${fmtInt(list.length - 40)} more on the canvas` }));
    wrap.appendChild(chips);
    body.appendChild(wrap);
  }
}

/* Select a node and bring the canvas to it, without changing the zoom. */
function gfocus(node) {
  G.view.x = gsize.w / 2 - node.x * G.view.k;
  G.view.y = gsize.h / 2 - node.y * G.view.k;
  gselect(node);
}

/* "How are these two related?" — pick one node, then another, and the server
   walks the shortest chain between them. Two clicks rather than a form,
   because the second node is usually one you find while looking around. */
function gtraceButton(node, acts) {
  if (G.mode === 'schema') return;
  const armed = G.traceFrom && G.traceFrom !== node.id;
  const from = armed ? G.nodes.get(G.traceFrom) : null;
  acts.appendChild(h('button', {
    class: 'btn btn-quiet',
    title: armed
      ? `Find the shortest chain from ${from ? from.label : 'the first node'}`
      : 'Pick this as one end, then open another node',
    onclick: () => {
      if (armed) { gtrace(G.traceFrom, node.id); return; }
      G.traceFrom = node.id;
      toast('Now open the other node and choose "Connect to this".');
      gselect(node);
    },
  }, armed ? 'Connect to this' : 'Trace from here'));
  if (armed)
    acts.appendChild(h('button', {
      class: 'btn btn-quiet', onclick: () => { G.traceFrom = null; gselect(node); },
    }, 'Cancel trace'));
}

async function gtrace(a, b) {
  G.traceFrom = null;
  let data;
  try {
    data = await api('/api/graph/path?' + new URLSearchParams({ a, b, depth: '6' }));
  } catch (e) { toast(e.message); return; }
  if (!data.ok) { toast(data.note || 'no connection found'); return; }

  gmerge(data, { x: 0, y: 0 });
  G.pathSet = new Set(data.path || []);
  G.pathEdges = new Set((data.edges || [])
    .filter(e => G.pathSet.has(e.src) && G.pathSet.has(e.dst))
    .map(gkey));
  // A hop that runs through a hidden kind would draw as a broken chain.
  for (const id of G.pathSet) {
    const n = G.nodes.get(id);
    if (n) G.off.delete(n.kind);
  }
  grenderKinds();
  grenderHud();

  const body = $('graphDetailBody');
  $('graphDetail').hidden = false;
  body.textContent = '';
  body.appendChild(h('div', { class: 'gd-kind' },
    h('i', { class: 'gdot', style: 'background:#FFB020' }), 'connection found'));
  const chain = (data.nodes || []);
  body.appendChild(h('div', { class: 'gd-title' },
    `${chain.length - 1} step${chain.length === 2 ? '' : 's'} apart`));
  const walk = h('div', { class: 'gd-chips' });
  chain.forEach((raw, i) => {
    if (i) walk.appendChild(h('span', { class: 'gd-arrow', text: '→' }));
    const live = G.nodes.get(raw.id) || raw;
    walk.appendChild(h('button', {
      class: 'gchip', onclick: () => { if (live.x !== undefined) gfocus(live); },
    }, h('i', { class: 'gdot', style: `background:${gcolor(live)}` }), live.label));
  });
  body.appendChild(walk);
  body.appendChild(h('div', { class: 'gd-acts' },
    h('button', {
      class: 'btn btn-quiet',
      onclick: () => { G.pathSet = G.pathEdges = null; gdetailClose(); gdraw(); },
    }, 'Clear the trace')));

  G.sel = null; G.selEdge = null;
  gheat(0.6);
}

function gvideoList(rows) {
  const box = h('div', { class: 'gd-vids' });
  for (const row of rows) {
    const btn = h('button', {
      class: 'gd-vid', onclick: () => {
        openLibraryVideo(row);
        // Bring the node onto the canvas too, so playing from the slab and
        // clicking the graph leave you in the same place.
        const node = G.nodes.get('v:' + row.video_key);
        if (node) { G.sel = node; gdraw(); }
      },
    });
    const img = h('img', {
      alt: '', loading: 'lazy',
      src: U(`/api/poster/${encodeURIComponent(row.video_key)}`),
      onerror: (ev) => { ev.target.replaceWith(h('span', { class: 'gv-blank' })); },
    });
    btn.appendChild(img);
    const meta = h('div');
    meta.appendChild(h('div', { class: 'gv-t', text: row.title || row.caption || row.video_key }));
    const bits = [];
    if (row.creator) bits.push(row.creator);
    if (row.duration) bits.push(timecode(row.duration));
    if (row.moment_count) bits.push(`${fmtInt(row.moment_count)} moments`);
    meta.appendChild(h('div', { class: 'gv-s', text: bits.join(' · ') }));
    btn.appendChild(meta);
    box.appendChild(btn);
  }
  return box;
}

async function gvideoDetail(node, body) {
  const key = node.meta && node.meta.video_key;
  const line = h('div', { class: 'gd-line' });
  const bits = [];
  if (node.meta.duration) bits.push(timecode(node.meta.duration));
  if (node.meta.moments) bits.push(`${fmtInt(node.meta.moments)} moments`);
  if (node.meta.likes) bits.push(`${fmtInt(node.meta.likes)} likes`);
  if (node.meta.created_at) bits.push(fmtWhen(node.meta.created_at));
  bits.forEach((b, i) => {
    if (i) line.appendChild(h('span', { class: 'sep', text: '·' }));
    line.appendChild(document.createTextNode(b));
  });
  body.appendChild(line);

  const acts = h('div', { class: 'gd-acts' });
  acts.appendChild(h('button', {
    class: 'btn', onclick: () => gplay(node),
  }, 'Play'));
  acts.appendChild(h('button', {
    class: 'btn btn-quiet', onclick: () => gexpand(node),
  }, 'What is in it'));
  acts.appendChild(h('button', {
    class: 'btn btn-quiet', onclick: () => gisolate(node),
  }, 'Focus on this'));
  gtraceButton(node, acts);
  body.appendChild(acts);

  // Everything the archive knows about this reel, from the same endpoint the
  // player's Database record panel reads — one source of truth, two views.
  body.appendChild(h('div', { class: 'gd-h', text: 'loading the record' }));
  try {
    const data = await api(`/api/video/${encodeURIComponent(key)}`);
    if (!G.sel || G.sel.id !== node.id) return;
    body.lastChild.remove();
    if (data.meta && Object.keys(data.meta).length) {
      body.appendChild(h('div', { class: 'gd-h', text: 'video' }));
      body.appendChild(kvTable(data.meta));
    }
    for (const rel of data.related || []) {
      if (!rel.rows || !rel.rows.length) continue;
      body.appendChild(h('div', { class: 'gd-h' }, rel.table,
        h('span', { class: 'rec-count', text: `${fmtInt(rel.rows.length)} row${rel.rows.length === 1 ? '' : 's'}` })));
      if (rel.rows.length === 1) { body.appendChild(kvTable(rel.rows[0])); continue; }
      const wrap = h('div', { class: 'gd-rows' });
      wrap.appendChild(rowTable(rel.columns,
        rel.rows.map(r => rel.columns.map(c => r[c])), null, rel.table));
      body.appendChild(wrap);
    }
  } catch (e) {
    if (body.lastChild) body.lastChild.remove();
    body.appendChild(h('p', { class: 'gd-note', text: e.message }));
  }
}

function gplay(node) {
  const m = node.meta || {};
  openLibraryVideo({
    video_key: m.video_key, title: node.label, caption: '',
    creator: '', category: '', duration: m.duration,
    likes: m.likes, created_at: m.created_at, msg_id: m.msg_id,
    moment_count: m.moments,
  });
}

function gschemaDetail(node, body) {
  const m = node.meta || {};
  body.appendChild(h('div', { class: 'gd-h', text: 'what Atlas found here' }));
  body.appendChild(kvTable({
    rows: fmtInt(m.rows),
    'video key': m.key || '— none, so this table is not indexed',
    'timeline start': m.start || '—',
    'timeline end': m.end || '—',
    'searchable text': (m.content || []).join(', ') || '—',
    columns: (m.columns || []).length,
  }));
  body.appendChild(h('div', { class: 'gd-acts' },
    h('button', {
      class: 'btn',
      onclick: () => { showTab('data'); openTable(node.label, 0, ''); },
    }, 'Browse the rows')));
}

async function gedgeSelect(edge) {
  G.selEdge = edge;
  G.sel = null;
  gdraw();
  const body = $('graphDetailBody');
  $('graphDetail').hidden = false;
  body.textContent = '';
  const a = G.nodes.get(edge.src), b = G.nodes.get(edge.dst);
  body.appendChild(h('div', { class: 'gd-kind' },
    h('i', { class: 'gdot', style: 'background:#FFB020' }), 'connection'));
  body.appendChild(h('div', { class: 'gd-title' },
    `${a ? a.label : edge.src} → ${b ? b.label : edge.dst}`));

  const why = h('div', { class: 'gd-why' });
  const parts = (edge.ref || '').split('|');
  why.appendChild(document.createTextNode('Linked by '));
  why.appendChild(h('b', { text: edge.rel }));
  if (parts.length >= 2) {
    why.appendChild(document.createTextNode(', read from '));
    why.appendChild(h('code', { text: `${parts[0]}.${parts[1]}` }));
  }
  why.appendChild(document.createTextNode('.'));
  body.appendChild(why);

  const acts = h('div', { class: 'gd-acts' });
  if (a) acts.appendChild(h('button', { class: 'btn btn-quiet', onclick: () => gselect(a) }, a.label.slice(0, 22)));
  if (b) acts.appendChild(h('button', { class: 'btn btn-quiet', onclick: () => gselect(b) }, b.label.slice(0, 22)));
  body.appendChild(acts);

  body.appendChild(h('div', { class: 'gd-h', text: 'the rows behind it' }));
  if (G.mode === 'schema') {
    // A schema edge is a statement about columns, not about rows: there is
    // nothing to fetch, and the Data tab is the right place to go next.
    body.appendChild(h('p', {
      class: 'gd-note',
      text: 'This line is a join Atlas inferred from the column names, not a '
          + 'stored row. Switch to the data graph to walk the actual values.',
    }));
    return;
  }
  try {
    const data = await api('/api/graph/edge?' + new URLSearchParams({
      src: edge.src, dst: edge.dst, rel: edge.rel,
    }));
    if (G.selEdge !== edge) return;
    const recs = data.records || [];
    if (!recs.length) {
      body.appendChild(h('p', {
        class: 'gd-note',
        text: 'The link is derived rather than stored, so there is no single '
            + 'row to show for it.',
      }));
      return;
    }
    for (const rec of recs) {
      const cols = Object.keys(rec.rows[0] || {});
      const wrap = h('div', { class: 'gd-rows' });
      wrap.appendChild(rowTable(cols, rec.rows.map(r => cols.map(c => r[c])),
        null, rec.table));
      body.appendChild(wrap);
    }
  } catch (e) {
    body.appendChild(h('p', { class: 'gd-note', text: e.message }));
  }
}

/* Keep a node and its neighbours; drop the rest. The fastest way out of a
   graph that has grown past what you can read. */
function gisolate(node) {
  const keep = gneighbourSet(node.id);
  keep.add(node.id);
  for (const id of Array.from(G.nodes.keys())) if (!keep.has(id)) G.nodes.delete(id);
  for (const [k, e] of Array.from(G.edges)) {
    if (!keep.has(e.src) || !keep.has(e.dst)) G.edges.delete(k);
  }
  gdegree();
  grenderHud();
  G.alpha = 1;
  for (let i = 0; i < 90; i++) gtick();
  gfit();
  gheat(0.4);
}

/* ── rail ───────────────────────────────────────────────────────────────── */
function grenderKinds() {
  const box = $('graphKinds');
  box.textContent = '';
  const tally = new Map();
  for (const n of G.nodes.values()) tally.set(n.kind, (tally.get(n.kind) || 0) + 1);
  const order = ['video', 'dim', 'tag', 'hashtag', 'table', 'anchor'];
  for (const kind of order) {
    if (!tally.has(kind)) continue;
    const on = !G.off.has(kind);
    box.appendChild(h('button', {
      class: 'gkind', 'aria-pressed': String(on),
      title: `Show or hide ${KIND_LABEL[kind] || kind} nodes`,
      onclick: (ev) => {
        if (G.off.has(kind)) G.off.delete(kind); else G.off.add(kind);
        ev.currentTarget.setAttribute('aria-pressed', String(!G.off.has(kind)));
        grenderHud();
        gheat(0.4);
      },
    },
      h('i', { class: 'gdot', style: `background:${KIND_COLOR[kind]}` }),
      KIND_LABEL[kind] || kind,
      h('span', { class: 'n', text: fmtInt(tally.get(kind)) })));
  }
}

function grenderHud() {
  const hud = $('graphHud');
  hud.textContent = '';
  const shown = gliveNodes().length;
  const edges = gliveEdges().length;
  hud.appendChild(h('div', {}, h('b', { text: fmtInt(shown) }),
    ' nodes · ', h('b', { text: fmtInt(edges) }), ' links on screen'));
  if (G.counts && G.counts.nodes)
    hud.appendChild(h('div', {
      text: `${fmtInt(G.counts.nodes)} nodes and ${fmtInt(G.counts.edges)} links derived`,
    }));

  const legend = $('graphLegend');
  legend.textContent = '';
  legend.appendChild(h('div', {
    text: 'click a node to inspect · double-click to expand · click a line to see why',
  }));
  legend.appendChild(h('div', {
    text: 'drag to move · scroll to zoom · shift-drag a node to pin it',
  }));
  grenderLegend();
}

/* A continuous colour with no key is decoration. When a scale is on, the
   legend says what its two ends mean in the units a reader actually has. */
function grenderLegend() {
  const legend = $('graphLegend');
  const old = legend.querySelector('.glegend-scale');
  if (old) old.remove();
  if (G.colorBy === 'kind' || G.colorBy === 'source') return;

  const d = G.scale && G.scale[G.colorBy];
  const row = h('div', { class: 'glegend-scale' });
  const label = G.colorBy === 'reach' ? 'links' : 'date';
  const fmt = (v) => G.colorBy === 'recency' ? (fmtWhen(v) || '—') : fmtInt(v);
  row.appendChild(h('span', { class: 'gls-t',
                              text: d ? fmt(d.lo) : 'no value' }));
  const bar = h('span', { class: 'gls-bar' });
  for (let i = 0; i < 24; i++)
    bar.appendChild(h('i', { style: `background:${gramp(i / 23)}` }));
  row.appendChild(bar);
  row.appendChild(h('span', { class: 'gls-t', text: d ? fmt(d.hi) : '' }));
  row.appendChild(h('span', { class: 'gls-l', text: label }));
  legend.appendChild(row);
}

let graphFindTimer = 0;
async function grunFind(value) {
  const q = (value || '').trim();
  const box = $('graphHits');
  if (!q) { box.hidden = true; box.textContent = ''; G.hits = []; return; }
  try {
    const data = await api('/api/graph/find?q=' + encodeURIComponent(q) + '&limit=24');
    G.hits = data.results || [];
    G.hitIndex = -1;
    box.textContent = '';
    if (!G.hits.length) {
      box.appendChild(h('div', { class: 'ghit', text: 'nothing by that name' }));
    } else {
      G.hits.forEach((n, i) => {
        box.appendChild(h('button', {
          class: 'ghit', role: 'option', 'data-i': i,
          onclick: () => gjumpTo(n),
        },
          h('i', { class: 'gdot', style: `background:${gcolor(n)}` }),
          h('span', { class: 'glabel', text: n.label }),
          h('span', { class: 'gsub', text: n.sub || n.kind })));
      });
    }
    box.hidden = false;
  } catch { box.hidden = true; }
}

async function gjumpTo(raw) {
  $('graphHits').hidden = true;
  $('graphQ').value = '';
  let node = G.nodes.get(raw.id);
  if (!node) {
    // Not on screen yet: pull it in with its neighbourhood so it lands in
    // context rather than as a lone dot in the middle of nowhere.
    try {
      const data = await api(
        `/api/graph/expand/${encodeURIComponent(raw.id).replace(/%3A/g, ':')}?limit=36`);
      if (data.ok) {
        gmerge({ nodes: [data.centre], edges: [] }, { x: 0, y: 0 });
        node = G.nodes.get(raw.id);
        if (node) { node.x = 0; node.y = 0; node.expanded = true; }
        gmerge(data, node || { x: 0, y: 0 });
      }
    } catch (e) { toast(e.message); return; }
  }
  node = G.nodes.get(raw.id);
  if (!node) return;
  G.off.delete(node.kind);
  grenderKinds();
  grenderHud();
  // Centre on it without changing the zoom, which would lose the reader's
  // sense of where they were.
  G.view.x = gsize.w / 2 - node.x * G.view.k;
  G.view.y = gsize.h / 2 - node.y * G.view.k;
  gselect(node);
  gheat(0.5);
}

/* ── pointer and keys ───────────────────────────────────────────────────── */
function gresize() {
  if (!gcv) return;
  const rect = gcv.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  gsize.dpr = Math.min(2, window.devicePixelRatio || 1);
  gsize.w = rect.width;
  gsize.h = rect.height;
  gcv.width = Math.round(rect.width * gsize.dpr);
  gcv.height = Math.round(rect.height * gsize.dpr);
  gdraw();
}

function gwire() {
  gcv = $('graphCanvas');
  if (!gcv) return;
  gctx = gcv.getContext('2d');

  const ro = new ResizeObserver(() => gresize());
  ro.observe($('graphStage'));

  gcv.addEventListener('pointerdown', (ev) => {
    gcv.setPointerCapture(ev.pointerId);
    G.moved = 0;
    const rect = gcv.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const node = gnodeAt(sx, sy);
    if (node) {
      G.drag = { node, dx: 0, dy: 0, pin: ev.shiftKey };
      const w = gtoWorld(sx, sy);
      G.drag.dx = node.x - w.x;
      G.drag.dy = node.y - w.y;
    } else {
      G.pan = { x: ev.clientX - G.view.x, y: ev.clientY - G.view.y };
    }
  });

  gcv.addEventListener('pointermove', (ev) => {
    const rect = gcv.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    if (G.drag) {
      G.moved += Math.abs(ev.movementX) + Math.abs(ev.movementY);
      const w = gtoWorld(sx, sy);
      G.drag.node.x = w.x + G.drag.dx;
      G.drag.node.y = w.y + G.drag.dy;
      G.drag.node.vx = G.drag.node.vy = 0;
      gheat(0.34);
      return;
    }
    if (G.pan) {
      G.moved += Math.abs(ev.movementX) + Math.abs(ev.movementY);
      G.view.x = ev.clientX - G.pan.x;
      G.view.y = ev.clientY - G.pan.y;
      gdraw();
      return;
    }
    const node = gnodeAt(sx, sy);
    const edge = node ? null : gedgeAt(sx, sy);
    const changed = (node !== G.hover) ||
      (gkey(edge || { src: '', dst: '', rel: '' }) !==
       gkey(G.hoverEdge || { src: '', dst: '', rel: '' }));
    G.hover = node;
    G.hoverEdge = edge;
    gcv.dataset.over = node || edge ? 'node' : '';
    gcv.title = node ? `${node.label}${node.sub ? ' — ' + node.sub : ''}` : '';
    if (changed) gdraw();
  });

  const release = (ev) => {
    const rect = gcv.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const wasDrag = G.drag, moved = G.moved;
    if (G.drag) {
      if (G.drag.pin) G.drag.node.pin = true;
      G.drag = null;
      gheat(0.22);
    }
    G.pan = null;
    G.moved = 0;
    if (moved > 5) return;                    // a drag is not a click
    const node = wasDrag ? wasDrag.node : gnodeAt(sx, sy);
    if (node) { gselect(node); return; }
    const edge = gedgeAt(sx, sy);
    if (edge) { gedgeSelect(edge); return; }
    G.sel = null; G.selEdge = null;
    gdetailClose();
    gdraw();
  };
  gcv.addEventListener('pointerup', release);
  gcv.addEventListener('pointercancel', () => { G.drag = null; G.pan = null; });

  gcv.addEventListener('dblclick', (ev) => {
    const rect = gcv.getBoundingClientRect();
    const node = gnodeAt(ev.clientX - rect.left, ev.clientY - rect.top);
    if (node) gexpand(node);
  });

  gcv.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const rect = gcv.getBoundingClientRect();
    const sx = ev.clientX - rect.left, sy = ev.clientY - rect.top;
    const before = gtoWorld(sx, sy);
    const step = Math.exp(-ev.deltaY * 0.0016);
    G.view.k = Math.max(0.08, Math.min(4.2, G.view.k * step));
    // Zoom toward the cursor: the point under the pointer must not move.
    const after = gtoWorld(sx, sy);
    G.view.x += (after.x - before.x) * G.view.k;
    G.view.y += (after.y - before.y) * G.view.k;
    gdraw();
  }, { passive: false });

  gcv.addEventListener('keydown', (ev) => {
    const step = ev.shiftKey ? 120 : 46;
    if (ev.key === 'ArrowLeft') { G.view.x += step; gdraw(); }
    else if (ev.key === 'ArrowRight') { G.view.x -= step; gdraw(); }
    else if (ev.key === 'ArrowUp') { G.view.y += step; gdraw(); }
    else if (ev.key === 'ArrowDown') { G.view.y -= step; gdraw(); }
    else if (ev.key === '+' || ev.key === '=') { G.view.k = Math.min(4.2, G.view.k * 1.2); gdraw(); }
    else if (ev.key === '-') { G.view.k = Math.max(0.08, G.view.k / 1.2); gdraw(); }
    else if (ev.key === 'Enter' && G.sel) gexpand(G.sel);
    else if (ev.key === 'Escape') {
      G.sel = null; G.selEdge = null; G.traceFrom = null;
      G.pathSet = G.pathEdges = null;
      gdetailClose(); gdraw();
    }
    else return;
    ev.preventDefault();
  });

  $('graphFit').addEventListener('click', () => { gfit(); gdraw(); });

  // Zoom about the middle of the stage, which is what a button press means —
  // the wheel handler zooms about the pointer, which is what a wheel means.
  const gzoom = (factor) => {
    const cx = gsize.w / 2, cy = gsize.h / 2;
    const before = gtoWorld(cx, cy);
    G.view.k = Math.max(0.08, Math.min(4.2, G.view.k * factor));
    const after = gtoWorld(cx, cy);
    G.view.x += (after.x - before.x) * G.view.k;
    G.view.y += (after.y - before.y) * G.view.k;
    gdraw();
  };
  $('graphIn').addEventListener('click', () => gzoom(1.28));
  $('graphOut').addEventListener('click', () => gzoom(1 / 1.28));

  // Frames off is for reading structure: the shapes and colours stay, the
  // imagery stops competing with them.
  $('graphPosters').addEventListener('click', (ev) => {
    G.posterOff = !G.posterOff;
    ev.currentTarget.setAttribute('aria-pressed', String(!G.posterOff));
    ev.currentTarget.classList.toggle('off', G.posterOff);
    gdraw();
  });
  $('graphClear').addEventListener('click', () => {
    G.loaded = false;
    G.traceFrom = null;
    G.pathSet = G.pathEdges = null;
    G.off.clear();
    graphBoot(true);
  });
  $('graphFreeze').addEventListener('click', (ev) => {
    G.frozen = !G.frozen;
    ev.currentTarget.setAttribute('aria-pressed', String(G.frozen));
    ev.currentTarget.textContent = G.frozen ? 'Resume' : 'Freeze';
    if (!G.frozen) gheat(0.3);
  });
  $('graphDetailClose').addEventListener('click', () => {
    G.sel = null; G.selEdge = null; gdetailClose(); gdraw();
  });

  $$('.gmode').forEach(b => b.addEventListener('click', () => {
    if (G.mode === b.dataset.mode) return;
    G.mode = b.dataset.mode;
    $$('.gmode').forEach(x => x.classList.toggle('on', x === b));
    G.loaded = false;
    G.off.clear();
    graphBoot(true);
  }));

  /* ── how the graph is encoded ──
   * All five of these read nodes already on screen, so none of them fetches
   * anything. Colour repaints; size has to settle, because the spring rest
   * length is a function of radius. */
  $('graphColorBy').addEventListener('change', (ev) => {
    G.colorBy = ev.target.value;
    gscale();
    grenderLegend();
    gdraw();
  });
  $('graphSizeBy').addEventListener('change', (ev) => {
    G.sizeBy = ev.target.value;
    gre_encode();
  });
  for (const [id, key] of [['graphSpread', 'spread'], ['graphPull', 'pull']]) {
    $(id).addEventListener('input', (ev) => {
      G[key] = Number(ev.target.value) || 3;
      // A layout change has to be re-simulated to be seen at all.
      gheat(0.6);
    });
  }
  $('graphLabels').addEventListener('input', (ev) => {
    G.labels = Number(ev.target.value) || 3;
    gdraw();   // labels are a paint decision, nothing moves
  });

  gmini = $('graphMini');
  if (gmini) {
    gminictx = gmini.getContext('2d');
    let miniDown = false;
    gmini.addEventListener('pointerdown', (ev) => {
      miniDown = true;
      gmini.setPointerCapture(ev.pointerId);
      gminiseek(ev);
    });
    // Dragging scrubs the view rather than requiring a click per hop.
    gmini.addEventListener('pointermove', (ev) => { if (miniDown) gminiseek(ev); });
    gmini.addEventListener('pointerup', () => { miniDown = false; });
    gmini.addEventListener('pointercancel', () => { miniDown = false; });
  }

  $('graphQ').addEventListener('input', (ev) => {
    clearTimeout(graphFindTimer);
    const value = ev.target.value;
    graphFindTimer = setTimeout(() => grunFind(value), 180);
  });
  $('graphQ').addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') { $('graphHits').hidden = true; return; }
    if (ev.key === 'Enter') {
      ev.preventDefault();
      const pick = G.hits[Math.max(0, G.hitIndex)];
      if (pick) gjumpTo(pick);
      return;
    }
    if (ev.key !== 'ArrowDown' && ev.key !== 'ArrowUp') return;
    ev.preventDefault();
    if (!G.hits.length) return;
    G.hitIndex = (G.hitIndex + (ev.key === 'ArrowDown' ? 1 : -1) + G.hits.length) % G.hits.length;
    $$('#graphHits .ghit').forEach((el, i) =>
      el.setAttribute('aria-selected', String(i === G.hitIndex)));
  });
  document.addEventListener('click', (ev) => {
    if (!$('graphRail').contains(ev.target)) $('graphHits').hidden = true;
  });
}

/* The bridge from search: plot what the current results have in common. */
async function graphFromResults() {
  const keys = S.results.map(r => r.video_key).filter(Boolean).slice(0, 24);
  if (!keys.length) return;
  // Claim the tab before switching, so showTab does not fetch the overview
  // graph we are about to throw away.
  G.loaded = true;
  G.mode = 'data';
  $$('.gmode').forEach(x => x.classList.toggle('on', x.dataset.mode === 'data'));
  showTab('graph');
  try {
    const data = await api('/api/graph/from?keys=' + encodeURIComponent(keys.join(',')));
    G.nodes.clear(); G.edges.clear();
    G.sel = null; G.selEdge = null;
    gdetailClose();
    gmerge(data, { x: 0, y: 0 });
    G.loaded = true;
    grenderKinds();
    grenderHud();
    gresize();
    G.alpha = 1;
    for (let i = 0; i < 160; i++) gtick();
    gfit();
    gheat(0.3);
  } catch (e) { toast(e.message); }
}

/* ════════════════════════════════════════════════════════════════════════
   MAPS — the archive as one picture

   Three views over one dataset. Two of them (semantic, cluster) share a
   projection computed on the server; the third (scatter) plots two numeric
   columns and needs no embedding at all, which is why there is always a map
   to look at even on a fresh archive.

   Everything is drawn to one canvas because the point count is the whole
   archive — 180k DOM nodes is not a thing a browser will do, and 180k canvas
   arcs is about eight milliseconds. The cost of canvas is that nothing is
   clickable for free, so hit-testing is a uniform grid built once per data
   load: a lookup is O(points in one cell) rather than O(all points), which is
   what keeps the hover readout live while the pointer moves.
   ════════════════════════════════════════════════════════════════════════ */
const M = {
  map: 'semantic',        // semantic | cluster | scatter
  level: 'video',         // video | moment
  loaded: false,
  // Which dataset is currently in M.xy. Semantic and cluster share one, the
  // scatter plot has its own, and both fill the same arrays — so the map name
  // alone cannot say whether a repaint is safe or a refetch is needed.
  dataset: '',            // '' | projection | scatter
  xy: null,               // Float32Array, 2 per point
  cluster: null,          // Int32Array
  refs: null, keys: null, times: null,
  clusters: [],           // legend, from /api/map
  method: '',
  // scatter
  axes: [], sx: '', sy: '', slogx: false, slogy: false,
  scatter: null,          // the /api/map/scatter payload
  // view transform
  k: 1, tx: 0, ty: 0,
  hover: -1, picked: -1,
  muted: new Set(),       // cluster ids toggled off in the legend
  grid: null, gridN: 0,
  drag: null, box: null,
  raf: 0,
};

/* The cluster palette. Built from the seven channel hues the rest of Atlas
 * already uses, then rotated through lightness so that a large archive with
 * forty clusters still reads as distinct without inventing colours that mean
 * something else somewhere else. */
const MAP_HUES = [199, 48, 142, 258, 325, 0, 172, 220, 30, 100, 280, 350];
function clusterColor(c, alpha) {
  if (c === null || c === undefined || c < 0) return `rgba(139,147,158,${alpha})`;
  const hue = MAP_HUES[c % MAP_HUES.length];
  const light = 62 + ((Math.floor(c / MAP_HUES.length) % 3) * 11);
  return `hsla(${hue},72%,${light}%,${alpha})`;
}

const mcanvas = () => $('mapsCanvas');

function mapsShowEmpty(title, note) {
  const box = $('mapsEmpty');
  box.hidden = false;
  box.textContent = '';
  box.appendChild(h('h3', { text: title }));
  if (note) box.appendChild(h('p', { text: note }));
  $('mapsViewport').style.visibility = 'hidden';
}

function mapsHideEmpty() {
  $('mapsEmpty').hidden = true;
  $('mapsViewport').style.visibility = '';
}

/* ── loading ───────────────────────────────────────────────────────────── */
async function mapsBoot(force) {
  // The canvas has no size until its view is visible, so measurement and the
  // first fit can only happen after the tab switch — same reason as the graph.
  if (M.loaded && !force) { mapsResize(); mapsDraw(); return; }
  M.loaded = true;
  if (M.map === 'scatter') { await mapsLoadScatter(); return; }
  await mapsLoadProjection();
}

async function mapsLoadProjection() {
  try {
    const meta = await api(`/api/map?level=${M.level}`);
    M.clusters = meta.clusters || [];
    M.method = meta.method || '';
    if (!meta.count) {
      mapsShowEmpty('No map yet',
        meta.note || 'The semantic and cluster maps are drawn from the dense ' +
        'index. Once the encoder has run they appear here. The scatter plot ' +
        'works without it.');
      mapsRenderLegend();
      return;
    }

    // The points come back as a packed buffer and the labels as JSON, in the
    // same row order. Fetched together because neither is useful alone.
    const [buf, refs] = await Promise.all([
      fetch(U(`/api/map/points?level=${M.level}`)).then(r => r.arrayBuffer()),
      api(`/api/map/refs?level=${M.level}`),
    ]);
    const view = new DataView(buf);
    const n = Math.floor(buf.byteLength / 12);
    M.xy = new Float32Array(n * 2);
    M.cluster = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      M.xy[i * 2] = view.getFloat32(i * 12, true);
      M.xy[i * 2 + 1] = view.getFloat32(i * 12 + 4, true);
      M.cluster[i] = view.getInt32(i * 12 + 8, true);
    }
    M.refs = refs.refs || [];
    M.keys = refs.keys || [];
    M.times = refs.t || [];
    M.dataset = 'projection';

    mapsHideEmpty();
    mapsBuildGrid();
    mapsRenderLegend();
    mapsRenderNote(meta);
    mapsResize();
    mapsFit();
  } catch (e) {
    mapsShowEmpty('The map could not load', String(e.message || e));
  }
}

function mapsRenderNote(meta) {
  const how = {
    umap: 'UMAP — neighbourhoods and global structure preserved',
    tsne: 't-SNE — local structure preserved, distances between far groups ' +
          'are not meaningful',
    pca: 'PCA — a linear projection: it shows the big splits and blurs the ' +
         'fine ones',
  }[M.method] || M.method;
  const n = M.refs ? M.refs.length : 0;
  $('mapsNote').textContent = n
    ? `${fmtInt(n)} ${M.level === 'video' ? 'videos' : 'passages'} · ${how}`
    : '';
}

async function mapsLoadScatter() {
  try {
    if (!M.axes.length) {
      const got = await api('/api/map/axes');
      M.axes = got.axes || [];
      if (!M.axes.length) {
        mapsShowEmpty('Nothing numeric to plot',
          'The scatter plot needs at least two numeric columns in ' +
          'video_index. Import a bundle and it will fill in.');
        return;
      }
      // Default to the pair that says the most about a video archive, but
      // only if the columns actually arrived in this bundle.
      const have = new Set(M.axes.map(a => a.name));
      M.sx = have.has('duration') ? 'duration' : M.axes[0].name;
      M.sy = have.has('moment_count') ? 'moment_count'
           : (M.axes[1] || M.axes[0]).name;
    }
    const p = new URLSearchParams({
      x: M.sx, y: M.sy, colour: 'cluster',
      log_x: String(M.slogx), log_y: String(M.slogy),
    });
    M.scatter = await api('/api/map/scatter?' + p.toString());
    if (!M.scatter.count) {
      mapsShowEmpty('No points to plot',
        M.scatter.note || 'Every video is missing one of the two columns.');
      mapsRenderLegend();
      return;
    }
    mapsHideEmpty();
    mapsProjectScatter();
    mapsRenderLegend();
    $('mapsNote').textContent =
      `${fmtInt(M.scatter.count)} videos · ${M.scatter.x_label} against ` +
      `${M.scatter.y_label}` + (M.scatter.note ? ` · ${M.scatter.note}` : '');
    mapsResize();
    mapsFit();
  } catch (e) {
    mapsShowEmpty('The scatter plot could not load', String(e.message || e));
  }
}

/* The scatter payload arrives in data units — seconds, megabytes, counts — so
 * it is normalised into the same [0,1] space the projection uses. That way one
 * draw path, one hit-test and one zoom implementation serve all three maps,
 * and only the axis furniture differs. */
function mapsProjectScatter() {
  const pts = M.scatter.points;
  let lo0 = Infinity, hi0 = -Infinity, lo1 = Infinity, hi1 = -Infinity;
  for (const p of pts) {
    if (p.x < lo0) lo0 = p.x; if (p.x > hi0) hi0 = p.x;
    if (p.y < lo1) lo1 = p.y; if (p.y > hi1) hi1 = p.y;
  }
  const sp0 = (hi0 - lo0) || 1, sp1 = (hi1 - lo1) || 1;
  M.scatter.domain = { lo0, hi0, lo1, hi1 };
  M.xy = new Float32Array(pts.length * 2);
  M.cluster = new Int32Array(pts.length);
  M.refs = []; M.keys = []; M.times = [];
  pts.forEach((p, i) => {
    M.xy[i * 2] = (p.x - lo0) / sp0;
    // Screen y grows downward and a chart's y grows upward, so this is
    // flipped here rather than in the draw, where it would have to be undone
    // again for hit-testing.
    M.xy[i * 2 + 1] = 1 - (p.y - lo1) / sp1;
    M.cluster[i] = p.g;
    M.refs.push(p.key); M.keys.push(p.key); M.times.push(null);
  });
  M.dataset = 'scatter';
  mapsBuildGrid();
}

/* ── hit testing ───────────────────────────────────────────────────────── */
/* A uniform grid over the unit square. Rebuilt only when the data changes,
 * never on zoom: zoom is a transform applied at draw time, so the grid stays
 * valid and a pan costs nothing. */
function mapsBuildGrid() {
  const n = M.xy ? M.xy.length / 2 : 0;
  const cells = Math.max(8, Math.min(160, Math.ceil(Math.sqrt(n / 3))));
  M.gridN = cells;
  M.grid = new Map();
  for (let i = 0; i < n; i++) {
    const cx = Math.min(cells - 1, Math.max(0, Math.floor(M.xy[i * 2] * cells)));
    const cy = Math.min(cells - 1, Math.max(0, Math.floor(M.xy[i * 2 + 1] * cells)));
    const id = cy * cells + cx;
    let bucket = M.grid.get(id);
    if (!bucket) { bucket = []; M.grid.set(id, bucket); }
    bucket.push(i);
  }
}

function mapsPick(px, py) {
  if (!M.grid || !M.xy) return -1;
  const cells = M.gridN;
  const u = (px - M.tx) / M.k, v = (py - M.ty) / M.k;
  const reach = 10 / M.k;                     // 10 screen px, in data units
  const c0 = Math.max(0, Math.floor((u - reach) * cells));
  const c1 = Math.min(cells - 1, Math.floor((u + reach) * cells));
  const r0 = Math.max(0, Math.floor((v - reach) * cells));
  const r1 = Math.min(cells - 1, Math.floor((v + reach) * cells));
  let best = -1, bestD = reach * reach;
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      const bucket = M.grid.get(r * cells + c);
      if (!bucket) continue;
      for (const i of bucket) {
        if (M.muted.has(M.cluster[i])) continue;
        const dx = M.xy[i * 2] - u, dy = M.xy[i * 2 + 1] - v;
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; best = i; }
      }
    }
  }
  return best;
}

/* ── drawing ───────────────────────────────────────────────────────────── */
function mapsResize() {
  const cv = mcanvas();
  const box = $('mapsViewport').getBoundingClientRect();
  if (!box.width || !box.height) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(box.width * dpr);
  cv.height = Math.round(box.height * dpr);
  cv.style.width = box.width + 'px';
  cv.style.height = box.height + 'px';
  M.dpr = dpr;
}

function mapsFit() {
  const cv = mcanvas();
  const w = cv.width, hgt = cv.height;
  if (!w || !hgt) return;
  const pad = M.map === 'scatter' ? 64 * (M.dpr || 1) : 28 * (M.dpr || 1);
  M.k = Math.min(w - pad * 2, hgt - pad * 2);
  M.tx = (w - M.k) / 2;
  M.ty = (hgt - M.k) / 2;
  mapsDraw();
}

function mapsSchedule() {
  if (M.raf) return;
  M.raf = requestAnimationFrame(() => { M.raf = 0; mapsDraw(); });
}

function mapsDraw() {
  const cv = mcanvas();
  const ctx = cv.getContext('2d');
  const w = cv.width, hgt = cv.height;
  if (!w || !hgt || !M.xy) return;
  const dpr = M.dpr || 1;

  ctx.clearRect(0, 0, w, hgt);
  if (M.map === 'scatter') mapsDrawAxes(ctx, w, hgt, dpr);

  const n = M.xy.length / 2;
  // Dots shrink as the archive grows so a dense region reads as a shape
  // rather than a solid block, and grow with zoom so a magnified cluster
  // becomes individually clickable.
  const base = n > 60000 ? 1.1 : n > 12000 ? 1.7 : n > 2000 ? 2.4 : 3.4;
  const r = Math.max(0.7, base * dpr * Math.min(2.4, Math.pow(M.k / 700, 0.4)));
  const flat = M.map === 'semantic';
  const alpha = n > 40000 ? 0.5 : n > 8000 ? 0.65 : 0.85;

  // Batched by colour: a fillStyle change is the expensive part of a canvas
  // scatter, so all points of one cluster are drawn in a single path.
  const byColour = new Map();
  for (let i = 0; i < n; i++) {
    const c = M.cluster[i];
    if (M.muted.has(c)) continue;
    const key = flat ? -1 : c;
    let list = byColour.get(key);
    if (!list) { list = []; byColour.set(key, list); }
    list.push(i);
  }

  for (const [c, list] of byColour) {
    ctx.fillStyle = flat
      ? `rgba(198,214,212,${alpha})` : clusterColor(c, alpha);
    ctx.beginPath();
    for (const i of list) {
      const x = M.xy[i * 2] * M.k + M.tx;
      const y = M.xy[i * 2 + 1] * M.k + M.ty;
      if (x < -8 || y < -8 || x > w + 8 || y > hgt + 8) continue;
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, 6.283185);
    }
    ctx.fill();
  }

  if (M.map === 'cluster') mapsDrawClusterLabels(ctx, w, hgt, dpr);
  for (const i of [M.hover, M.picked]) {
    if (i < 0 || i >= n || M.muted.has(M.cluster[i])) continue;
    const x = M.xy[i * 2] * M.k + M.tx, y = M.xy[i * 2 + 1] * M.k + M.ty;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(5 * dpr, r + 3.5 * dpr), 0, 6.283185);
    ctx.lineWidth = 2 * dpr;
    // Amber means "this is the selection" everywhere else in Atlas, and the
    // hover ring borrows the visual-evidence teal rather than inventing a
    // colour, so the map reads as the same application as the ribbons.
    ctx.strokeStyle = i === M.picked ? '#FFB020' : '#5EC8D8';
    ctx.stroke();
  }

  if (M.box) {
    const b = M.box;
    ctx.setLineDash([5 * dpr, 4 * dpr]);
    ctx.strokeStyle = 'rgba(94,200,216,0.9)';
    ctx.lineWidth = 1.4 * dpr;
    ctx.strokeRect(Math.min(b.x0, b.x1), Math.min(b.y0, b.y1),
                   Math.abs(b.x1 - b.x0), Math.abs(b.y1 - b.y0));
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(94,200,216,0.07)';
    ctx.fillRect(Math.min(b.x0, b.x1), Math.min(b.y0, b.y1),
                 Math.abs(b.x1 - b.x0), Math.abs(b.y1 - b.y0));
  }
}

/* Cluster names, drawn at the centre of each group and hidden when two would
 * overlap. A legend nobody can read is worse than no legend, so this drops
 * labels rather than stacking them. */
function mapsDrawClusterLabels(ctx, w, hgt, dpr) {
  if (!M.clusters.length) return;
  ctx.font = `${12 * dpr}px "Public Sans", system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const taken = [];
  for (const c of M.clusters) {
    if (M.muted.has(c.cluster)) continue;
    const x = c.cx * M.k + M.tx, y = c.cy * M.k + M.ty;
    if (x < 0 || y < 0 || x > w || y > hgt) continue;
    const label = c.label || `group ${c.cluster + 1}`;
    const wide = ctx.measureText(label).width;
    if (taken.some(t => Math.abs(t.x - x) < (t.w + wide) / 2 + 8 * dpr &&
                        Math.abs(t.y - y) < 20 * dpr)) continue;
    taken.push({ x, y, w: wide });
    ctx.lineWidth = 3.5 * dpr;
    ctx.strokeStyle = 'rgba(11,20,22,0.92)';
    ctx.strokeText(label, x, y);
    ctx.fillStyle = clusterColor(c.cluster, 1);
    ctx.fillText(label, x, y);
  }
}

function mapsDrawAxes(ctx, w, hgt, dpr) {
  const d = M.scatter && M.scatter.domain;
  if (!d) return;
  ctx.strokeStyle = 'rgba(36,64,63,0.9)';
  ctx.lineWidth = 1 * dpr;
  ctx.fillStyle = '#5B7876';
  ctx.font = `${11 * dpr}px "IBM Plex Mono", monospace`;
  const fmt = (v, isLog) => {
    const n = isLog ? Math.pow(10, v) : v;
    if (Math.abs(n) >= 1000) return fmtInt(Math.round(n));
    return (Math.abs(n) < 10 ? n.toFixed(1) : String(Math.round(n)));
  };
  for (let i = 0; i <= 4; i++) {
    const f = i / 4;
    const x = f * M.k + M.tx, y = f * M.k + M.ty;
    ctx.beginPath(); ctx.moveTo(x, M.ty); ctx.lineTo(x, M.ty + M.k); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(M.tx, y); ctx.lineTo(M.tx + M.k, y); ctx.stroke();
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(fmt(d.lo0 + f * (d.hi0 - d.lo0), M.slogx), x, M.ty + M.k + 6 * dpr);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(fmt(d.hi1 - f * (d.hi1 - d.lo1), M.slogy), M.tx - 8 * dpr, y);
  }
  ctx.fillStyle = '#E8EAED';
  ctx.font = `${12 * dpr}px "Public Sans", system-ui, sans-serif`;  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText(M.scatter.x_label + (M.slogx ? ' (log)' : ''),
               M.tx + M.k / 2, M.ty + M.k + 24 * dpr);
  ctx.save();
  ctx.translate(M.tx - 46 * dpr, M.ty + M.k / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(M.scatter.y_label + (M.slogy ? ' (log)' : ''), 0, 0);
  ctx.restore();
}

/* ── legend and controls ───────────────────────────────────────────────── */
function mapsRenderLegend() {
  const box = $('mapsLegend');
  box.textContent = '';

  if (M.map === 'scatter') {
    box.appendChild(mapsAxisPicker());
    return;
  }

  const head = h('div', { class: 'mlegend-head' },
    h('span', { text: M.map === 'cluster' ? 'Groups of meaning' : 'Colour off' }),
    M.clusters.length ? h('button', {
      class: 'btn btn-quiet btn-tiny',
      onclick: () => { M.muted.clear(); mapsRenderLegend(); mapsDraw(); },
      text: 'Show all',
    }) : null);
  box.appendChild(head);

  if (M.map === 'semantic') {
    box.appendChild(h('p', { class: 'mlegend-note', text:
      'Distance is meaning: two dots near each other are two passages the ' +
      'models described in similar terms. Drag a box to select, click a dot ' +
      'to open it, scroll to zoom.' }));
  }

  const list = h('div', { class: 'mlegend-list' });
  for (const c of M.clusters) {
    const off = M.muted.has(c.cluster);
    list.appendChild(h('button', {
      class: 'mlegend-item' + (off ? ' off' : ''),
      title: (c.terms || []).join(', '),
      onclick: (ev) => {
        // Plain click isolates, which is what people actually want from a
        // legend; modifier-click toggles one off, for building a comparison.
        if (ev.shiftKey || ev.metaKey || ev.ctrlKey) {
          if (off) M.muted.delete(c.cluster); else M.muted.add(c.cluster);
        } else if (M.muted.size === M.clusters.length - 1 && !off) {
          M.muted.clear();
        } else {
          M.muted = new Set(M.clusters.map(x => x.cluster)
            .filter(x => x !== c.cluster));
        }
        mapsRenderLegend();
        mapsDraw();
      },
      ondblclick: () => mapsOpenCluster(c.cluster),
    },
      h('i', { style: `background:${clusterColor(c.cluster, 1)}` }),
      h('span', { class: 'mli-label', text: c.label || `group ${c.cluster + 1}` }),
      h('span', { class: 'mli-n', text: fmtInt(c.size) })));
  }
  box.appendChild(list);
  if (M.clusters.length) {
    box.appendChild(h('p', { class: 'mlegend-note', text:
      'Click a group to isolate it · shift-click to toggle · double-click for ' +
      'its videos' }));
  }
}

function mapsAxisPicker() {
  const wrap = h('div', { class: 'mscatter-controls' });
  const pick = (which) => {
    const sel = h('select', { class: 'mini-select', onchange: (ev) => {
      if (which === 'x') M.sx = ev.target.value; else M.sy = ev.target.value;
      mapsLoadScatter();
    } });
    for (const a of M.axes) {
      sel.appendChild(h('option', {
        value: a.name, selected: (which === 'x' ? M.sx : M.sy) === a.name,
      }, a.label));
    }
    return sel;
  };
  wrap.appendChild(h('label', { class: 'mctl' }, h('span', { text: 'X' }), pick('x')));
  wrap.appendChild(h('label', { class: 'mctl' }, h('span', { text: 'Y' }), pick('y')));
  wrap.appendChild(h('button', {
    class: 'btn btn-quiet btn-tiny' + (M.slogx ? ' on' : ''),
    onclick: () => { M.slogx = !M.slogx; mapsLoadScatter(); },
    text: 'log X',
  }));
  wrap.appendChild(h('button', {
    class: 'btn btn-quiet btn-tiny' + (M.slogy ? ' on' : ''),
    onclick: () => { M.slogy = !M.slogy; mapsLoadScatter(); },
    text: 'log Y',
  }));
  wrap.appendChild(h('button', {
    class: 'btn btn-quiet btn-tiny',
    onclick: () => { M.sx = [M.sy, M.sy = M.sx][0]; mapsLoadScatter(); },
    text: 'Swap',
  }));
  if (M.scatter && M.scatter.groups && M.scatter.groups.length > 1) {
    const list = h('div', { class: 'mlegend-list' });
    M.scatter.groups.forEach((g, i) => {
      list.appendChild(h('span', { class: 'mlegend-item static' },
        h('i', { style: `background:${clusterColor(i, 1)}` }),
        h('span', { class: 'mli-label', text: g })));
    });
    wrap.appendChild(list);
  }
  return wrap;
}

/* ── the drill-down: what a dot actually is ────────────────────────────── */
async function mapsOpenPoint(i) {
  if (i < 0 || !M.refs) return;
  M.picked = i;
  mapsDraw();
  const panels = $('mapsPanels');
  panels.textContent = '';
  panels.appendChild(h('div', { class: 'mpanel', text: 'Loading…' }));

  // A scatter point is a video and needs no projection lookup — it already
  // carries everything the panel shows, so it skips the round trip.
  if (M.map === 'scatter') {
    const p = M.scatter.points[i];
    mapsRenderPoint({
      ok: true, level: 'video', ref: p.key, video_key: p.key,
      video: { title: p.title, creator: p.creator, moment_count: p.n },
      scatter: { x: p.x, y: p.y },
    });
    return;
  }
  try {
    const got = await api(`/api/map/point?level=${M.level}&ref=` +
                          encodeURIComponent(M.refs[i]));
    mapsRenderPoint(got);
  } catch (e) {
    panels.textContent = '';
    panels.appendChild(h('div', { class: 'mpanel', text: String(e.message || e) }));
  }
}

function mapsRenderPoint(got) {
  const panels = $('mapsPanels');
  panels.textContent = '';
  const v = got.video || {};
  const key = got.video_key;
  const at = got.t_start;

  const card = h('div', { class: 'mpanel mpanel-point' });
  card.appendChild(h('button', {
    class: 'mpanel-close', 'aria-label': 'Close',
    onclick: () => { panels.textContent = ''; M.picked = -1; mapsDraw(); },
  }, '×'));

  card.appendChild(h('h3', { class: 'mp-title',
    text: v.title || key || got.ref }));

  const facts = h('dl', { class: 'mp-facts' });
  const fact = (k, val) => {
    if (val === null || val === undefined || val === '') return;
    facts.appendChild(h('dt', { text: k }));
    facts.appendChild(h('dd', { text: String(val) }));
  };
  fact('video', key);
  if (v.creator) fact('creator', v.creator);
  if (v.category) fact('collection', v.category);
  if (v.duration) fact('duration', timecode(v.duration));
  if (v.moment_count !== undefined) fact('indexed passages', fmtInt(v.moment_count));
  if (at !== null && at !== undefined) fact('at', timecode(at));
  if (got.source) fact('found by', SOURCE_LABEL[got.source] || got.source);
  if (got.scatter) {
    fact(M.scatter.x_label, got.scatter.x.toFixed(2));
    fact(M.scatter.y_label, got.scatter.y.toFixed(2));
  }
  card.appendChild(facts);

  if (got.cluster_info) {
    const ci = got.cluster_info;
    card.appendChild(h('div', { class: 'mp-cluster' },
      h('i', { style: `background:${clusterColor(got.cluster, 1)}` }),
      h('button', {
        class: 'linky', onclick: () => mapsOpenCluster(got.cluster),
        text: `${ci.label} · ${fmtInt(ci.size)} points`,
      })));
    if (ci.terms && ci.terms.length) {
      card.appendChild(h('p', { class: 'mp-terms',
        text: 'what names this group: ' + ci.terms.join(' · ') }));
    }
  }

  if (got.moment && got.moment.text) {
    card.appendChild(h('p', { class: 'mp-text', text: got.moment.text }));
  } else if (got.moments && got.moments.length) {
    const list = h('ul', { class: 'mp-moments' });
    for (const m of got.moments.slice(0, 12)) {
      list.appendChild(h('li', {},
        h('button', {
          class: 'linky',
          onclick: () => openVideo({ video_key: key, title: v.title,
                                     creator: v.creator }, m.t_start),
          text: timecode(m.t_start),
        }),
        h('span', { class: 'mp-src', text: SOURCE_LABEL[m.source] || m.source,
                    style: `color:${color(m.source)}` }),
        h('span', { class: 'mp-mt', text: (m.text || '').slice(0, 160) })));
    }
    card.appendChild(list);
  }

  // The cross-tab links. A map is only worth having if a dot leads somewhere.
  const acts = h('div', { class: 'mp-acts' });
  if (key) {
    acts.appendChild(h('button', {
      class: 'btn',
      onclick: () => openVideo({ video_key: key, title: v.title,
                                 creator: v.creator },
                               at === null || at === undefined ? 0 : at),
      text: at ? `Play at ${timecode(at)}` : 'Play',
    }));
    acts.appendChild(h('button', {
      class: 'btn btn-quiet',
      onclick: () => { showTab('graph'); graphFocusKey(key); },
      text: 'In the graph',
    }));
    acts.appendChild(h('button', {
      class: 'btn btn-quiet',
      onclick: () => { showTab('search'); $('q').value = v.title || key;
                       runSearch(v.title || key); },
      text: 'Find similar',
    }));
  }
  if (got.sql) {
    acts.appendChild(h('button', {
      class: 'btn btn-quiet',
      onclick: () => {
        navigator.clipboard.writeText(got.sql)
          .then(() => toast('query copied'))
          .catch(() => toast(got.sql));
      },
      title: got.sql,
      text: 'Copy the query',
    }));
  }
  card.appendChild(acts);
  panels.appendChild(card);
}

async function mapsOpenCluster(cluster) {
  const panels = $('mapsPanels');
  panels.textContent = '';
  panels.appendChild(h('div', { class: 'mpanel', text: 'Loading…' }));
  try {
    const got = await api(`/api/map/cluster/${cluster}?level=${M.level}&limit=40`);
    panels.textContent = '';
    const card = h('div', { class: 'mpanel' });
    card.appendChild(h('button', {
      class: 'mpanel-close', 'aria-label': 'Close',
      onclick: () => { panels.textContent = ''; },
    }, '×'));
    card.appendChild(h('h3', { class: 'mp-title' },
      h('i', { class: 'mp-swatch',
               style: `background:${clusterColor(cluster, 1)}` }),
      got.label || `group ${cluster + 1}`));
    card.appendChild(h('p', { class: 'mp-terms',
      text: `${fmtInt(got.size)} points across ${fmtInt(got.videos)} videos · ` +
            (got.terms || []).join(' · ') }));

    const grid = h('div', { class: 'mp-grid' });
    const seen = new Set();
    for (const it of got.items) {
      if (seen.has(it.video_key)) continue;
      seen.add(it.video_key);
      const video = { video_key: it.video_key, title: it.title,
                      creator: it.creator };
      const tile = h('button', {
        class: 'mp-tile',
        onclick: () => openVideo(video, it.t_start || 0),
      },
        posterImg(video, it.t_start, 'mp-shot'),
        h('span', { class: 'mp-tile-t',
                    text: it.title || it.video_key }));
      grid.appendChild(tile);
    }
    card.appendChild(grid);
    card.appendChild(h('div', { class: 'mp-acts' },
      h('button', {
        class: 'btn',
        onclick: () => mapsSendToGraph(Array.from(seen)),
        text: 'Graph this group',
      }),
      h('button', {
        class: 'btn btn-quiet',
        onclick: () => { M.muted = new Set(M.clusters.map(x => x.cluster)
                           .filter(x => x !== cluster));
                         mapsRenderLegend(); mapsDraw(); },
        text: 'Isolate on the map',
      })));
    panels.appendChild(card);
  } catch (e) {
    panels.textContent = '';
    panels.appendChild(h('div', { class: 'mpanel', text: String(e.message || e) }));
  }
}

/* A dragged box becomes a set of videos, and a set of videos is exactly what
 * the graph's `from_keys` view already takes. That is the whole point of the
 * selection: the map answers "what is over here", the graph answers "how is
 * it connected", and neither has to reimplement the other. */
async function mapsSelectRegion(x0, y0, x1, y1) {
  const u0 = (Math.min(x0, x1) - M.tx) / M.k, u1 = (Math.max(x0, x1) - M.tx) / M.k;
  const v0 = (Math.min(y0, y1) - M.ty) / M.k, v1 = (Math.max(y0, y1) - M.ty) / M.k;

  if (M.map === 'scatter') {
    // Local: the scatter payload is already in memory, so there is nothing to
    // ask the server for.
    const keys = [];
    for (let i = 0; i < M.xy.length / 2; i++) {
      if (M.muted.has(M.cluster[i])) continue;
      const x = M.xy[i * 2], y = M.xy[i * 2 + 1];
      if (x >= u0 && x <= u1 && y >= v0 && y <= v1) keys.push(M.keys[i]);
    }
    mapsRenderRegion({ ok: true, count: keys.length, videos: keys.length,
                       keys, items: [] });
    return;
  }
  try {
    const p = new URLSearchParams({ level: M.level, x0: u0, y0: v0,
                                    x1: u1, y1: v1, limit: '600' });
    mapsRenderRegion(await api('/api/map/region?' + p.toString()));
  } catch (e) { toast(String(e.message || e)); }
}

function mapsRenderRegion(got) {
  const panels = $('mapsPanels');
  panels.textContent = '';
  if (!got.count) {
    panels.appendChild(h('div', { class: 'mpanel',
      text: 'Nothing in that box. Drag over a denser area.' }));
    return;
  }
  const card = h('div', { class: 'mpanel' });
  card.appendChild(h('button', {
    class: 'mpanel-close', 'aria-label': 'Close',
    onclick: () => { panels.textContent = ''; },
  }, '×'));
  card.appendChild(h('h3', { class: 'mp-title',
    text: `${fmtInt(got.count)} points · ${fmtInt(got.videos)} videos` }));

  const grid = h('div', { class: 'mp-grid' });
  const shown = got.items && got.items.length
    ? got.items : got.keys.map(k => ({ video_key: k }));
  const seen = new Set();
  for (const it of shown) {
    if (seen.has(it.video_key) || seen.size >= 60) continue;
    seen.add(it.video_key);
    const video = { video_key: it.video_key, title: it.title,
                    creator: it.creator };
    grid.appendChild(h('button', {
      class: 'mp-tile', onclick: () => openVideo(video, it.t_start || 0),
    },
      posterImg(video, it.t_start, 'mp-shot'),
      h('span', { class: 'mp-tile-t', text: it.title || it.video_key })));
  }
  card.appendChild(grid);
  card.appendChild(h('div', { class: 'mp-acts' },
    h('button', { class: 'btn',
      onclick: () => mapsSendToGraph(got.keys.slice(0, 24)),
      text: 'Graph this selection' }),
    h('button', { class: 'btn btn-quiet',
      onclick: () => { showTab('library'); },
      text: 'Open the library' })));
  panels.appendChild(card);
}

async function mapsSendToGraph(keys) {
  if (!keys || !keys.length) return;
  G.loaded = true;
  G.mode = 'data';
  showTab('graph');
  try {
    const data = await api('/api/graph/from?keys=' +
                           encodeURIComponent(keys.slice(0, 24).join(',')));
    G.nodes.clear(); G.edges.clear();
    gabsorb(data);
    gresize(); G.alpha = 1;
    for (let i = 0; i < 160; i++) gtick();
    gfit();
  } catch (e) { toast(String(e.message || e)); }
}

/* Focus the graph on one video, for the "in the graph" link on a point. */
async function graphFocusKey(key) {
  G.loaded = true;
  try {
    const data = await api('/api/graph/from?keys=' + encodeURIComponent(key));
    G.nodes.clear(); G.edges.clear();
    gabsorb(data);
    gresize(); G.alpha = 1;
    for (let i = 0; i < 160; i++) gtick();
    gfit();
  } catch (e) { toast(String(e.message || e)); }
}

/* ── pointer, keyboard and wiring ──────────────────────────────────────── */
function mapsTooltip(i, px, py) {
  const tip = $('mapsTooltip');
  if (i < 0) { tip.hidden = true; return; }
  const cluster = M.clusters.find(c => c.cluster === M.cluster[i]);
  let label = M.keys ? M.keys[i] : '';
  let sub = '';
  if (M.map === 'scatter' && M.scatter) {
    const p = M.scatter.points[i];
    label = p.title || p.key;
    sub = `${M.scatter.x_label} ${p.x.toFixed(1)} · ` +
          `${M.scatter.y_label} ${p.y.toFixed(1)}`;
  } else {
    const t = M.times ? M.times[i] : null;
    sub = [cluster ? cluster.label : '',
           (t !== null && t !== undefined) ? timecode(t) : '']
      .filter(Boolean).join(' · ');
  }
  tip.textContent = '';
  tip.appendChild(h('strong', { text: String(label) }));
  if (sub) tip.appendChild(h('span', { text: sub }));
  tip.hidden = false;
  const box = $('mapsViewport').getBoundingClientRect();
  const w = tip.offsetWidth || 180, hh = tip.offsetHeight || 40;
  tip.style.left = Math.min(box.width - w - 8, Math.max(8, px + 14)) + 'px';
  tip.style.top = Math.min(box.height - hh - 8, Math.max(8, py + 14)) + 'px';
}

function mapsWire() {
  const cv = mcanvas();
  const dpr = () => M.dpr || 1;

  $$('.maps-tabs button').forEach(b => b.addEventListener('click', () => {
    if (M.map === b.dataset.map) return;
    M.map = b.dataset.map;
    $$('.maps-tabs button').forEach(x =>
      x.classList.toggle('on', x.dataset.map === M.map));
    M.picked = -1; M.hover = -1; M.muted.clear();
    $('mapsPanels').textContent = '';
    // Semantic and cluster are the same fetch, so switching between them is
    // a repaint; scatter is a different dataset entirely.
    if (M.map === 'scatter') {
      if (M.dataset === 'scatter') {
        mapsRenderLegend(); mapsResize(); mapsFit();
      } else mapsLoadScatter();
    } else if (M.dataset === 'projection') {
      mapsRenderLegend(); mapsRenderNote({}); mapsResize(); mapsFit();
    } else mapsLoadProjection();
  }));

  cv.addEventListener('mousemove', (ev) => {
    const box = cv.getBoundingClientRect();
    const px = (ev.clientX - box.left) * dpr(), py = (ev.clientY - box.top) * dpr();
    if (M.drag) {
      if (M.drag.mode === 'pan') {
        M.tx = M.drag.tx0 + (px - M.drag.px0);
        M.ty = M.drag.ty0 + (py - M.drag.py0);
      } else {
        M.box = { x0: M.drag.px0, y0: M.drag.py0, x1: px, y1: py };
      }
      mapsSchedule();
      return;
    }
    const i = mapsPick(px, py);
    if (i !== M.hover) { M.hover = i; mapsSchedule(); }
    mapsTooltip(i, ev.clientX - box.left, ev.clientY - box.top);
    cv.style.cursor = i >= 0 ? 'pointer' : 'crosshair';
  });

  cv.addEventListener('mouseleave', () => {
    M.hover = -1; $('mapsTooltip').hidden = true; mapsSchedule();
  });

  cv.addEventListener('mousedown', (ev) => {
    const box = cv.getBoundingClientRect();
    const px = (ev.clientX - box.left) * dpr(), py = (ev.clientY - box.top) * dpr();
    // Shift or middle drags a selection box; a plain drag pans, which is what
    // a pointer over a map is expected to do.
    const boxing = ev.shiftKey || ev.button === 1;
    M.drag = { mode: boxing ? 'box' : 'pan', px0: px, py0: py,
               tx0: M.tx, ty0: M.ty, moved: false };
    $('mapsTooltip').hidden = true;
    ev.preventDefault();
  });

  window.addEventListener('mouseup', (ev) => {
    if (!M.drag) return;
    const cvv = mcanvas();
    const box = cvv.getBoundingClientRect();
    const px = (ev.clientX - box.left) * dpr(), py = (ev.clientY - box.top) * dpr();
    const far = Math.hypot(px - M.drag.px0, py - M.drag.py0) > 5 * dpr();
    if (M.drag.mode === 'box' && far) {
      mapsSelectRegion(M.drag.px0, M.drag.py0, px, py);
    } else if (!far) {
      const i = mapsPick(px, py);
      if (i >= 0) mapsOpenPoint(i);
    }
    M.drag = null; M.box = null;
    mapsDraw();
  });

  cv.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const box = cv.getBoundingClientRect();
    const px = (ev.clientX - box.left) * dpr(), py = (ev.clientY - box.top) * dpr();
    const factor = Math.exp(-ev.deltaY * 0.0016);
    const next = Math.max(60, Math.min(90000, M.k * factor));
    // Zoom about the pointer, so the thing under the cursor stays under it.
    M.tx = px - (px - M.tx) * (next / M.k);
    M.ty = py - (py - M.ty) * (next / M.k);
    M.k = next;
    mapsSchedule();
  }, { passive: false });

  cv.addEventListener('dblclick', () => mapsFit());

  // Keyboard operation, because a canvas that only answers to a mouse is a
  // canvas half the people who open it cannot use.
  cv.addEventListener('keydown', (ev) => {
    const step = 60;
    if (ev.key === 'ArrowLeft') M.tx += step;
    else if (ev.key === 'ArrowRight') M.tx -= step;
    else if (ev.key === 'ArrowUp') M.ty += step;
    else if (ev.key === 'ArrowDown') M.ty -= step;
    else if (ev.key === '+' || ev.key === '=') M.k *= 1.25;
    else if (ev.key === '-') M.k /= 1.25;
    else if (ev.key === '0') { mapsFit(); return; }
    else if (ev.key === 'Tab' && M.refs && M.refs.length) {
      // Walk the points in order, so every dot is reachable without a mouse.
      ev.preventDefault();
      let n = M.refs.length, i = M.picked;
      for (let step2 = 0; step2 < n; step2++) {
        i = ((ev.shiftKey ? i - 1 : i + 1) + n) % n;
        if (!M.muted.has(M.cluster[i])) break;
      }
      mapsOpenPoint(i);
      return;
    } else return;
    ev.preventDefault();
    mapsSchedule();
  });

  window.addEventListener('resize', () => {
    if (S.tab !== 'maps') return;
    mapsResize();
    mapsDraw();
  });
}

/* ════════════════════════════════════════════════════════════════════════
   STATUS
   ════════════════════════════════════════════════════════════════════════ */
let statusTimer = 0;
let lastPhase = '';

async function pollStatus() {
  try {
    const st = await api('/api/status');
    S.status = st;
    paintPulse(st);
    // The landing page is a live report, so it repaints on the same tick as the
    // pulse rather than only when the tab is opened.
    if (S.tab === 'home') renderHome();

    const phase = `${st.boot.phase}|${st.index.phase}|${st.ingest.phase}`;
    if (phase !== lastPhase) {
      lastPhase = phase;
      if (S.tab === 'sources') loadSources();
      // A finished index or import invalidates every cached answer.
      if (st.boot.phase === 'ready' && st.index.phase === 'done') {
        S.searchCache.clear();
        if (S.query && !S.results.length) runSearch(S.query);
        if (!S.facets || !S.facets.totals.videos) loadFacets(); else renderOpening();
      }
    }
    if (!S.facets && st.search && st.search.videos) loadFacets();
  } catch { /* the server is still coming up */ }

  const busyNow = S.status && (S.status.boot.phase !== 'ready' ||
    S.status.ingest.running || S.status.index.running);
  clearTimeout(statusTimer);
  statusTimer = setTimeout(pollStatus, busyNow ? 1500 : 12000);
}

function paintPulse(st) {
  const dot = $$('.pulse-dot')[0];
  const text = $$('.pulse-text')[0];
  const ing = st.ingest, idx = st.index;

  let state = 'warming', label = st.boot.detail || st.boot.phase;

  if (ing.running) {
    state = 'warming';
    label = ing.scan_total
      ? `scanning ${fmtInt(ing.scanned)}/${fmtInt(ing.scan_total)}`
      : (ing.current || ing.detail || 'scanning channel');
    if (ing.bytes_total)
      label = `importing ${Math.round(100 * ing.bytes_done / ing.bytes_total)}%`;
  } else if (idx.running) {
    state = 'warming';
    label = idx.embed_total
      ? `embedding ${fmtInt(idx.embedded)}/${fmtInt(idx.embed_total)}`
      : (idx.detail || 'indexing');
  } else if (st.boot.phase === 'error') {
    state = 'error';
    label = 'needs attention';
  } else if (st.boot.phase === 'ready') {
    state = 'ready';
    const s = st.search;
    label = `${fmtInt(s.videos)} videos · ${fmtInt(s.moments)} passages` +
            (s.dense_ready ? '' : ' · words only');
  }

  dot.dataset.state = state;
  text.textContent = label;
  $('pulse').title = st.boot.detail || label;
}

/* ════════════════════════════════════════════════════════════════════════
   WIRING
   ════════════════════════════════════════════════════════════════════════ */
function wire() {
  $('finder').addEventListener('submit', (ev) => {
    ev.preventDefault();
    closeSuggest();
    S.sourceFilter.clear();
    // A new question starts unnarrowed. Carrying the previous query's creator
    // over would silently hide most of what this one found, and the filter
    // doing it is scrolled out of sight. Sort is a standing preference, not a
    // property of the query, so that one stays.
    S.narrow = { creator: '', category: '', min_dur: '', max_dur: '', min_hits: '' };
    for (const id of ['narrowMinDur', 'narrowMaxDur', 'narrowMinHits']) $(id).value = '';
    if (S.tab !== 'search') showTab('search', { push: false });
    runSearch($('q').value);
  });

  $('q').addEventListener('input', (ev) => {
    if (!ev.target.value.trim()) { closeSuggest(); showOpening(); return; }
    scheduleSuggest(ev.target.value);
  });

  $('q').addEventListener('keydown', (ev) => {
    if (ev.key === 'ArrowDown') { ev.preventDefault(); moveSuggest(1); }
    else if (ev.key === 'ArrowUp') { ev.preventDefault(); moveSuggest(-1); }
    else if (ev.key === 'Escape') closeSuggest();
  });

  document.addEventListener('click', (ev) => {
    if (!$('finder').contains(ev.target)) closeSuggest();
  });

  document.addEventListener('keydown', (ev) => {
    if (ev.key === '/' && document.activeElement !== $('q')) {
      ev.preventDefault(); $('q').focus(); $('q').select();
    }
    // The inspector is the most recently opened thing, so it is the thing
    // Escape means. Only once it is gone does Escape reach the player.
    if (ev.key === 'Escape' && !$('cellSlab').hidden) {
      $('cellSlab').hidden = true;
      // document bubbles to window, where the player's own Escape lives, and
      // one press should dismiss one layer.
      ev.stopPropagation();
      return;
    }
    if (ev.key === 'Escape' && S.video && document.activeElement !== $('q')) closePlayer();
  });

  $$('.tabs button').forEach(b =>
    b.addEventListener('click', () => showTab(b.dataset.tab)));
  document.querySelector('.brand').addEventListener('click', (ev) => {
    ev.preventDefault(); showTab('home');
  });
  $('pulse').addEventListener('click', () => showTab('sources'));

  /* ── the landing page ──
   * Its search field is the top bar's, at the size the page is about; handing
   * the query to the same code path means the two can never disagree about what
   * a query means. */
  $('homeFind').addEventListener('submit', (ev) => {
    ev.preventDefault();
    const q = $('homeQ').value.trim();
    if (!q) { $('homeQ').focus(); return; }
    $('q').value = q;
    showTab('search');
    runSearch(q);
  });
  // Every "go here" button on the page, in one place: the doors, and the link
  // out to the library.
  $$('[data-goto]').forEach(b =>
    b.addEventListener('click', () => showTab(b.dataset.goto)));

  $('moreBtn').addEventListener('click', () => runSearch(S.query, { append: true }));

  /* ── search layout, order and narrowing ── */
  $$('.rc-view').forEach(b =>
    b.addEventListener('click', () => {
      setSearchView(b.dataset.view);
      // Nothing is refetched — the rows in hand are relaid out in place.
      if (S.results.length) renderCards(S.results, false);
    }));

  let savedView = '', savedSDensity = '';
  try {
    savedView = localStorage.getItem('atlas.searchView') || '';
    savedSDensity = localStorage.getItem('atlas.searchDensity') || '';
  } catch { /* private mode */ }
  if (savedSDensity) $('searchDensity').value = savedSDensity;
  S.density = Number($('searchDensity').value) || 3;
  setSearchView(savedView || 'list', { remember: false });

  $('searchDensity').addEventListener('input', (ev) => {
    applySearchDensity(Number(ev.target.value), { remember: true });
  });

  $('searchSort').addEventListener('change', (ev) => {
    S.sort = ev.target.value;
    if (S.query) runSearch(S.query);
  });

  $('searchNarrowBtn').addEventListener('click', () => {
    setNarrowOpen($('searchNarrow').hidden);
  });
  $('narrowClear').addEventListener('click', clearNarrow);

  // Number fields debounce: typing "120" should not run a search at "1".
  let narrowTimer = 0;
  const narrowRun = () => {
    clearTimeout(narrowTimer);
    if (S.query) runSearch(S.query);
  };
  for (const [id, key] of [['narrowMinDur', 'min_dur'], ['narrowMaxDur', 'max_dur'],
                           ['narrowMinHits', 'min_hits']]) {
    const el = $(id);
    el.addEventListener('input', () => {
      S.narrow[key] = el.value.trim();
      clearTimeout(narrowTimer);
      narrowTimer = setTimeout(narrowRun, 420);
    });
    el.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') { ev.preventDefault(); narrowRun(); }
    });
  }

  $('libMoreBtn').addEventListener('click', () => loadLibrary(false));
  $('libSort').addEventListener('change', () => loadLibrary(true));
  $('libHas').addEventListener('change', () => loadLibrary(true));

  let libTimer = 0;
  const libRun = () => { clearTimeout(libTimer); loadLibrary(true); };
  $('libQ').addEventListener('input', () => {
    clearTimeout(libTimer);
    libTimer = setTimeout(() => loadLibrary(true), 220);
  });
  // The debounce is a convenience, not the contract: pressing Enter or the
  // button searches now, because a search box that only reacts to typing
  // pauses feels broken the moment you expect it to obey.
  $('libQ').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') { ev.preventDefault(); libRun(); }
  });
  $('libGo').addEventListener('click', libRun);

  const density = $('libDensity');
  let savedDensity = '';
  try { savedDensity = localStorage.getItem('atlas.libDensity') || ''; } catch { /* private mode */ }
  if (savedDensity) density.value = savedDensity;
  libDensity(Number(density.value));
  density.addEventListener('input', () => libDensity(Number(density.value)));

  let browseTimer = 0;
  $('browserQ').addEventListener('input', (ev) => {
    S.browse.q = ev.target.value.trim();
    clearTimeout(browseTimer);
    browseTimer = setTimeout(() => openTable(S.browse.table, 0), 260);
  });
  $('browserClose').addEventListener('click', () => {
    $('browser').hidden = true;
    $('cellSlab').hidden = true;
  });
  $('cellClose').addEventListener('click', () => { $('cellSlab').hidden = true; });

  gwire();
  mapsWire();

  $('rescanBtn').addEventListener('click', async () => {
    $('rescanBtn').disabled = true;
    try {
      const r = await api('/api/scan?full=true', { method: 'POST' });
      toast(r.ok ? 'Scanning the channel — progress is in the status pill.'
                 : (r.note || 'already running'));
      pollStatus();
    } catch (e) { toast(e.message); }
    setTimeout(() => { $('rescanBtn').disabled = false; }, 2500);
  });

  $('reindexBtn').addEventListener('click', async () => {
    $('reindexBtn').disabled = true;
    toast('Rebuilding the index — search stays up on the old one until it swaps.');
    try {
      await api('/api/reindex?embed=true', { method: 'POST' });
      S.searchCache.clear();
      toast('Index rebuilt.');
      if (S.query) runSearch(S.query);
    } catch (e) { toast(e.message); }
    $('reindexBtn').disabled = false;
  });

  $$('.strip button').forEach(b =>
    b.addEventListener('click', () => showPanel(b.dataset.panel)));
  $('screenClose').addEventListener('click', closePlayer);
  momNavWire();

  const vid = $('video');
  vid.addEventListener('loadeddata', () => { busy(false); clearInterval(statePoll); });
  vid.addEventListener('playing', () => busy(false));
  vid.addEventListener('error', () => {
    if (!S.video) return;
    // Almost always a 503 while the file is still arriving. One retry, then
    // the poller keeps the person informed rather than showing a dead frame.
    busy(true, 'waiting for the file');
    clearTimeout(retryTimer);
    retryTimer = setTimeout(() => {
      if (S.video) { vid.load(); vid.play().catch(() => {}); }
    }, 3000);
  });
  vid.addEventListener('timeupdate', () => {
    const span = (S.video && S.video.duration) || vid.duration || 0;
    if (!span) return;
    const head = $('playhead');
    if (head) head.style.left = (vid.currentTime / span * 100).toFixed(2) + '%';
    $('tNow').textContent = timecode(vid.currentTime);
    const now = vid.currentTime;
    $$('#panel-moments .mrow').forEach(row => {
      const t = row.dataset.t === '' ? null : Number(row.dataset.t);
      row.dataset.now = String(t !== null && now >= t && now < t + 6);
    });
    // Keep the "3 / 17" counter honest while the video plays past passages
    // the person did not step to.
    const i = momIndex(now);
    if (i !== momAt) momStepTo(now);
  });

  window.addEventListener('hashchange', () => {
    const { tab, params } = readHash();
    if (tab !== S.tab) showTab(tab, { push: false });
    const q = params.get('q') || '';
    if (tab === 'search' && q && q !== S.query) runSearch(q);
  });
}

/* ── boot ─────────────────────────────────────────────────────────────── */
function start() {
  wire();
  pollStatus();

  const { tab, params } = readHash();
  const q = params.get('q');
  // A link carrying a query is a link to results, whatever tab the hash names —
  // otherwise a shared URL lands on the landing page with the query invisible.
  showTab(q ? 'search' : tab, { push: false });
  if (q) { $('q').value = q; runSearch(q); }
  const v = params.get('v');
  if (v) openVideoKey(v, null);
  loadFacets();
}

if (document.readyState === 'loading')
  document.addEventListener('DOMContentLoaded', start);
else start();
