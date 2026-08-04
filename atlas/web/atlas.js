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
async function api(path, opts) {
  const res = await fetch(path, opts);
  const body = await res.text();
  let data;
  try { data = body ? JSON.parse(body) : {}; }
  catch { throw new Error(`${res.status}: ${body.slice(0, 160)}`); }
  if (!res.ok && data.note) throw new Error(data.note);
  return data;
}

/* ── state ─────────────────────────────────────────────────────────────── */
const S = {
  tab: 'search',
  query: '',
  results: [],
  resultTotal: 0,
  resultMeta: null,
  sourceFilter: new Set(),
  searchCache: new Map(),      // query|offset|filter → response
  facets: null,
  status: null,
  video: null,                 // the open video's search-result shape
  record: null,                // /api/video payload for the open video
  lib: { offset: 0, rows: [], total: 0, creator: '', category: '' },
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
function readHash() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [tab, qs] = raw.split('?');
  return {
    tab: ['search', 'library', 'graph', 'data', 'sources'].includes(tab) ? tab : 'search',
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
  if (tab === 'data' && !$('schema').childElementCount) loadSchema();
  if (tab === 'sources') loadSources();
}

/* ════════════════════════════════════════════════════════════════════════
   SEARCH
   ════════════════════════════════════════════════════════════════════════ */
function cacheKey(q, offset) {
  const f = Array.from(S.sourceFilter).sort().join(',');
  return `${q}|${offset}|${f}`;
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
  if (data.mode === 'hybrid') bits.push('meaning + words');
  else if (data.mode === 'lexical') bits.push('words only');
  else if (data.mode === 'dense') bits.push('meaning only');
  $('resultsCount').innerHTML = bits.join(' <span class="sep">·</span> ');
  // What these results have in common is often the more interesting question,
  // and it is one click away rather than a separate search.
  $('resultsCount').appendChild(h('button', {
    class: 'plot-link', onclick: graphFromResults,
    title: 'Plot these results and what they share on the graph',
  }, 'see the graph'));
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
  const img = h('img', { alt: '', loading: 'lazy', 'data-src': `/api/poster/${key}${t}` });
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

function renderCards(results, append) {
  const list = $('cards');
  if (!append) list.textContent = '';
  const frag = document.createDocumentFragment();

  for (const r of results.slice(append ? list.childElementCount : 0)) {
    const best = r.best || (r.moments || [])[0] || {};
    const at = (best.t_start === null || best.t_start === undefined)
      ? null : Number(best.t_start);

    const shot = posterImg(r, at, 'card-shot');
    if (r.duration) shot.appendChild(h('span', { class: 'dur', text: timecode(r.duration) }));
    if (r.has_file) shot.appendChild(h('i', { class: 'cached', title: 'already on this machine' }));

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
  fetch('/api/prefetch?keys=' + encodeURIComponent(fresh.join(',')),
        { method: 'POST' }).catch(() => {});
}

let statePoll = 0;
let retryTimer = 0;

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
    vid.src = `/api/play/${encodeURIComponent(key)}${frag}`;
    vid.load();
    vid.play().catch(() => {});
    pollMediaState(key);
  } else if (at !== null && at !== undefined && isFinite(at)) {
    seekTo(at);
  }
  loadRecord(key);
}

function seekTo(t) {
  const vid = $('video');
  const go = () => { try { vid.currentTime = Math.max(0, t); vid.play().catch(() => {}); } catch {} };
  if (vid.readyState >= 1) go();
  else vid.addEventListener('loadedmetadata', go, { once: true });
}

function busy(on, text, pct) {
  $('screenBusy').hidden = !on;
  if (text) $('busyText').textContent = text;
  $('busyBar').style.width = (pct === undefined ? (on ? 8 : 100) : pct) + '%';
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
        busy(true, st.note || 'could not fetch this video from the channel', 0);
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
  if (!list.length) {
    panel.appendChild(h('p', { class: 'hint',
      text: 'No indexed passages for this video yet.' }));
    return;
  }
  for (const m of list) {
    panel.appendChild(h('div', {
      class: 'mrow', 'data-t': m.t_start === null ? '' : m.t_start,
      onclick: () => { if (m.t_start !== null && m.t_start !== undefined) seekTo(m.t_start); },
    },
      h('span', { class: 't', text: m.t_start === null || m.t_start === undefined
        ? '—' : timecode(m.t_start) }),
      h('span', { class: 'rail', style: `background:${color(m.source)}` }),
      h('span', { class: 'txt' }, marked(m.text, S.query))));
  }
}

function markActiveCard() {
  const key = S.video && S.video.video_key;
  $$('#cards .card').forEach(c =>
    c.dataset.active = String(c.dataset.key === key));
}

function closePlayer() {
  clearInterval(statePoll);
  clearTimeout(retryTimer);
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
      wrap.appendChild(rowTable(rel.columns, rel.rows.map(r => rel.columns.map(c => r[c]))));
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

function rowTable(columns, rows, types) {
  const t = h('table', { class: 'grid-table' });
  const head = h('tr', {});
  columns.forEach(c => head.appendChild(h('th', { text: c })));
  t.appendChild(head);
  for (const row of rows) {
    const tr = h('tr', {});
    row.forEach((cell, i) => {
      if (cell === null || cell === undefined) {
        tr.appendChild(h('td', {}, h('span', { class: 'null', text: 'null' })));
        return;
      }
      const numeric = typeof cell === 'number' ||
        (types && /INT|REAL|NUM|FLOAT|DOUBLE/i.test(types[i] || ''));
      const text = typeof cell === 'object' ? JSON.stringify(cell) : String(cell);
      tr.appendChild(h('td', { class: numeric ? 'num' : '' },
        h('span', { class: 'cell', text })));
    });
    t.appendChild(tr);
  }
  return t;
}

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
  const p = new URLSearchParams({
    limit: String(LIB_LIMIT), offset: String(S.lib.offset),
    sort: $('libSort').value, has: $('libHas').value, q: $('libQ').value.trim(),
  });
  if (S.lib.creator) p.set('creator', S.lib.creator);
  if (S.lib.category) p.set('category', S.lib.category);

  try {
    const data = await api('/api/library?' + p.toString());
    S.lib.rows = S.lib.rows.concat(data.results || []);
    S.lib.total = data.total || 0;
    S.lib.offset = S.lib.rows.length;
    renderLibrary(reset);
  } catch (e) { toast('Library failed: ' + e.message); }

  if (!S.facets) loadFacets();
}

function renderLibrary(reset) {
  const grid = $('libGrid');
  if (reset) grid.textContent = '';
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
      const card = h('div', { class: 'tcard', onclick: () => openTable(t.name) },
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

async function openTable(name, offset) {
  S.browse.table = name;
  S.browse.offset = offset || 0;
  $('browser').hidden = false;
  $('browserTitle').textContent = name;
  const p = new URLSearchParams({
    limit: '50', offset: String(S.browse.offset), q: S.browse.q,
  });
  try {
    const data = await api(`/api/table/${encodeURIComponent(name)}?` + p.toString());
    const table = rowTable(data.columns, data.rows, data.types);
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
    box.appendChild(h('div', {
      class: 'feed', style: `border-left-color:${color(s.source)}`,
      title: s.via ? `joined through ${s.via}` : `keyed on ${s.key}`,
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
  hits: [], hitIndex: -1,
  labelBoxes: [],
  traceFrom: null,            // first end of a "how are these two related" ask
  pathSet: null, pathEdges: null,
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

function gcolor(node) {
  if (!node) return '#6E7F8C';
  if (node.kind === 'tag') {
    const s = node.meta && node.meta.source;
    return RAW_SOURCE_COLOR[s] || KIND_COLOR.tag;
  }
  return KIND_COLOR[node.kind] || '#6E7F8C';
}

function gradius(node) {
  // Degree decides size, on a log curve: a creator with 400 videos should read
  // as bigger than one with 4, not a hundred times bigger.
  const w = Math.max(1, Number(node.weight) || 1);
  if (node.kind === 'video') return 9 + Math.min(7, Math.log2(w + 1) * 1.6);
  return 7 + Math.min(19, Math.log2(w + 1) * 3.4);
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
  for (const n of G.nodes.values()) { n.deg = 0; n.r = gradius(n); }
  for (const e of G.edges.values()) {
    const a = G.nodes.get(e.src), b = G.nodes.get(e.dst);
    if (a) a.deg++;
    if (b) b.deg++;
  }
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

function gtick() {
  const nodes = gliveNodes();
  if (!nodes.length) return;
  const alpha = G.alpha;
  const root = gtree(nodes);

  for (const n of nodes) {
    if (root) grepel(root, n, GRAPH_TICK.charge * alpha);
    n.vx += -n.x * GRAPH_TICK.gravity * alpha;
    n.vy += -n.y * GRAPH_TICK.gravity * alpha;
  }

  for (const { a, b } of gliveEdges()) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.hypot(dx, dy) || 0.01;
    // Longer rest length for hubs so their satellites form a readable ring
    // instead of a solid disc of overlapping labels.
    const rest = 58 + a.r + b.r + Math.min(120, (a.deg + b.deg) * 0.9);
    const pull = (d - rest) * GRAPH_TICK.spring * alpha;
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
  for (const n of ranked) {
    const p = gtoScreen(n.x, n.y);
    const r = n.r * k;
    if (p.x < -40 || p.y < -40 || p.x > w + 40 || p.y > h + 40) continue;
    const must = (G.sel && G.sel.id === n.id) || (G.hover && G.hover.id === n.id);
    if (!must && r < 9) continue;
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
}

function gcollides(box) {
  for (const b of G.labelBoxes) {
    if (box.x < b.x + b.w && box.x + box.w > b.x &&
        box.y < b.y + b.h && box.y + box.h > b.y) return true;
  }
  return false;
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

function gposter(key, r) {
  if (r < 11) return null;                   // too small to be worth a request
  const held = G.posters.get(key);
  if (held) return held;
  const img = new Image();
  img.decoding = 'async';
  img.onload = () => gdraw();
  img.onerror = () => G.posters.set(key, 'no');
  G.posters.set(key, img);
  img.src = `/api/poster/${encodeURIComponent(key)}`;
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
      wrap.appendChild(rowTable(cols, rec.rows.map(r => cols.map(c => r[c]))));
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
      src: `/api/poster/${encodeURIComponent(row.video_key)}`,
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
        rel.rows.map(r => rel.columns.map(c => r[c]))));
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
      class: 'btn', onclick: () => { showTab('data'); openTable(node.label, 0); },
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
      wrap.appendChild(rowTable(cols, rec.rows.map(r => cols.map(c => r[c]))));
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
   STATUS
   ════════════════════════════════════════════════════════════════════════ */
let statusTimer = 0;
let lastPhase = '';

async function pollStatus() {
  try {
    const st = await api('/api/status');
    S.status = st;
    paintPulse(st);

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
    if (ev.key === 'Escape' && S.video && document.activeElement !== $('q')) closePlayer();
  });

  $$('.tabs button').forEach(b =>
    b.addEventListener('click', () => showTab(b.dataset.tab)));
  document.querySelector('.brand').addEventListener('click', (ev) => {
    ev.preventDefault(); showTab('search');
  });
  $('pulse').addEventListener('click', () => showTab('sources'));

  $('moreBtn').addEventListener('click', () => runSearch(S.query, { append: true }));
  $('libMoreBtn').addEventListener('click', () => loadLibrary(false));
  $('libSort').addEventListener('change', () => loadLibrary(true));
  $('libHas').addEventListener('change', () => loadLibrary(true));

  let libTimer = 0;
  $('libQ').addEventListener('input', () => {
    clearTimeout(libTimer);
    libTimer = setTimeout(() => loadLibrary(true), 220);
  });

  let browseTimer = 0;
  $('browserQ').addEventListener('input', (ev) => {
    S.browse.q = ev.target.value.trim();
    clearTimeout(browseTimer);
    browseTimer = setTimeout(() => openTable(S.browse.table, 0), 260);
  });
  $('browserClose').addEventListener('click', () => { $('browser').hidden = true; });

  gwire();

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
  showTab(tab, { push: false });
  const q = params.get('q');
  if (q) { $('q').value = q; runSearch(q); }
  const v = params.get('v');
  if (v) {
    api(`/api/video/${encodeURIComponent(v)}`).then(data => {
      const m = data.meta || {};
      openVideo({
        video_key: data.video_key, title: m.title || m.caption || data.video_key,
        caption: m.caption, creator: m.creator, category: m.category,
        duration: m.duration, width: m.width, height: m.height, likes: m.likes,
        created_at: m.created_at, msg_id: m.msg_id,
        moment_count: m.moment_count, hit_count: 0, moments: data.moments || [],
      }, null);
    }).catch(() => {});
  }
  loadFacets();
}

if (document.readyState === 'loading')
  document.addEventListener('DOMContentLoaded', start);
else start();
