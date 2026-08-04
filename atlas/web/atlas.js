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
    tab: ['search', 'library', 'data', 'sources'].includes(tab) ? tab : 'search',
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
