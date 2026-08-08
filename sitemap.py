"""
VIOS sitemap — one footer, served once, injected into every page.

Every interface in this system is a separate HTML file written at a different
time: the workstation, the capture tab, the processing tab, Atlas, the admin
panel, the V17 workspace, the Omniscient explorer. Nothing linked them, so
finding a page meant remembering its path — and a page added later was
invisible to anyone who did not already know it existed.

The fix is deliberately not "paste a footer into seven files". That version
goes stale the first time a page is added, and it goes stale silently. Instead
the list lives here, once, and `/sitemap.js` renders it into whatever page
loads the script. Adding a page is one entry in `PAGES` below.

The script is self-contained and dependency-free: it must run inside pages with
wildly different CSS, including two that set `* { box-sizing: border-box }` and
one that styles bare `a`. So every rule it needs is scoped under `.vios-sitemap`
and set explicitly rather than inherited.
"""

from __future__ import annotations

import json

# (path, title, one-line description). Order is the order shown.
PAGES = [
    ("/", "Workstation",
     "The main desk — library, player, queue"),
    ("/capture", "Capture",
     "Instagram → Telegram: import an export, download reels, upload them"),
    ("/process", "Process",
     "The GPU rotation — every pass, every video, the evidence database"),
    ("/atlas", "Atlas",
     "The reader — search the database, find moments, play them"),
    ("/admin", "Admin",
     "Storage, categories, queues, backup and restore, factory reset"),
    ("/v17", "Workspace",
     "Per-video workspace: frames, chunks, notes"),
    ("/omni", "Explorer",
     "God-Mode Explorer — the Omniscient layer's own dashboard"),
    ("/docs", "API",
     "Every endpoint this server exposes, with a form to call it"),
]

_CSS = """
.vios-sitemap{margin:64px 0 0;padding:26px 22px 30px;
  border-top:1px solid rgba(255,255,255,.10);
  background:rgba(255,255,255,.02);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:13px;line-height:1.5;color:#8b93a1;box-sizing:border-box}
.vios-sitemap *{box-sizing:border-box}
.vios-sitemap h2{margin:0 0 14px;padding:0;font-size:11px;font-weight:600;
  letter-spacing:.14em;text-transform:uppercase;color:#5d6570;border:0}
.vios-sitemap ul{list-style:none;margin:0;padding:0;display:grid;gap:2px;
  grid-template-columns:repeat(auto-fill,minmax(232px,1fr))}
.vios-sitemap li{margin:0;padding:0}
.vios-sitemap a{display:block;padding:9px 11px;border-radius:8px;
  text-decoration:none;color:#c8cfda;border:1px solid transparent;
  transition:background .12s,border-color .12s}
.vios-sitemap a:hover{background:rgba(255,255,255,.05);
  border-color:rgba(255,255,255,.10);color:#fff}
.vios-sitemap a.here{background:rgba(255,255,255,.06);
  border-color:rgba(255,255,255,.14);color:#fff}
.vios-sitemap b{display:block;font-weight:600;font-size:13px;color:inherit}
.vios-sitemap span{display:block;font-size:11.5px;color:#6f7885;margin-top:2px}
.vios-sitemap a.here b:after{content:" · you are here";font-weight:400;
  color:#6f7885;font-size:11px}
.vios-sitemap .vios-foot{margin-top:16px;font-size:11px;color:#5d6570}
"""

_JS_TEMPLATE = """/* VIOS sitemap footer — served by sitemap.py, injected into every page. */
(function () {
  if (window.__viosSitemap) return;
  window.__viosSitemap = true;

  var PAGES = __PAGES__;

  function build() {
    if (document.querySelector('.vios-sitemap')) return;

    var style = document.createElement('style');
    style.textContent = __CSS__;
    document.head.appendChild(style);

    // Longest match wins, so /atlas/library marks Atlas rather than the
    // workstation at "/" — which is a prefix of every path there is.
    var path = location.pathname.replace(/\\/+$/, '') || '/';
    var best = '/';
    PAGES.forEach(function (p) {
      var href = p[0];
      if (href === '/') return;
      if ((path === href || path.indexOf(href + '/') === 0)
          && href.length > best.length) best = href;
    });
    if (path !== '/' && best === '/') best = '';

    var nav = document.createElement('nav');
    nav.className = 'vios-sitemap';
    nav.setAttribute('aria-label', 'All pages');
    var items = PAGES.map(function (p) {
      return '<li><a href="' + p[0] + '"' + (p[0] === best ? ' class="here"' : '')
        + '><b>' + p[1] + '</b><span>' + p[2] + '</span></a></li>';
    }).join('');
    nav.innerHTML = '<h2>Every page in this system</h2><ul>' + items + '</ul>'
      + '<div class="vios-foot">VIOS · Video Intelligence OS. '
      + 'One server, one database, these ' + PAGES.length + ' doors into it.</div>';

    (document.body || document.documentElement).appendChild(nav);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', build);
  } else {
    build();
  }
})();
"""


def sitemap_js() -> str:
    """The injected script, with the page list baked in."""
    return (_JS_TEMPLATE
            .replace("__PAGES__", json.dumps(PAGES))
            .replace("__CSS__", json.dumps(_CSS)))
