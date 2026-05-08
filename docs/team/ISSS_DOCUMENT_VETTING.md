# ISSS document vetting — URLs to verify by hand

The Day-5b infrastructure (schema + render layer) ships ready. The remaining
work is manual URL verification you (Mukul) need to do, since fabricating
links would defeat the point of "go to source."

For each of the six documents below:

1. Visit the URL.
2. Confirm it loads, is the right document, and is publicly accessible (no
   terpconnect login). If login is required, set `requires_login: true`.
3. Open browser DevTools → Network → response headers. If
   `X-Frame-Options` is `DENY` or `SAMEORIGIN`, the iframe preview will not
   render — set `embeddable: false` and the card will show a link only.
   If `Content-Security-Policy: frame-ancestors` is present and excludes
   `localhost` / your demo origin, treat as not embeddable.
4. Paste the entry into the appropriate ISSS service node in
   `data/curated/service_graph.jsonl`. The node's `documents` field is a
   list of `{title, url, document_type, embeddable, requires_login}` dicts.

## The six URLs

| Topic | What to find | Attach to service_id |
|---|---|---|
| 1. CPT request form | The current ISSS Curricular Practical Training application form. Likely `globalmaryland.umd.edu/.../cpt` or a Terrapin Service Network link. | `umd_isss_opt_cpt` |
| 2. OPT application guide | ISSS's step-by-step OPT timeline / application instructions. | `umd_isss_opt_cpt` |
| 3. Academic Difficulty / Reduced Course Load policy | The page or PDF explaining when an F-1 student can drop below full-time enrollment. | `umd_isss_academic_difficulty` |
| 4. Travel Signature info | What it is, when to request, processing time. | `umd_isss` |
| 5. SEVIS Reinstatement guide | What to do if status has lapsed. | `umd_isss_academic_difficulty` |
| 6. ISSS staff directory | Page listing advisors and contact channels. | `umd_isss` |

## Document object schema

```json
{
  "title": "OPT Application Guide",
  "url": "https://globalmaryland.umd.edu/...",
  "document_type": "guide",
  "embeddable": false,
  "requires_login": false
}
```

`document_type` is one of: `form` · `policy` · `guide` · `directory`.

## Example: how it appears in service_graph.jsonl

Existing row (one line, abbreviated):

```json
{"service_id": "umd_isss_opt_cpt", "resource_name": "ISSS Employment & OPT/CPT Authorization", ..., "documents": []}
```

After vetting OPT guide + CPT form, replace `"documents": []` with:

```json
"documents": [
  {"title": "OPT Application Guide", "url": "https://...", "document_type": "guide", "embeddable": false, "requires_login": false},
  {"title": "CPT Request Form", "url": "https://...", "document_type": "form", "embeddable": false, "requires_login": false}
]
```

Stay on one line per row in JSONL.

## What renders

When `documents` is non-empty on a retrieved source, the support card shows
a `Read directly` section beneath the resource link. Each document gets:

- A small uppercase type badge (`form`, `policy`, etc.)
- A link to the document (opens in a new tab)
- An optional inline iframe preview (only if `embeddable: true && requires_login: false`), hidden behind a `<details>` toggle
- A "(terpconnect login)" note if `requires_login: true`

When `documents` is empty (the default for all non-ISSS resources), the
card looks exactly like before — no behavioral change for non-F-1 cells.

## Why not auto-host

The student needs to read ISSS's *current* policy, not a snapshot we
captured. Embedding the live UMD URL means policy changes propagate
automatically; self-hosting would create a freshness liability worse than
the current "last_verified" date staleness.

## Once URLs are verified

After updating the JSONL, restart the demo. No code changes required.
