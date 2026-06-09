# Vendored nf-core schemas (test fixtures)

These are **real, unmodified** `nextflow_schema.json` files fetched from the
nf-core pipeline repositories. They are committed verbatim so the parsing and
classification logic is validated against authentic artifacts we did **not**
author — guarding against tests that merely pass because they were written
around synthetic inputs.

| File | Source | Schema convention |
|------|--------|-------------------|
| `rnaseq_3.14.0.json` | https://github.com/nf-core/rnaseq (tag `3.14.0`) | `definitions` |
| `sarek_3.5.1.json`   | https://github.com/nf-core/sarek (tag `3.5.1`)   | `$defs` |
| `demo_1.0.1.json`    | https://github.com/nf-core/demo (tag `1.0.1`)    | `$defs` |

Both schema conventions are represented on purpose: nf-core migrated the
JSON-Schema definitions block from `definitions` to `$defs` (between rnaseq
3.16 and 3.18), and the parser must handle both.

nf-core pipelines are MIT-licensed; these files are redistributed here under the
same terms for testing purposes only. To refresh, re-fetch from the URLs above
at the pinned tag.
