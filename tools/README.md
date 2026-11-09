# Precomputed documentation examples

Heavy examples are listed in `examples/precomputed_runs.toml`. Normal docs
builds render their source without executing it. Until results are published,
the page displays a note explaining that results are pending.

To update the hosted results:

1. In GitHub Actions, select **Regenerate documentation examples** and choose
   **Run workflow** on the desired source branch or tag.
2. Leave `paths` empty for all registered examples, or enter space-separated
   paths such as `contact/tube_compression.py plasticity/shear_test.py`.
   Split large batches into multiple runs to stay within the six-hour job limit.
3. The workflow runs the examples headless, saves the media and software/commit
   records as an artifact, then commits them to `3MAH/fedoo-docs`, branch `main`,
   under `media/examples/`. It reuses the `FEDOO_DOC_TOKEN` secret used by the
   documentation workflow; that token needs write access to the docs repository.
   Existing results for examples outside the batch are preserved.
4. Run **Documentation** on the same source ref to rebuild the gallery pages.
   The build reads the published `records.json` and links the hosted animations.

A failed generation does not publish its batch. Its completed media and records
remain available in the workflow artifact for 14 days. Fix the failing example
and rerun it or a smaller batch. No results or metadata need to be committed back
to the source branch.

Locally, use `python tools/run_precomputed.py --list` to list registered examples,
then `python tools/run_precomputed.py <path> --media-dir <directory>` to generate
media and `records.json`. This also updates the local TOML run records. Set
`FEDOO_PRECOMPUTED_RECORDS` to the absolute path of a downloaded `records.json`
when building docs locally with published records.
