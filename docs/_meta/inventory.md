# Documentation inventory

The authoritative inventory is [`inventory.csv`](inventory.csv). It classifies
every repository Markdown file by domain, document type, state, authority,
publication status, migration destination, and next action.

Run the validation after adding, moving, or reclassifying documentation:

```bash
python scripts/check_docs.py
python scripts/check_markdown_links.py
```

An unclassified Markdown file, duplicate entry, missing file, invalid enum, or
non-normalized path fails validation. This makes documentation placement part
of the change that introduces a page rather than a cleanup task deferred to a
later release.

## Reading the classifications

- `authoritative` means the page owns facts in its declared scope.
- `derived` means the page is navigation or a projection of another source.
- `record` means the page preserves a decision, benchmark, or history.
- `transitional` means the page currently mixes roles and is queued to split.

The `destination` column records the intended stable home. A destination does
not claim that migration is complete. The `action` column distinguishes pages
that can be retained from those that must be split, archived, moved with
compatibility, generated, or excluded from publication.
