# Releasing

How this fork relates to upstream's versions, how a pull request title becomes
a commit, how an upstream release is adopted and how the fork tags one. What a
version promises to users is in [versions](../versions.md).

This fork publishes no binaries. The container image, the Debian package and the
Windows ZIP belong to upstream [gufo-org/gufo](https://github.com/gufo-org/gufo);
the fork ships source. The fork's release automation was removed because it
drove upstream's release pull requests, Toolboxes' publishing and the website
sync, none of which exist here.

## Pull request titles

A pull request is squash-merged into `unstable`, so its title becomes the one
commit that lands. Titles have this form:

```text
<type>[(optional-scope)][!]: <description>
```

The accepted types are:

| Type | Purpose | Release effect |
| --- | --- | --- |
| `fix` | Correct user-visible behavior | Patch |
| `perf` | Improve user-visible performance | Patch |
| `feat` | Add backward-compatible functionality | Minor |
| `docs` | Change documentation only | None |
| `test` | Change tests only | None |
| `build` | Change the build system or dependencies | None by default |
| `ci` | Change automation only | None |
| `chore` | Repository maintenance | None |
| `refactor` | Restructure code without changing behavior | None |
| `style` | Change formatting without changing behavior | None |
| `revert` | Revert an earlier change | Determined from the reverted change |

Use a short, lowercase scope when it adds ownership context. Examples:

```text
fix(cache): preserve image prefixes after cancellation
perf(qwen): make prompt tokenization linear
feat(server): support constrained JSON output
docs: explain the release policy
feat(server)!: remove the legacy completion endpoint
```

Mark a breaking change with `!` before the colon; a footer in the body is lost
in the squash. `tools/ci/check-pr-title.py` enforces this table, and its test
reads the table and examples from this page.

## Versions

Semantic Versioning applies to the documented CLI, HTTP API and public Nix flake
outputs. Before 1.0.0, `fix` and `perf` bump the patch, and `feat` and breaking
changes bump the minor. After 1.0.0, breaking changes bump the major; 1.0.0
itself is an explicit decision, never automatic.

The version number belongs to upstream. `version.txt` holds the upstream release
the tree is based on, and adopting an upstream release is what moves it. The
fork's own work does not raise it, because the fork has no release line of its
own to version.

Ordinary pull requests never edit `version.txt` or `CHANGELOG.md`.

## Adopting an upstream release

1. `git fetch upstream` and merge `upstream/main` into a branch off `unstable`.
2. Resolve conflicts. They land in the upstream files the fork edits; the list
   is in [FORK.md](../../FORK.md). Keep the fork's edits as their own commits so
   the resolution stays a reviewable diff.
3. Take `version.txt` and `CHANGELOG.md` from upstream. They describe upstream's
   release, not the fork's work.
4. Run `nix flake check` and the affected model suites. A shared-infrastructure
   change needs the GPU and functional tiers, not just the hosted CPU gate.
5. Merge into `unstable`, then fast-forward `stable` to it.
6. Tag `stable` with upstream's version, `vX.Y.Z`, and push the tag. The tag
   records which upstream release the fork adopted and where its own models
   stood; `git log --oneline vPREV..vX.Y.Z` is the fork's change list for it.

A tag is created once and never moves. If a tag turns out to be wrong, delete it
before anyone fetches it and cut it again from a corrected `stable`.

## Tracking upstream between releases

Upstream commits land on `unstable` as they become worth taking, not only at
release time. Keep each upstream merge as its own commit with upstream's
subject, so `git log --first-parent` still reads as the fork's history and the
merges stay separable from fork work.

`main` does not exist in this repository. Upstream history is reachable through
the `upstream` remote, and a mirror branch only invited the two to disagree.

## Extension API

The extension API is versioned separately from Gufo and does not follow these
tags. CMake derives its revision from the active series in
`include/gufo/ext/api-series.json` plus the count of records under
`include/gufo/ext/changes/<major>.<minor>/`; a change to a public contract adds a
record. See [the extension API](../extension-api.md#versions) for what that
promises.
