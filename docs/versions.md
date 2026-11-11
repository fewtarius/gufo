# Versions

Gufo follows Semantic Versioning, so a version number tells you whether an
upgrade can break your setup. This page explains what counts as breaking and
where releases are published.

## What a version promises

The promise covers the documented command line, HTTP API and public Nix flake
outputs. A change is **breaking** when something these pages document stops
working without a change on your side.

Before 1.0.0:

- fixes and speedups raise the patch version: 0.31.0 to 0.31.1;
- new features and breaking changes raise the minor version: 0.31.1 to 0.32.0.

After 1.0.0, breaking changes raise the major version. Upstream's
[changelog](https://github.com/gufo-org/gufo/blob/main/CHANGELOG.md) lists the
engine changes for each version.

Extensions have a separate version; see the
[extension API](extension-api.md#versions).

## Which version this fork is on

This fork follows upstream's version numbers and adds its own models on top.
`version.txt` names the upstream release the tree is based on, and the fork
tags the same number on `stable` once it has adopted that release. A version
therefore means "upstream X.Y.Z, plus the fork's models at their state in that
tag". Compare tags to see what the fork changed between them.

Until a release is tagged, `stable` is the newest qualified state and
`gufo --version` reports the exact Git revision, so a bug report says what ran.

## Where releases go

| Channel | Tags | Contents |
| --- | --- | --- |
| [This fork](https://github.com/fewtarius/gufo/tags) | `vX.Y.Z` | Source at that revision |
| [Upstream releases](https://github.com/gufo-org/gufo/releases) | `vX.Y.Z` | Upstream source, Debian package and Windows ZIP |
| Container images | `X.Y.Z`, `X.Y`, `latest` | Upstream builds only; they do not contain this fork's models |
| Nix flake | `github:fewtarius/gufo/vX.Y.Z` | The fork at that tag |

A released version never changes. The container images and the Debian package
come from upstream, so use them for upstream's models and build this fork from
source for the models it adds; [install](install.md) covers both.

## Which documentation to read

[`docs/`](README.md) in this repository is the documentation for this fork, and
it describes the code in the same revision. Upstream publishes its own copy at
[gufo.blog/docs](https://gufo.blog/docs), one version per upstream release,
which does not cover the fork's models.
