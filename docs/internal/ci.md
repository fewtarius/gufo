# CI workflows

What runs on a pull request, on `unstable` and at release, and which checks
gate a merge. All workflow files live in `.github/workflows/`. Job and step
names start with a verb; a reusable workflow's jobs appear as
`caller / called job`.

| File | Display name | Job | Runner | When |
| --- | --- | --- | --- | --- |
| `checks.yml` | Repository checks | Run CPU and repository checks | GitHub Ubuntu | PRs to `unstable`, pushes to `unstable`, manual. Bounded CPU, repository and functional-harness tests. |
| `checks.yml` | Repository checks | Warm Linux package cache | GitHub Ubuntu, via `linux-build.yml` | Not on a pull request. Builds the production package to fill the shared cache. |
| `linux-build.yml` | Linux build | Linux package | GitHub Ubuntu | Reusable. Cache warming from `checks.yml`, and candidate and baseline packages for comparisons. |
| `windows-build.yml` | Windows build | Windows package | GitHub Windows | Reusable. Builds the native application, runs CPU contracts and verifies the bundle. Called by Windows validation. |
| `devlab-windows.yml` | Windows validation | Build, then Run Windows model checks | GitHub Windows, then Devlab Windows Strix Halo | PRs to `unstable`, pushes to `unstable`, manual. GPU diagnostics and model smoke checks on the same run's bundle. |
| `devlab-linux.yml` | Linux validation | Authorize and select checks | GitHub Ubuntu | Any non-draft PR update and manual runs, on upstream's repository only. Pins revisions and plans the affected checks. |
| `devlab-linux.yml` | Linux validation | Build | GitHub Ubuntu, via `linux-build.yml` | When the plan selects work. Builds the candidate and baseline packages. |
| `devlab-linux.yml` | Linux validation | Run Linux model checks | Devlab Linux Strix Halo | Prepares while the hosted build runs, then tests once the packages arrive. |
| `devlab-linux.yml` | Linux validation | Publish results | GitHub Ubuntu | After testing, failures included. Publishes the report and the `devlab/functional` status. |
| `devlab-comment.yml` | Functional reporting | Update PR comment | GitHub Ubuntu | When Linux validation starts or ends. Keeps the PR comment current, also for forks, without running PR code. |
| `pr-title.yml` | PR title | Validate PR title | GitHub Ubuntu | PR opened, edited, reopened, relabeled or updated against `unstable`. |

This fork does not run upstream's `release-please.yml` or `docs-website.yml`.
Both drove upstream's release pull requests and its website, and the fork has
neither; [releasing](releasing.md) describes what replaces them.

The three Devlab jobs are gated on `github.repository == 'gufo-org/gufo'`,
because they run on upstream's self-hosted Strix Halo runners and fetch a
revision from upstream. In this fork those jobs are skipped: Linux validation
reports every job as skipped, `Run Windows model checks` never starts, and
`devlab/functional` never appears as a status. Windows validation still builds
the package and runs `check-pr`, because that part is a hosted job.

## What gates a merge

Neither `stable` nor `unstable` has branch protection configured, so nothing is
enforced by GitHub. The rule the maintainer applies:

- **Repository checks** and **Windows validation** are the gate. Both run on a
  pull request into `unstable`, and both must be green before a pull request is
  squash-merged.
- **Functional evidence comes from the editing host.** Because Linux validation
  is skipped here, the model and functional tiers in
  [testing](testing.md#text-api-functional-tests) have to be run locally and
  belong in the pull request description. On upstream, that same tier reports
  as `devlab/functional`: it fails on correctness failures and on phases
  consistently slower than the baseline, and noisy but complete comparisons pass
  with a visible warning.
- `stable` advances only after both gating workflows pass on the commit being
  promoted.

A required context is the caller and leaf job name, not the workflow display
name. If a ruleset is added later, renaming a job means updating it,
`devlab-comment.yml`'s `workflow_run` subscription and any consuming workflow in
the same change.

## Caches

- **Repository checks** use the compiler cache, Cachix and the Magic Nix Cache.
  Each run saves under a fresh key, so the Actions cache stays current.
- **Linux package builds** use the package compiler cache and Cachix. The
  warming job exists because pull request builds don't push to Cachix.
- **Windows builds** cache the SDK archive, the compiler cache and the vcpkg
  binary cache.
- **Devlab runners** download the verified bundles that the hosted builds
  produce; Windows uses the bundle from the same workflow run.

Native Windows `check-pr` runs every Windows test from the platform-only build
with assertions enabled. A MinGW Nix package remains for local use only.
