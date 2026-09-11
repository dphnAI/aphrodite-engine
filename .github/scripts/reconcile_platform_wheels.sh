#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${B2_S3_BUCKET:?B2_S3_BUCKET is required}"
: "${B2_PUBLIC_BASE_URL:?B2_PUBLIC_BASE_URL is required}"
: "${GITHUB_SHA:?GITHUB_SHA is required}"
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
: "${PLATFORM_BACKEND:?PLATFORM_BACKEND is required}"
: "${PLATFORM_ARCHITECTURE:?PLATFORM_ARCHITECTURE is required}"
: "${PLATFORM_VERSION_LOCAL:?PLATFORM_VERSION_LOCAL is required}"
: "${PLATFORM_BUILD_SCRIPT:?PLATFORM_BUILD_SCRIPT is required}"

output_dir="${1:?usage: reconcile_platform_wheels.sh OUTPUT_DIR INDEX_DIR}"
index_root="${2:?missing index output directory}"
rclone="${RUNNER_TEMP}/rclone-bin/rclone"
remote_root="b2:${B2_S3_BUCKET}/platform-wheels/nightly/${PLATFORM_BACKEND}/${PLATFORM_ARCHITECTURE}"
public_root="${B2_PUBLIC_BASE_URL%/}/platform-wheels/nightly/${PLATFORM_BACKEND}/${PLATFORM_ARCHITECTURE}"
state_remote="${remote_root}/_ci/last-built-main.txt"
target_commit="${GITHUB_SHA}"

if [[ ! -x "$rclone" ]]; then
  echo "::error::rclone is not installed at ${rclone}"
  exit 1
fi
if [[ ! -x "$PLATFORM_BUILD_SCRIPT" ]]; then
  echo "::error::Platform build script is not executable: ${PLATFORM_BUILD_SCRIPT}"
  exit 1
fi

encode_path() {
  printf '%s' "${1//+/%2B}"
}

build_with_current_rust_helper() (
  local commit="$1"
  local version="$2"
  # Packaging fixes must also apply to commits awaiting their first wheel.
  trap 'git restore --source="$commit" -- tools/build_rust.py' EXIT
  git show "${target_commit}:tools/build_rust.py" > tools/build_rust.py
  APHRODITE_VERSION_OVERRIDE="$version" "$PLATFORM_BUILD_SCRIPT" "$output_dir"
)

upload_wheel() {
  local wheel="$1"
  local remote="$2"
  local wheel_name
  local -a upload_command
  wheel_name="$(basename "$wheel")"

  for attempt in {1..5}; do
    echo "Uploading ${wheel_name} (attempt ${attempt}/5)"
    upload_command=(
      "$rclone" copyto "$wheel" "$remote"
      --s3-upload-cutoff 16M
      --s3-chunk-size 16M
      --s3-upload-concurrency 8
      --retries 5
      --low-level-retries 10
      --retries-sleep 5s
      --progress
      --stats 30s
      --stats-one-line
    )
    if command -v timeout >/dev/null 2>&1; then
      upload_command=(
        timeout --signal=TERM --kill-after=30s 20m
        "${upload_command[@]}"
      )
    fi
    if "${upload_command[@]}"; then
      return 0
    fi
    echo "Upload attempt ${attempt}/5 failed; retrying in 15 seconds"
    sleep 15
  done

  echo "::error::Wheel upload failed after 5 attempts"
  return 1
}

find_commit_wheel() {
  local commit="$1"
  "$rclone" lsf \
    "${remote_root}/commits/${commit}" \
    --files-only \
    --include '*.whl' 2>/dev/null |
    head -n 1 || true
}

restore_tracked_build_changes() {
  if git diff --quiet && git diff --cached --quiet; then
    return
  fi

  echo "Restoring tracked files modified by the previous build:"
  git status --short --untracked-files=no
  git restore --source=HEAD --staged --worktree -- .
}

write_commit_index() {
  local commit="$1"
  local wheel_name="$2"
  local digest="${3:-}"
  local entries
  local output
  local encoded_name

  entries="$(mktemp "${RUNNER_TEMP}/sonar-platform-entries.XXXXXX")"
  output="$(mktemp "${RUNNER_TEMP}/sonar-platform-index.XXXXXX")"
  encoded_name="$(encode_path "$wheel_name")"
  if [[ -n "$digest" ]]; then
    printf '%s\t%s\t%s\n' \
      "$wheel_name" \
      "${public_root}/commits/${commit}/${encoded_name}" \
      "$digest" >"$entries"
  else
    printf '%s\t%s\n' \
      "$wheel_name" \
      "${public_root}/commits/${commit}/${encoded_name}" >"$entries"
  fi

  python3 .github/scripts/generate_nightly_index.py \
    --entry-file "$entries" \
    --commit "$commit" \
    --title "Sonar ${PLATFORM_BACKEND} wheel for ${commit:0:12}" \
    --description "Precompiled ${PLATFORM_BACKEND} wheel for ${PLATFORM_ARCHITECTURE} at commit ${commit}." \
    --install-command \
    "uv pip install aphrodite-engine --extra-index-url ${public_root}/commits/${commit}/aphrodite-engine --index-strategy first-index" \
    --output "$output"
  "$rclone" copyto \
    "$output" \
    "${remote_root}/commits/${commit}/aphrodite-engine/index.html"
  rm -f "$entries" "$output"
}

last_built=""
if last_built="$("$rclone" cat "$state_remote" 2>/dev/null)"; then
  last_built="${last_built//$'\r'/}"
  last_built="${last_built//$'\n'/}"
fi

base_commit=""
if [[ "$last_built" =~ ^[0-9a-f]{40}$ ]] &&
  git cat-file -e "${last_built}^{commit}" 2>/dev/null; then
  if git merge-base --is-ancestor "$last_built" "$target_commit"; then
    base_commit="$last_built"
  else
    # A force-push can replace already-built commits with equivalent commits
    # that have new IDs. Resume from the histories' common ancestor so every
    # replacement commit receives a wheel.
    base_commit="$(git merge-base "$last_built" "$target_commit" || true)"
  fi
fi

if [[ -z "$base_commit" ]] &&
  [[ "${PUSH_BEFORE:-}" =~ ^[0-9a-f]{40}$ ]] &&
  [[ ! "${PUSH_BEFORE}" =~ ^0+$ ]] &&
  git cat-file -e "${PUSH_BEFORE}^{commit}" 2>/dev/null &&
  git merge-base --is-ancestor "$PUSH_BEFORE" "$target_commit"; then
  base_commit="$PUSH_BEFORE"
fi

if [[ -z "$base_commit" ]] &&
  git rev-parse "${target_commit}^" >/dev/null 2>&1; then
  base_commit="$(git rev-parse "${target_commit}^")"
fi

if [[ -n "$base_commit" ]]; then
  commits=()
  while IFS= read -r commit; do
    commits+=("$commit")
  done < <(git rev-list --reverse "${base_commit}..${target_commit}")
else
  commits=("$target_commit")
fi

echo "Target main commit: ${target_commit}"
echo "Platform: ${PLATFORM_BACKEND}/${PLATFORM_ARCHITECTURE}"
echo "Last contiguous build: ${last_built:-none}"
echo "Commits requiring reconciliation: ${#commits[@]}"

# "${commits[@]}" on a genuinely empty array trips "unbound variable" under
# `set -u` on bash < 4.4 (macOS system /bin/bash is 3.2.57); ${#commits[@]}
# is safe either way, so gate the loop on the count instead.
if ((${#commits[@]} > 0)); then
for commit in "${commits[@]}"; do
  restore_tracked_build_changes
  git checkout --detach "$commit"
  wheel_name="$(find_commit_wheel "$commit")"

  if [[ -n "$wheel_name" ]]; then
    echo "Commit ${commit} already has ${wheel_name}; skipping build"
    write_commit_index "$commit" "$wheel_name"
  else
    tag="$(git describe --tags --abbrev=0 --match 'v[0-9]*')"
    release="${tag#v}"
    IFS=. read -r major minor patch <<<"$release"
    timestamp="$(date -u +%Y%m%d%H%M%S)"
    short_commit="$(git rev-parse --short=9 "$commit")"
    version="${major}.${minor}.$((patch + 1)).dev${timestamp}+${PLATFORM_VERSION_LOCAL}.g${short_commit}"
    echo "Building ${version} from ${commit}"

    build_with_current_rust_helper "$commit" "$version"
    wheel="$(find "$output_dir" -maxdepth 1 -type f -name '*.whl' -print -quit)"
    if [[ -z "$wheel" ]]; then
      echo "::error::The build did not export a wheel for ${commit}"
      exit 1
    fi

    wheel_name="$(basename "$wheel")"
    if command -v sha256sum >/dev/null 2>&1; then
      digest="$(sha256sum "$wheel" | cut -d ' ' -f 1)"
    else
      digest="$(shasum -a 256 "$wheel" | cut -d ' ' -f 1)"
    fi
    upload_wheel \
      "$wheel" \
      "${remote_root}/commits/${commit}/${wheel_name}"

    curl --fail --silent --show-error --location \
      --retry 10 \
      --retry-all-errors \
      --retry-delay 3 \
      --range 0-0 \
      --output /dev/null \
      "${public_root}/commits/${commit}/$(encode_path "$wheel_name")"

    write_commit_index "$commit" "$wheel_name" "$digest"
    rm -f "$wheel"
  fi

  printf '%s\n' "$commit" | "$rclone" rcat "$state_remote"
done
fi

restore_tracked_build_changes
git checkout --detach "$target_commit"

entries="${RUNNER_TEMP}/sonar-${PLATFORM_BACKEND}-${PLATFORM_ARCHITECTURE}.tsv"
: >"$entries"
while IFS= read -r stored_path; do
  [[ -n "$stored_path" ]] || continue
  wheel_name="$(basename "$stored_path")"
  printf '%s\t%s\n' \
    "$wheel_name" \
    "${public_root}/$(encode_path "$stored_path")" >>"$entries"
done < <(
  {
    "$rclone" lsf \
      "${remote_root}/wheels" \
      --files-only \
      --include '*.whl' 2>/dev/null |
      sed 's#^#wheels/#'
    "$rclone" lsf \
      "${remote_root}/commits" \
      --recursive \
      --files-only \
      --include '*.whl' 2>/dev/null |
      sed 's#^#commits/#'
  } | sort -r
)

index_dir="${index_root}/whl/nightly/${PLATFORM_BACKEND}/${PLATFORM_ARCHITECTURE}"
package_dir="${index_dir}/simple/aphrodite-engine"
nightly_dir="${index_root}/nightly/${PLATFORM_BACKEND}/${PLATFORM_ARCHITECTURE}"
mkdir -p "$package_dir" "$nightly_dir"
python3 .github/scripts/generate_nightly_index.py \
  --entry-file "$entries" \
  --commit "$target_commit" \
  --title "Sonar nightly ${PLATFORM_BACKEND} wheels" \
  --description "Precompiled ${PLATFORM_BACKEND} wheels for ${PLATFORM_ARCHITECTURE} from every commit on main." \
  --install-command \
  "uv pip install aphrodite-engine --extra-index-url https://sonar.dphn.ai/whl/nightly/${PLATFORM_BACKEND}/${PLATFORM_ARCHITECTURE}/simple --index-strategy first-index" \
  --output "${index_dir}/index.html"
cp "${index_dir}/index.html" "${package_dir}/index.html"
cp "${index_dir}/index.html" "${nightly_dir}/index.html"
"$rclone" copyto "${index_dir}/index.html" "${remote_root}/index.html"

tip_wheel="$(find_commit_wheel "$target_commit")"
if [[ -z "$tip_wheel" ]]; then
  echo "::error::No ${PLATFORM_BACKEND} wheel is available for ${target_commit}"
  exit 1
fi
"$rclone" copyto \
  "${remote_root}/commits/${target_commit}/aphrodite-engine/index.html" \
  "${remote_root}/aphrodite-engine/index.html"
