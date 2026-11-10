import os
import json
import re
from collections import defaultdict

FRAME_DIRS = [
    "data/2024_Paxos_Frames/cropped_matched_frames",
]

CLIP_DIRS = [
    "data/ensemble_results_paxos2025/cleaned_videos",
]


def get_core_id_from_video(filename):
    # Removes extension and the extractor's 'CLEAN_' prefix to get the base ID.
    return re.sub(r'^CLEAN_', '', os.path.splitext(filename)[0])


# Frame naming convention, e.g.:
#   '2024_02_11_12_26_IMG_4608 LE MILD NPDR_frame31_score0.9990.png' -> '2024_02_11_12_26_IMG_4608 LE MILD NPDR'
FRAME_SUFFIX_RE = re.compile(r'_frame\d+_score[0-9.]+\.png$')


def get_core_id_from_image(filename):
    return FRAME_SUFFIX_RE.sub('', filename)


# Repeat recordings of an eye ('R012L3') are merged into one clip ('R012L', data/merge_repeats.py)
R_REPEAT_RE = re.compile(r'^(R\d+[LR])\d+$')


# 1. Get all files across all provided dirs
# clips/images become (filename, source_dir) pairs so we know where each file lives
clips = []
for d in CLIP_DIRS:
    clips.extend((f, d) for f in os.listdir(d) if f.endswith('.mp4'))

images = []
for d in FRAME_DIRS:
    images.extend((f, d) for f in os.listdir(d) if f.endswith('.png'))

# Sanity check: warn if the same clip/image filename shows up in more than one dir
clip_names_seen = {}
for name, d in clips:
    clip_names_seen.setdefault(name, []).append(d)
duplicate_clips = {name: dirs for name, dirs in clip_names_seen.items() if len(dirs) > 1}
if duplicate_clips:
    print(f"WARNING: {len(duplicate_clips)} clip filename(s) found in multiple CLIP_DIRS: {duplicate_clips}")

image_names_seen = {}
for name, d in images:
    image_names_seen.setdefault(name, []).append(d)
duplicate_images = {name: dirs for name, dirs in image_names_seen.items() if len(dirs) > 1}
if duplicate_images:
    print(f"WARNING: {len(duplicate_images)} image filename(s) found in multiple FRAME_DIRS: {duplicate_images}")

# 2. Map Core IDs to their full video filename(s)
# Build as core_id -> [full_clip_names...] first so we can detect collisions
# (two different clip filenames reducing to the same core id), which the
# original dict comprehension would have silently resolved by letting the
# last one win.
core_id_to_clips = defaultdict(list)
for c, _ in clips:
    core_id_to_clips[get_core_id_from_video(c)].append(c)

ambiguous_core_ids = {cid: names for cid, names in core_id_to_clips.items() if len(names) > 1}
if ambiguous_core_ids:
    print(f"WARNING: {len(ambiguous_core_ids)} core id(s) map to multiple clip files; "
          f"these will be treated as unmatchable (ambiguous): {ambiguous_core_ids}")

# Only unambiguous core ids get a usable lookup entry
video_lookup = {cid: names[0] for cid, names in core_id_to_clips.items() if len(names) == 1}

# 3. Initialize results
matches = {c: [] for c, _ in clips}
matched_images = set()

# Track which video(s) each image got assigned to, so we can catch any image
# that ends up matched into more than one clip's list.
image_match_targets = defaultdict(list)

# 4. Perform Strict Matching
for img, _ in images:
    img_core = get_core_id_from_image(img)
    if img_core not in core_id_to_clips and R_REPEAT_RE.match(img_core):
        img_core = R_REPEAT_RE.match(img_core).group(1)

    if img_core in ambiguous_core_ids:
        # Core id exists but is ambiguous among multiple clip files; skip.
        continue

    # Check for an exact match in the video dictionary
    if img_core in video_lookup:
        full_clip_name = video_lookup[img_core]
        matches[full_clip_name].append(img)
        matched_images.add(img)
        image_match_targets[img].append(full_clip_name)

# Sanity check: no image should ever be matched into more than one clip's list
multiply_matched_images = {img: targets for img, targets in image_match_targets.items() if len(targets) > 1}
if multiply_matched_images:
    print(f"WARNING: {len(multiply_matched_images)} image(s) matched to multiple clips "
          f"(this indicates a bug, not expected data): {multiply_matched_images}")

# 5. Format for JSON output
# We use the filename without extension as the key to stay consistent
final_matches = {get_core_id_from_video(k): v for k, v in matches.items() if v}

results = {
    "matches": final_matches,
    "unmatched_clips": [get_core_id_from_video(c) for c, _ in clips if not matches[c]],
    "unmatched_images": [img for img, _ in images if img not in matched_images],
}

# 6. Save
output_path = "data/matching_results_2025.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"--- Matching Complete ---")
print(f"Matched Clips:  {len(results['matches'])}")
print(f"Unmatched Clips: {len(results['unmatched_clips'])}")
print(f"Unmatched Imgs:  {len(results['unmatched_images'])}")
