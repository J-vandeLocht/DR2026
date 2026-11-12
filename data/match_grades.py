import json
import csv
import os
import re
from collections import defaultdict

matching_file = "data/matching_results_2025.json"
# Grade sources (path, id column, grade column); campR = sheet 2 of paxos_grading_campR.xlsx (R-series only).
grade_csvs = [
    ("data/DR_Grading_Summary_v3.csv", "File_Name", "Grading"),
    ("data/DR_Grading_campR.csv", "Eye_ID", "Paxos_DR"),
]


def normalize_id(s):
    # Cleans an already-bare ID (e.g. CSV File_Name, or an extracted ID) down
    # to just uppercase alphanumerics, so formatting quirks (spacing, case,
    # stray punctuation) don't cause false mismatches.
    if not s:
        return ""
    return re.sub(r'[^a-zA-Z0-9]', '', s).upper()


# Clip/video/image filenames are full descriptive names (e.g.
# 'IMG_5063 RE HEALTHY', '2024_06_14_13_19_IMG_3502', 'R034L2'), but the
# CSV's File_Name column is just the bare ID ('5063', '3502', 'R027R').
# This pulls that bare ID back out of a descriptive name.
IMG_ID_RE = re.compile(r'IMG_?(\d+)', re.IGNORECASE)

# Repeat recordings of an eye ('R008R2', 'R034L3') are separate clips graded by their eye's entry ('R008R').
R_CODE_SUFFIX_RE = re.compile(r'^(R\d+[LR])\d+$')


def extract_file_id(name):
    name = os.path.splitext(name)[0]  # strip extension if present
    match = IMG_ID_RE.search(name)
    if match:
        return match.group(1)
    # No 'IMG_<digits>' pattern found (e.g. R-coded names like 'R027R') --
    # use the whole name as-is; normalize_id() will clean it up.
    return name


# Frame naming convention, e.g.:
#   '2024_02_11_12_26_IMG_4608 LE MILD NPDR_frame31_score0.9990.png' -> '2024_02_11_12_26_IMG_4608 LE MILD NPDR'
FRAME_SUFFIX_RE = re.compile(r'_frame\d+_score[0-9.]+\.png$')


def get_video_id_from_image(filename):
    return FRAME_SUFFIX_RE.sub('', filename)


# 1. Load the matching results (matches + the two unmatched lists)
with open(matching_file, "r") as f:
    matching_results = json.load(f)

matches_data = matching_results["matches"]
unmatched_clips = matching_results["unmatched_clips"]
unmatched_images = matching_results["unmatched_images"]

# 2. Load groundtruth from the CSVs into a normalized dict.
csv_grades = {}
grade_sources = {}  # normalized_key -> source label (first row that set it)
row_count = 0

for csv_path, id_col, grade_col in grade_csvs:
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            file_name_raw = row[id_col].strip()
            normalized_key = normalize_id(file_name_raw)
            grade = int(row[grade_col])
            source = f"{os.path.basename(csv_path)} ({id_col}={file_name_raw!r})"

            if normalized_key in csv_grades:
                if csv_grades[normalized_key] != grade:
                    print(f"WARNING: conflicting grade for {normalized_key!r} -- "
                          f"{source} says {grade}, but {grade_sources[normalized_key]} "
                          f"already set {csv_grades[normalized_key]}. Keeping the first value.")
                continue

            csv_grades[normalized_key] = grade
            grade_sources[normalized_key] = source

print(f"\nRead {row_count} CSV rows -> {len(csv_grades)} unique groundtruth entries "
      f"({row_count - len(csv_grades)} collapsed as duplicates, conflicts are warned above).\n")

key_usage = defaultdict(list)  # normalized CSV key -> list of (source, id) that resolved to it (exact matches only now)


def find_grade(query_id):
    if query_id in csv_grades:
        return csv_grades[query_id], query_id
    # Repeat recording: fall back to the eye's grade
    match = R_CODE_SUFFIX_RE.match(query_id)
    if match and match.group(1) in csv_grades:
        return csv_grades[match.group(1)], match.group(1)

    return None, None


# 3. Create the two label maps
final_clip_labels = {}
final_image_labels = {}
ungraded_clip_ids = []  # matched clips (i.e. have >=1 image) with no CSV grade
graded_clips_without_images = []  # labeled for the video pipeline, but no stills for the frame classifier

# Unmatched clips (no still photos) are labeled too: the video-level pipeline doesn't need stills.
for clip_id in list(matches_data) + unmatched_clips:
    image_list = matches_data.get(clip_id, [])
    query_id = normalize_id(extract_file_id(clip_id))
    grade, matched_key = find_grade(query_id)

    if grade is not None:
        final_clip_labels[f"{clip_id}.mp4"] = grade
        key_usage[matched_key].append(("matched clip" if image_list else "unmatched clip", clip_id))
        for img_name in image_list:
            final_image_labels[img_name] = grade
        if not image_list:
            graded_clips_without_images.append(clip_id)
    elif image_list:
        ungraded_clip_ids.append(clip_id)
        print(f"Warning: No grade found for: {clip_id} (Extracted ID: {query_id})")

# 4. Overview: where are graded/ungraded clips and images falling through?

# (b) Unmatched images (no video match found during frame matching) whose
#     video *would* have had a grade in the CSV, had it been matched.
unmatched_images_with_graded_video = []
for img_name in unmatched_images:
    video_query_id = normalize_id(extract_file_id(get_video_id_from_image(img_name)))
    grade, matched_key = find_grade(video_query_id)
    if grade is not None:
        unmatched_images_with_graded_video.append(img_name)
        key_usage[matched_key].append(("unmatched image's video", img_name))

# (c) Images that WERE matched to a video, but that video has no grade in
#     the CSV -- i.e. images belonging to one of the ungraded_clip_ids.
images_with_matched_but_ungraded_video = []
for clip_id in ungraded_clip_ids:
    images_with_matched_but_ungraded_video.extend(matches_data[clip_id])

print("--- Grading Coverage Overview ---")
print(f"Graded clips without matched images (video labels only): {len(graded_clips_without_images)}")
if graded_clips_without_images:
    print(f"  -> {graded_clips_without_images}")
print(f"Unmatched images whose video has a CSV grade:        {len(unmatched_images_with_graded_video)}")
if unmatched_images_with_graded_video:
    print(f"  -> {unmatched_images_with_graded_video}")
print(f"Images matched to a video that has no CSV grade:     {len(images_with_matched_but_ungraded_video)}")
if ungraded_clip_ids:
    print(f"  -> from clips: {ungraded_clip_ids}")

reused_keys = {k: v for k, v in key_usage.items() if len(v) > 1}
print(f"\nCSV entries claimed by more than one clip/image (exact normalized-key collision): {len(reused_keys)}")
for k, users in reused_keys.items():
    print(f"  -> {k!r} used by: {users}")

# (d) CSV groundtruth entries that don't correspond to ANY clip file present
#     in this dataset at all -- not in matches_data, not in unmatched_clips.
#     These are grades for videos we simply don't have a clip for here.
all_clip_ids = list(matches_data.keys()) + unmatched_clips
referenced_keys = set()
for clip_id in all_clip_ids:
    query_id = normalize_id(extract_file_id(clip_id))
    _, matched_key = find_grade(query_id)
    if matched_key is not None:
        referenced_keys.add(matched_key)

unreferenced_csv_keys = set(csv_grades.keys()) - referenced_keys
print(f"\nCSV groundtruth entries with no corresponding clip file in this dataset: {len(unreferenced_csv_keys)}")
if unreferenced_csv_keys:
    print(f"  -> {[grade_sources[k] for k in sorted(unreferenced_csv_keys)]}")
print()

# 5. Save
with open("data/clip_labels_2025.json", "w") as f:
    json.dump(final_clip_labels, f, indent=4)

with open("data/image_labels_2025.json", "w") as f:
    json.dump(final_image_labels, f, indent=4)

print("Success!")
print(f"Mapped {len(final_clip_labels)} clips and {len(final_image_labels)} images.")