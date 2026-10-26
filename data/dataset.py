import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import cv2
import numpy as np
import json
import os
import re
from collections import Counter


class FrameDataset(Dataset):
    def __init__(self, json_path, img_dirs, transform=None):
        with open(json_path, 'r') as f:
            label_data = json.load(f)

        self.filename_to_path = {}
        for img_dir in img_dirs:
            for fname in os.listdir(img_dir):
                full_path = os.path.join(img_dir, fname)
                if fname in self.filename_to_path:
                    print(f"WARNING: '{fname}' found in multiple img_dirs -- "
                          f"keeping '{self.filename_to_path[fname]}', "
                          f"ignoring '{full_path}'.")
                    continue
                self.filename_to_path[fname] = full_path

        self.image_paths = []
        self.grades = []
        self.labels = []
        missing = []

        for img_name, grade in label_data.items():
            if img_name not in self.filename_to_path:
                missing.append(img_name)
                continue

            self.image_paths.append(self.filename_to_path[img_name])
            self.grades.append(grade)
            # Convert 0-4 to binary: 0-1 is non-referable (0), 2-4 is referable (1)
            self.labels.append(1 if grade >= 2 else 0)

        if missing:
            print(f"WARNING: {len(missing)} image(s) from {json_path} were not "
                  f"found in any of img_dirs and were skipped. "
                  f"First few: {missing[:5]}")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.grades[idx], dtype=torch.int64), torch.tensor(self.labels[idx], dtype=torch.float32)


class MILVideoDataset(Dataset):
    def __init__(self, json_path, num_frames=32, transform=None, random_segment_sample=False, search_dir_paths=None,
                 bag_transform=None, min_bag_size=None):
        """`transform` is applied per frame; optional `bag_transform` once to the stacked [T, C, H, W] bag.
        min_bag_size (train only): each bag keeps a random sorted subset of k ~ U[min_bag_size, num_frames] sampled frames."""
        self.min_bag_size = min_bag_size
        with open(json_path, 'r') as f:
            self.label_map = json.load(f)

        self.num_frames = num_frames
        self.transform = transform
        self.bag_transform = bag_transform
        self.random_segment_sample = random_segment_sample

        if search_dir_paths is None:
            search_dir_paths = ["data/ensemble_results/cleaned_videos"]

        all_video_paths = []
        for search_dir_path in search_dir_paths:
            search_dir = Path(search_dir_path)

            if search_dir.exists():
                all_video_paths.extend(search_dir.glob("*.mp4"))
                all_video_paths.extend(search_dir.glob("*.MOV"))

        self.data = []
        missing_count = 0
        corrupt_count = 0

        print(f"Mapping dataset from {json_path}...")

        for vid_name, grade in self.label_map.items():
            matched_path = None
            base_name = vid_name.replace('.mp4', '').replace('.MOV', '')
            pattern = re.compile(rf"{re.escape(base_name)}(?![a-zA-Z0-9])")

            for path in all_video_paths:
                if pattern.search(path.name):
                    matched_path = path
                    break

            if matched_path and matched_path.exists():
                # Check if video is valid (has a certain size)
                if matched_path.stat().st_size > 1000:
                    self.data.append({
                        'vid_name': vid_name,
                        'path': str(matched_path),
                        'label': 1.0 if grade >= 2 else 0.0,
                        'grade': grade
                    })
                else:
                    print(f"Corrupt: {matched_path}")
                    corrupt_count += 1
            else:
                print(f"Missing: {vid_name}")
                missing_count += 1

        print(f"Successfully mapped {len(self.data)} videos.")
        if missing_count > 0:
            print(f"Skipped {missing_count} videos (not found).")
        if corrupt_count > 0:
            print(f"Skipped {corrupt_count} videos (found but appear empty/corrupt).")

    @property
    def labels(self):
        return [int(item['label']) for item in self.data]

    @property
    def grades(self):
        return [int(item['grade']) for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        bag = self._load_video_frames(item)
        return bag, torch.tensor(item['label'], dtype=torch.float32), torch.tensor(item['grade'])

    def _sample_indices(self, item, total_frames):
        if not self.random_segment_sample:
            # 1. Uniform Sampling (Deterministic)
            target_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
        else:
            # 2. Temporal Segment Sampling (Stochastic)
            target_indices = []
            boundaries = np.linspace(0, total_frames, self.num_frames + 1, dtype=int)

            for i in range(self.num_frames):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]

                if start_idx >= end_idx:
                    # Failsafe for short videos (e.g., 20 frames total, but need 32)
                    # It will just duplicate the frame, acting similarly to uniform fallback
                    target_indices.append(min(start_idx, total_frames - 1))
                else:
                    # Pick a random frame within the temporal segment
                    sampled_idx = np.random.randint(start_idx, end_idx)
                    target_indices.append(sampled_idx)

            # Ensure they are sorted so the sequential OpenCV reader works properly
            target_indices.sort()
        return target_indices

    def _load_video_frames(self, item):
        path = item['path']
        frames = []
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Video {path} has 0 frames or unreadable metadata.")

        target_indices = self._sample_indices(item, total_frames)
        if self.min_bag_size is not None:
            k = np.random.randint(self.min_bag_size, self.num_frames + 1)
            keep = np.sort(np.random.choice(len(target_indices), k, replace=False))
            target_indices = [target_indices[j] for j in keep]
        bag_size = len(target_indices)
        target_counts = Counter(target_indices)

        # Only sampled frames are retrieved; grab() just advances past the others (~32 of ~870 frames are used).
        # Padding below therefore repeats the last *sampled* frame that could be read.
        last_valid_frame = None

        for i in range(total_frames):
            if i not in target_counts:
                if not cap.grab():
                    break
                continue
            ret, frame = cap.read()
            if not ret:
                break

            last_valid_frame = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            if self.transform:
                img = self.transform(img)

            for _ in range(target_counts[i]):
                frames.append(img)

            if len(frames) == bag_size:
                break

        cap.release()

        while len(frames) < bag_size and last_valid_frame is not None:
            frame_rgb = cv2.cvtColor(last_valid_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        if len(frames) != bag_size:
            raise RuntimeError(
                f"Critical failure: Expected {bag_size} frames from {path}, but ended up with {len(frames)}.")

        bag = torch.stack(frames)
        bag = self.bag_transform(bag) if self.bag_transform else bag
        return bag


class KaggleDRDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for grade_folder in sorted(os.listdir(split_dir)):
            grade_path = os.path.join(split_dir, grade_folder)
            if os.path.isdir(grade_path) and grade_folder.isdigit():
                grade = int(grade_folder)
                for img_name in os.listdir(grade_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(grade_path, img_name))
                        self.labels.append(grade) # Return 0, 1, 2, 3, or 4

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)
