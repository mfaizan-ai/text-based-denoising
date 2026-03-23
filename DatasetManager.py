import os
import json
import random
from pathlib import Path
from PIL import Image
from collections import defaultdict


class DatasetManager:
    """
    Dataset Manager with centralized metadata storage.
    All JSON metadata and TXT reports are saved in a dedicated folder.
    """

    # Static mapping to ensure consistent task IDs across different runs
    TASK_MAPPING = {
        "blur": 0,
        "raindrop": 1,
        "rainstreak": 2,
        "rainstreak_raindrop": 3,
        "reflection": 4
    }

    def __init__(self, root_path="dataset_full", output_folder="dataset_metadata"):
        """
        :param root_path: The root directory containing the raw image folders.
        :param output_folder: The directory where JSON and TXT metadata will be stored.
        """
        self.root = Path(os.getcwd()) / root_path
        self.all_samples = []
        self.report_lines = []
        self.category_stats = {}

        # Initialize the output directory for metadata
        self.output_dir = Path(os.getcwd()) / output_folder
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message):
        """Prints a message to the console and stores it for the TXT report."""
        print(message)
        self.report_lines.append(message)

    @staticmethod
    def _is_image_file(filename):
        """Checks if a file is a supported image format."""
        return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    def _get_image_info(self, file_path):
        """Retrieves image dimensions and format using a lightweight header scan."""
        try:
            with Image.open(file_path) as img:
                return img.size, img.format
        except Exception:
            return None, None

    def scan_database(self):
        """
        Phase 1: Scan the directory structure and perform a data audit.
        Matches 'blended' and 'clean' pairs and collects resolution metadata.
        """
        if not self.root.exists():
            print(f"❌ Error: Root directory '{self.root}' not found.")
            return

        self.all_samples = []
        self.category_stats = {}

        # Iterate through Task folders (e.g., blur, raindrop)
        for task_dir in sorted(self.root.iterdir()):
            if not task_dir.is_dir(): continue

            # Iterate through Sub-types (e.g., real, syn)
            for sub_dir in task_dir.iterdir():
                if not sub_dir.is_dir(): continue

                blended_path = sub_dir / "blended"
                clean_path = sub_dir / "clean"

                # Ensure both required subfolders exist
                if not (blended_path.exists() and clean_path.exists()):
                    continue

                cat_name = f"{task_dir.name.upper()} ({sub_dir.name.upper()})"
                cat_samples = []
                total_w, total_h, formats = 0, 0, set()

                # Match files in 'blended' with those in 'clean'
                for fname in sorted(os.listdir(blended_path)):
                    if not self._is_image_file(fname): continue

                    in_f, tg_f = blended_path / fname, clean_path / fname
                    if tg_f.exists():
                        size, fmt = self._get_image_info(in_f)
                        if size:
                            w, h = size
                            total_w += w
                            total_h += h
                            formats.add(fmt)

                            # Store relative paths and metadata for the JSON
                            cat_samples.append({
                                "input": str(in_f.relative_to(self.root.parent)),
                                "target": str(tg_f.relative_to(self.root.parent)),
                                "task_name": task_dir.name,
                                "task_id": self.TASK_MAPPING.get(task_dir.name, 99),
                                "data_type": sub_dir.name,
                                "width": w,
                                "height": h
                            })

                # If the category has valid pairs, save its statistics
                if cat_samples:
                    count = len(cat_samples)
                    self.category_stats[cat_name] = {
                        "count": count,
                        "avg_res": (total_w // count, total_h // count),
                        "formats": list(formats),
                        "samples": cat_samples
                    }
                    self.all_samples.extend(cat_samples)

        self._print_enhanced_summary()
        # Save the audit report to the metadata folder
        self.save_report_to_txt(filename=self.output_dir / "dataset_info.txt")

    def _print_enhanced_summary(self):
        """Generates the [TASK SUMMARY] log for the console and report file."""
        self.report_lines = []
        self.log("\n" + "=" * 80)
        self.log("[TASK SUMMARY]")
        for cat_label, info in self.category_stats.items():
            res = f"{info['avg_res'][0]}x{info['avg_res'][1]}"
            fmts = "/".join(info['formats'])
            self.log(f"- {cat_label}: {info['count']} pairs | Average: {res} | Formats: {fmts}")

        self.log(f"\n[TOTAL]: {len(self.all_samples)} pairs across {len(self.category_stats)} tasks.")
        self.log("=" * 80)

    def save_report_to_txt(self, filename):
        """Writes the collected log lines into a text file."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report_lines))
        print(f"📝 Audit report saved to: {filename}")

    def split_and_save_json(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        """
        Phase 2: Performs stratified sampling to split the dataset while maintaining
        the task distribution across Train, Val, and Test sets.
        """
        if not self.category_stats:
            self.log("❌ Error: No data found. Please run scan_database() first.")
            return

        random.seed(seed)
        train_set, val_set, test_set = [], [], []
        self.log("\n[SPLITTING DATASET]")

        # Split each sub-category independently to ensure a balanced distribution
        for cat_name, info in self.category_stats.items():
            samples = list(info['samples'])
            random.shuffle(samples)

            n = len(samples)
            tr_idx = int(n * train_ratio)
            val_idx = int(n * (train_ratio + val_ratio))

            train_set.extend(samples[:tr_idx])
            val_set.extend(samples[tr_idx:val_idx])
            test_set.extend(samples[val_idx:])

            self.log(f"  -> {cat_name:<25}: Tr={tr_idx}, Val={val_idx - tr_idx}, Te={n - val_idx}")

        # Shuffle the global sets so the model doesn't see tasks in sequence
        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)

        # Save splits to JSON files in the output directory
        splits = {"train": train_set, "val": val_set, "test": test_set}
        for name, data in splits.items():
            file_path = self.output_dir / f"{name}_metadata.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            self.log(f"💾 Saved metadata to: {file_path}")

        self.log("=" * 80)



if __name__ == "__main__":
    # All metadata will be generated and saved into the 'dataset_metadata' folder
    manager = DatasetManager(root_path="dataset_full", output_folder="dataset_metadata")

    # Run the audit and split process
    manager.scan_database()
    manager.split_and_save_json(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)