from pathlib import Path

from src.preprocess_data import SplitResult, timestep_from_name, train_val_split


def test_timestep_from_name_parses_integer():
+    assert timestep_from_name(Path("dump.0.LAMMPS")) == 0
+    assert timestep_from_name(Path("dump.12345.LAMMPS")) == 12345
+
+
+def test_timestep_from_name_falls_back_to_zero_on_bad_name():
+    assert timestep_from_name(Path("dump.notanumber.LAMMPS")) == 0
+    assert timestep_from_name(Path("weird_name.LAMMPS")) == 0
+
+
+def test_train_val_split_uses_contiguous_80_20_split():
+    files = [Path(f"dump.{i}.LAMMPS") for i in range(10)]
+
+    split = train_val_split(files, train_fraction=0.8)
+
+    # 80% of 10 is 8 (floor), remaining 2 for validation
+    assert len(split.train_files) == 8
+    assert len(split.val_files) == 2
+
+    # Ensure contiguity and non-overlap
+    assert split.train_files == files[:8]
+    assert split.val_files == files[8:]
+
+    # Ensure the union covers all files
+    assert split.train_files + split.val_files == files
