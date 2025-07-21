# CCU CLEANUP SCRIPT FOR NARRAHUNT PHASE 2
# BEFORE EXECUTING, READ /home/computeruse/.anthropic/ALWAYSREAD.txt AND FOLLOW EVERY RULE IN IT EXACTLY.

import os
import shutil
import re

base_dir = "/home/computeruse/.anthropic/narrahunt_phase2/"
src_dir = os.path.join(base_dir, "src")
modules_dir = os.path.join(src_dir, "modules")
wayback_src = os.path.join(base_dir, "cache", "wayback")
wayback_dst = os.path.join(base_dir, "cache", "wayback_archive")
backup_dir = os.path.join(base_dir, "backup")
report_path = os.path.join(base_dir, "code_bloat_report.txt")

# --- 1. MAKE DIRECTORIES ---
os.makedirs(src_dir, exist_ok=True)
os.makedirs(modules_dir, exist_ok=True)
os.makedirs(wayback_dst, exist_ok=True)
os.makedirs(backup_dir, exist_ok=True)

# --- 2. DELETE __pycache__ AND .pyc FILES ---
for root, dirs, files in os.walk(base_dir):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d))
    for f in files:
        if f.endswith(".pyc"):
            os.remove(os.path.join(root, f))

# --- 3. MOVE PY FILES TO /src/ ---
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".py") and "backup" not in root and "src" not in root:
            src_path = os.path.join(root, f)
            dst_path = os.path.join(src_dir, f)
            shutil.move(src_path, dst_path)

# --- 4. MOVE BACKUPS ---
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".bak") or f.endswith(".backup") or f.endswith(".zip"):
            shutil.move(os.path.join(root, f), os.path.join(backup_dir, f))

# --- 5. MOVE WAYBACK JSONS ---
for f in os.listdir(wayback_src):
    if f.endswith(".json"):
        shutil.move(os.path.join(wayback_src, f), os.path.join(wayback_dst, f))

# --- 6. BLOAT REPORT ---
with open(report_path, "w") as report:
    for f in os.listdir(src_dir):
        if not f.endswith(".py"):
            continue
        path = os.path.join(src_dir, f)
        with open(path, "r", encoding="utf-8", errors="ignore") as code:
            content = code.read()
            lines = content.splitlines()
            num_lines = len(lines)
            num_imports = len(re.findall(r'^\s*(import|from) ', content, re.MULTILINE))
            num_defs = len(re.findall(r'^\s*def ', content, re.MULTILINE))
            num_classes = len(re.findall(r'^\s*class ', content, re.MULTILINE))

            bloat = (
                num_lines > 500 or
                num_imports > 20 or
                num_defs > 15
            )
            report.write(f"{f}: {num_lines} lines, {num_imports} imports, {num_defs} functions, {num_classes} classes")
            if bloat:
                report.write("  << BLOAT FLAGGED")
            report.write("\n")

print("✅ Cleanup complete. Bloat report saved to code_bloat_report.txt")