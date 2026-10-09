import re

with open('src/data/mevid_dataset.py', 'r') as f:
    content = f.read()

# Replace process_dir existing_files logic
new_logic1 = """        existing_files = set()
        if os.path.exists(img_dir):
            for root, _, files in os.walk(img_dir):
                for f in files:
                    existing_files.add(os.path.relpath(os.path.join(root, f), img_dir))"""
content = re.sub(
    r"        existing_files = set\(os\.listdir\(img_dir\)\) if os\.path\.exists\(img_dir\) else set\(\)",
    new_logic1,
    content
)

# Replace process_test existing_files logic
new_logic2 = """        existing_files = set()
        if os.path.exists(self.test_dir):
            for root, _, files in os.walk(self.test_dir):
                for f in files:
                    existing_files.add(os.path.relpath(os.path.join(root, f), self.test_dir))"""
content = re.sub(
    r"        existing_files = set\(os\.listdir\(self\.test_dir\)\) if os\.path\.exists\(self\.test_dir\) else set\(\)",
    new_logic2,
    content
)

with open('src/data/mevid_dataset.py', 'w') as f:
    f.write(content)

