import re

with open('src/data/mevid_dataset.py', 'r') as f:
    content = f.read()

# Fix process_dir existing_files logic
new_logic1 = """        existing_files = set()
        if os.path.exists(img_dir):
            for root, _, files in os.walk(img_dir):
                for f in files:
                    existing_files.add(f)"""
content = re.sub(
    r"        existing_files = set\(\)\n        if os\.path\.exists\(img_dir\):\n            for root, _, files in os\.walk\(img_dir\):\n                for f in files:\n                    existing_files\.add\(os\.path\.relpath\(os\.path\.join\(root, f\), img_dir\)\)",
    new_logic1,
    content
)

# Fix process_test existing_files logic
new_logic2 = """        existing_files = set()
        if os.path.exists(self.test_dir):
            for root, _, files in os.walk(self.test_dir):
                for f in files:
                    existing_files.add(f)"""
content = re.sub(
    r"        existing_files = set\(\)\n        if os\.path\.exists\(self\.test_dir\):\n            for root, _, files in os\.walk\(self\.test_dir\):\n                for f in files:\n                    existing_files\.add\(os\.path\.relpath\(os\.path\.join\(root, f\), self\.test_dir\)\)",
    new_logic2,
    content
)

# Fix path creation
content = content.replace("img_path = os.path.join(img_dir, img_name)", "img_path = os.path.join(img_dir, f'{pid:04d}', img_name)")
content = content.replace("img_path = os.path.join(self.test_dir, img_name)", "img_path = os.path.join(self.test_dir, f'{pid:04d}', img_name)")

with open('src/data/mevid_dataset.py', 'w') as f:
    f.write(content)

