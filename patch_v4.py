import re

with open('src/data/mevid_dataset.py', 'r') as f:
    content = f.read()

new_class = """@DATASET_REGISTRY.register()
class MEVID_Mini(ImageDataset):
    \"\"\"
    Bullet-proof dataloader for the shrunk MEVID_Sample_Dataset.
    OPTIMIZED: Uses memory sets to bypass slow filesystem checks!
    CORRECTED: Uses exact 1-based indexing logic from original MEVID.
    \"\"\"
    def __init__(self, root='datasets', **kwargs):
        # We handle Kaggle input paths automatically, or fallback to relative
        if os.path.exists("/kaggle/input/datasets/quangnguyen97/mevid-sample/MEVID_Sample_Dataset"):
            self.dataset_dir = "/kaggle/input/datasets/quangnguyen97/mevid-sample/MEVID_Sample_Dataset"
        else:
            self.dataset_dir = os.path.join(root, "MEVID_Sample_Dataset")
            
        self.train_dir = os.path.join(self.dataset_dir, 'bbox_train')
        self.test_dir = os.path.join(self.dataset_dir, 'bbox_test')
        self.anno_dir = os.path.join(self.dataset_dir, 'mevid-v1-annotation-data')
        
        self.pid_map = {}
        train_items = self._process_dir("train")
        query_items, gallery_items = self._process_test()
        
        super(MEVID_Mini, self).__init__(train_items, query_items, gallery_items, **kwargs)

    def _process_dir(self, split):
        track_info_file = os.path.join(self.anno_dir, f'track_{split}_info.txt')
        name_file = os.path.join(self.anno_dir, f'{split}_name.txt')
        img_dir = self.train_dir if split == "train" else self.test_dir
        
        with open(track_info_file, 'r') as f:
            track_lines = f.read().splitlines()
        with open(name_file, 'r') as f:
            name_lines = f.read().splitlines()
            
        existing_files = set()
        if os.path.exists(img_dir):
            for r, _, files in os.walk(img_dir):
                for f in files:
                    existing_files.add(f)
        
        items = []
        pid_counter = 0
        
        for line in track_lines:
            parts = line.split()
            if len(parts) != 5: continue
            
            start_idx, end_idx, pid, oid, cid = [int(float(x)) for x in parts]
            
            if pid not in self.pid_map:
                self.pid_map[pid] = pid_counter
                pid_counter += 1
            mapped_pid = self.pid_map[pid]
            
            for i in range(start_idx, end_idx + 1):
                idx = i - 1 if start_idx > 0 else i
                if idx < 0 or idx >= len(name_lines): continue
                
                img_name = name_lines[idx]
                
                if img_name in existing_files:
                    img_path = os.path.join(img_dir, f'{pid:04d}', img_name)
                    items.append((img_path, mapped_pid, cid))
                    
        return items
        
    def _process_test(self):
        track_info_file = os.path.join(self.anno_dir, 'track_test_info.txt')
        name_file = os.path.join(self.anno_dir, 'test_name.txt')
        query_idx_file = os.path.join(self.anno_dir, 'query_IDX.txt')
        
        with open(track_info_file, 'r') as f:
            track_lines = f.read().splitlines()
        with open(name_file, 'r') as f:
            name_lines = f.read().splitlines()
        with open(query_idx_file, 'r') as f:
            query_indices = set([int(float(x)) for x in f.read().splitlines()])
            
        existing_files = set()
        if os.path.exists(self.test_dir):
            for r, _, files in os.walk(self.test_dir):
                for f in files:
                    existing_files.add(f)
        
        query_items = []
        gallery_items = []
        
        for row_idx, line in enumerate(track_lines):
            parts = line.split()
            if len(parts) != 5: continue
            
            start_idx, end_idx, pid, oid, cid = [int(float(x)) for x in parts]
            
            for i in range(start_idx, end_idx + 1):
                idx = i - 1 if start_idx > 0 else i
                if idx < 0 or idx >= len(name_lines): continue
                
                img_name = name_lines[idx]
                
                if img_name in existing_files:
                    img_path = os.path.join(self.test_dir, f'{pid:04d}', img_name)
                    if row_idx in query_indices:
                        query_items.append((img_path, pid, cid))
                    else:
                        gallery_items.append((img_path, pid, cid))
                        
        return query_items, gallery_items
"""

# Replace anything from @DATASET_REGISTRY.register() down
content = re.sub(r"@DATASET_REGISTRY\.register\(\)\nclass MEVID_Mini\(ImageDataset\):.*", new_class, content, flags=re.DOTALL)
# Also handle MEVID_NoSkip if it was appended
content = re.sub(r"@DATASET_REGISTRY\.register\(\)\nclass MEVID_NoSkip\(ImageDataset\):.*", "", content, flags=re.DOTALL)

if "@DATASET_REGISTRY.register()\nclass MEVID_Mini(ImageDataset):" not in content:
    content += "\n" + new_class

with open('src/data/mevid_dataset.py', 'w') as f:
    f.write(content)

