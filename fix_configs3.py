with open("configs/Base-bagtricks.yml", "r") as f:
    lines = f.read().split('\n')

new_lines = []
in_solver = False
for line in lines:
    if line.startswith("SOLVER:"):
        in_solver = True
    elif line.startswith("TEST:") or line.startswith("OUTPUT_DIR:") or line.startswith("DATALOADER:") or line.startswith("INPUT:"):
        in_solver = False
        
    if in_solver and "EVAL_PERIOD" in line:
        continue # Skip EVAL_PERIOD if it's inside the SOLVER block
        
    new_lines.append(line)

with open("configs/Base-bagtricks.yml", "w") as f:
    f.write('\n'.join(new_lines))
