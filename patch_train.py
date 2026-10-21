with open("scripts/train.py", "r") as f:
    content = f.read()

replacement = """
    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)
        
        from fastreid.utils.checkpoint import Checkpointer
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)
        
        res = DefaultTrainer.test(cfg, model)
        return res
"""

import re
# Regex to match the buggy eval_only block
content = re.sub(
    r"    if args\.eval_only:.*?return \s*", 
    replacement.strip(), 
    content, 
    flags=re.DOTALL
)

with open("scripts/train.py", "w") as f:
    f.write(content)
