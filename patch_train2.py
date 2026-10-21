with open("scripts/train.py", "r") as f:
    content = f.read()

content = content.replace("return restrainer", "return res\n\n    trainer")

with open("scripts/train.py", "w") as f:
    f.write(content)
