import subprocess

with open('requirements.txt', 'w', encoding="utf-8") as f:
    subprocess.run(["pip", "freeze"], stdout=f)