import os

for cfg in ['configs-w.json']:
    os.system("py main.py  %s  aplicar >> ./SAIDA.txt" % cfg)
    print("\n\nFim da execucao...")
