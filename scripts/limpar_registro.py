import os
from os import system, listdir, path
from sys import argv, path

url_base_ox = "../../Bases/Cache/Oxford/"

"""
    Limpa o registro da base de Oxford de uma palavra
    - Definicoes
    - Sinonimos
    - Frases de exemplos extraidos com XPath
"""

palavra = argv[1]

atributos = [url_base_ox + d for d in listdir(url_base_ox) if os.path.isdir(url_base_ox + d)]

for atr in atributos:
    d = d + "/%s/%s.json"%(atr, palavra)
    if os.path.exists(d):
        print("\tRemovendo: %s: %d"%(d, system('rm ' + d)))