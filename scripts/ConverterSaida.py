import json
import sys

obj_abertura = json.loads(open(sys.argv[1], 'r').read())
separador = '@@@@'

tarefa = raw_input('\n\nDigite a subtarefa: ')

if tarefa.lower() == 'best':
    separador = '::'
    max_elementos = 1
elif tarefa.lower() == 'oot':
    max_elementos = 10
    separador = ':::'

for chave in obj_abertura:
	nome_arquivo = raw_input("Digite o nome do arquivo para '%s': "%chave)

	obj_saida = open(nome_arquivo, 'w')

	for lexelt in obj_abertura[chave]:
		obj_saida.write(lexelt + ' ' + separador + ' ' + ';'.join(obj_abertura[chave][lexelt][:max_elementos]) + '\n')

	obj_saida.close()

print("\n\n\nFim do Script...\n\n\n")