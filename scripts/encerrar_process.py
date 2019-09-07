from sys import argv
from os import system

diretorio = '/home/alvaro'

for k in range(2):
	try:
		nome = argv[1]
	except:
		nome = raw_input('Programa: ')

	arquivo_tmp = diretorio + '/encerramento.tmp'
	system('ps -C ' + nome + ' > ' + arquivo_tmp)

	arq = open(arquivo_tmp, 'r')

	linhas = list(arq.readlines())
	linhas.pop(0)

	for l in linhas:
		try:
			pid = str(l).split(' ')[0]
			print('kill -term ' + pid)
			system('kill -term ' + pid)
			system('kill -cont ' + pid)
		except Exception, e:
			print(e)
		try:
			pid = str(l).split(' ')[1]
			print('kill -term ' + pid)
			system('kill -term ' + pid)
			system('kill -cont ' + pid)
		except Exception, e:
			print(e)
		try:
			system('kill -cont ' + pid)
		except: pass
		try:
			system('kill -term ' + pid)
		except: pass

	arq.close()
	try:
		system('rm ' + arquivo_tmp)
	except: pass
	system('clear')

system('clear')
print('\nFim do script...')
