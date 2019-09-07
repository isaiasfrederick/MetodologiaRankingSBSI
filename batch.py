import json
import os

todas_iteracoes = []

pasta_batch = "Variando-K"
os.system('mkdir "%s"' % pasta_batch)

dir_config_template = 'configs.json'

for k in range(2, 11):
	arq = dict(json.loads(open(dir_config_template, 'r').read()))
	arq['peso_ngram'] = k

	arq_saida = open("configs-%s.json" % k, 'w')
	arq_saida.write(json.dumps(arq, indent=4))
	arq_saida.close()

	reg = {
		'configs': 'configs-' + str(k) + '.json',
		'saida_best': 'saida_best_' + str(k),
		'saida_oot': 'saida_oot_' + str(k)
	}

	todas_iteracoes.append(reg)

	# Execucao do programa
	comando = 'py ./main.py %s aplicar >> SAIDA.txt' % reg['configs']

	print("\t" + comando)
	print(os.system(comando))

	dir_saida = "C=assembled_G=False_T=test_MAXEX=30_USAREX=False_FONTESDEF=oxford"

	print("Copiando arquivos para a pasta certa!")
	os.system('cp ./Experimentos/' + dir_saida + '.best ./Variando-K/best-%s.best'%k)
	os.system('cp ./Experimentos/' + dir_saida + '.oot ./Variando-K/best-%s.oot'%k)

	# Remove configs.json temporario
	print("Removendo configs-%s.json!"%k)
	os.system('rm configs-%s.json'%k)
