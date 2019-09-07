#! coding: utf-8
import copy
import io
import operator
import os.path
import re
import traceback
import xml.etree.ElementTree as ET
from lxml import etree
from operator import itemgetter
from os import listdir, system
from os.path import isfile, join
from random import shuffle
from sys import argv

from nltk.corpus import wordnet as wordnet
from pywsd.lesk import cosine_lesk as cosine_lesk

from OxAPI import *
from Utilitarios import *


class VlddrSemEval(object):
    INSTANCE = None

    def __init__(self, cfgs):
        self.cfgs = cfgs

        cfgs_se = cfgs['semeval2007']
        dir_se = cfgs['caminho_bases']+'/'+cfgs['semeval2007']['dir_raiz']

        self.dir_resp_compet = dir_se+'/' + \
            cfgs_se['dir_resultados_concorrentes']
        self.gold_file_test = dir_se+'/'+cfgs_se['test']['gold_file']
        self.comando_scorer = dir_se+'/'+cfgs_se['comando_scorer']
        self.dir_tmp = cfgs['caminho_bases']+'/'+cfgs['dir_temporarios']

        self.todas_abordagens = dict()

    """
        Recebe um arquivo de submissao, na forma de pasta <folder> + diretorio do arquivo <dir_arq>
        junto à métrica utilizada <metrica>, e o conjunto de POS filtradas <pos_filtrada>
    """
    def aval_saida(self, tarefa, folder, dir_arq, pos_filtradas=None, metrica='Recall'):
        if '~' in tarefa:
            raise Exception("O diretorio passado utiliza de caminho relativo. Utilize de caminhos absolutos!")

        originais = self.aval_parts_orig(tarefa, [None])

        dir_completo = folder+'/'+dir_arq

        if pos_filtradas != None:
            os.system('cat %s | grep "\.%s " >> %s' %
                  (dir_completo, pos_filtradas, dir_completo+'.tmp'))
        else:
            os.system('cat %s >> %s' %
                  (dir_completo, dir_completo+'.tmp'))

        # def obter_score(self, dir_pasta_submissao, participante, pos_filtradas=None):
        abordagem_selecionada = self.obter_score(folder, dir_arq+'.tmp', pos_filtradas=pos_filtradas)

        originais[dir_completo] = abordagem_selecionada
        os.system("rm %s" % dir_completo+'.tmp')

        originais = [(reg['nome'], reg) for reg in sorted(
            originais.values(), key=itemgetter(metrica), reverse=True)]

        print("\n\n")
        for _reg_ in originais:
            nome, reg = _reg_
            print(nome)
            etmp = dict(reg)
            del etmp['nome']
            print(etmp)
            print('\n')

        return abordagem_selecionada

    # Gera o score das metricas das tarefas do SemEval para as abordagens originais da competicao
    def aval_parts_orig(self, tarefa, pos_filtradas=None, lexelts_filtrados=None):
        # Alias para funcao
        carregar_submissao = self.carregar_arquivo_submissao

        todos_participantes_originais = [p for p in Util.list_arqs(
            self.dir_resp_compet) if '.'+tarefa in p]
        todos_participantes_originais = [
            p for p in todos_participantes_originais if p.count('.') == 1]
        todos_participantes_originais = [
            p for p in todos_participantes_originais if not '-' in p]

        if pos_filtradas in [None, []]:
            pos_filtradas.sort()

        # Filtrando participante por POS
        todos_participantes_tmp = list(todos_participantes_originais)
        resultados_json = {}

        for participante in todos_participantes_tmp:
            dir_arq_original = self.dir_resp_compet+"/"+participante

            if pos_filtradas:
                pos_filtradas.sort()
                dir_arq_filtrado = dir_arq_original.replace(
                    tarefa, "")+"".join(pos_filtradas)+"."+tarefa
                regex = "\|".join(["\.%s " % pt for pt in pos_filtradas])
                comando = 'cat %s | grep "%s" > %s' % (
                    dir_arq_original, regex, dir_arq_filtrado)
                os.system(comando)
            else:
                dir_arq_filtrado = dir_arq_original

            participante_filtrado = dir_arq_filtrado.split("/")[-1]
            resultados_json[participante_filtrado] = self.obter_score(
                self.dir_resp_compet, participante_filtrado)
            system("rm " + dir_arq_filtrado)

        return resultados_json

    # Gera o score das metricas das tarefas do SemEval para as abordagens originais da competicao
    def aval_parts_orig_(self, tarefa, pos_filtradas=None, lexelts_filtrados=None):
        # Alias para funcao
        carregar_submissao = self.carregar_arquivo_submissao

        pos_semeval = self.cfgs['semeval2007']['todas_pos']
        pos_semeval.sort()

        todos_participantes = [p for p in Util.list_arqs(
            self.dir_resp_compet) if '.'+tarefa in p]
        todos_participantes = [p for p in todos_participantes if not '-' in p]

        if pos_filtradas in [None, []]:
            pos_filtradas = pos_semeval

        pos_filtradas.sort()

        # Filtrando participante por POS
        todos_participantes_tmp = list(todos_participantes)
        todos_participantes = []

        for participante in todos_participantes_tmp:
            cfgs_tarefa = self.cfgs['semeval2007']['tarefas']

            max_sugestoes = int(cfgs_tarefa['limites'][tarefa])
            separador = cfgs_tarefa['separadores'][tarefa]

            dir_arq_original = self.dir_resp_compet+"/"+participante
            predicao_filtr, linhas_arq_subm = carregar_submissao(self.cfgs,
                                                                 dir_arq_original, tarefa,
                                                                 pos_filtradas=pos_filtradas,
                                                                 lexelts_filtrados=lexelts_filtrados)

            # Convertendo o nome do arquivo original
            # para o nome do arquivo filtrado por POS-tags
            # SWAG.oot" => "SWAG-n.oot
            velho_sufixo = '.'+tarefa
            novo_sufixo = "-%s%s" % ("".join(pos_filtradas), velho_sufixo)
            dir_arq_filtrado = dir_arq_original.replace(
                velho_sufixo, novo_sufixo)

            # if lexelts_filtrados in [None, [ ]]:
            #    lexelts_filtrados = predicao_filtr.keys()
            # else:
            #    lexelts_filtrados = list(set(lexelts_filtrados)&set(predicao_filtr.keys()))

            # Convertendo a resposta do formato dicionario para list
            for lexelt in lexelts_filtrados:
                if lexelt in predicao_filtr:
                    resp_list = sorted(
                        predicao_filtr[lexelt].items(), key=lambda x: x[1], reverse=True)
                    resp_list = [e[0] for e in resp_list]
                    #if lexelt in casos_filtrados or True: pass
                    predicao_filtr[lexelt] = resp_list

            # Formatando arquivo filtrado
            self.formtr_submissao(
                dir_arq_filtrado, predicao_filtr, linhas_arq_subm, max_sugestoes, separador)
            # Retirando nome do participante, porém sem o diretorio que o contém
            todos_participantes.append(dir_arq_filtrado.split("/")[-1])

        resultados_json = {}

        for participante in todos_participantes:
            resultados_json[participante] = self.obter_score(
                self.dir_resp_compet, participante)

        # Se filtro de POS fora ativado, limpe os arquivos com submissao filtrada por POS-tags
        if pos_filtradas != pos_semeval:
            for participante in todos_participantes:
                pass

        return resultados_json

    # Checa se dada submissao nao sugeriu uma misera instancia contendo resposta!
    def submissao_invalida(self, dir_entrada, tarefa):
        submissao, linhas_arq_subm = self.carregar_arquivo_submissao(
            self.cfgs, dir_entrada, tarefa)
        resp = False
        for lexelt in submissao:
            if len(submissao[lexelt]) > 0:
                resp = True
        return resp == False

    # Executa o script Perl para gerar o score das abordagens
    def obter_score(self, dir_pasta_submissao, participante, pos_filtradas=None):
        tarefa = set(['oot', 'best']) & set(participante.split('.'))
        tarefa = list(tarefa)[0]

        arquivo_tmp = "%s/%s.tmp" % (self.dir_tmp, participante)

        if dir_pasta_submissao[-1] == "/":
            dir_pasta_submissao = dir_pasta_submissao[:-1]

        comando_scorer = self.comando_scorer
        dir_entrada = dir_pasta_submissao+'/'+participante
        dir_saida = dir_pasta_submissao+'/'+arquivo_tmp

        if self.submissao_invalida(dir_entrada, tarefa):
            raise Exception(
                "Esta submissao ('%s') nao sugeriu respostas!" % participante)

        if pos_filtradas == None:
            pos_filtradas = "anrv"

        if pos_filtradas == str: list(pos_filtradas)

        gold_tmp = self.gold_file_test + '.' + pos_filtradas + '.tmp'
        dir_entrada_tmp = dir_entrada + '.' + pos_filtradas + '.tmp'

        pattern = "\|".join(['\.'+p for p in pos_filtradas])

        saida = system('cat %s | grep "%s" >> %s' % (self.gold_file_test, pattern, gold_tmp))
        saida = system('cat %s | grep "%s" >> %s' % (dir_entrada, pattern, dir_entrada_tmp))

        args = (comando_scorer, dir_entrada_tmp,
                gold_tmp, tarefa, arquivo_tmp)

        # script, file, gold -t best
        comando = "perl %s %s %s -t %s > %s" % args

        try:
            system(comando)

            # Le a saida do formatoo <chave>:<valor> por linha
            obj = self.ler_registro(arquivo_tmp)
            
            obj['nome'] = participante

            try: system('rm ' + gold_tmp)
            except: pass
            try: system('rm ' + dir_entrada_tmp)
            except: pass
            try: system('rm ' + arquivo_tmp)
            except: pass

            return obj
        except Exception, e:
            print(e)
            return None

    def filtrar_participantes_tarefa(self, participantes, tarefa):
        return [p for p in participantes if tarefa in p]

    def ler_registro(self, path_arquivo):
        obj = dict()
        arq = open(str(path_arquivo), 'r')
        linhas = arq.readlines()

        for linha_tmp in linhas:
            try:
                l = str(linha_tmp).replace('\n', '')
                chave, valor = l.split(':')
                if chave in obj:
                    chave = "Mode" + chave
                obj[chave] = float(valor)
            except:
                pass

        arq.close()
        return obj

    # Formata a submissao para o padrao da competicao, que é lido pelo script Perl
    def formtr_submissao(self, dir_arquivo_saida, predicao, linhas_arq_subm, max_sugestoes, separador):
        if len(predicao) == 0:
            return

        arquivo_saida = open(dir_arquivo_saida, 'w')

        if not linhas_arq_subm in [[], None]:
            linhas_arq_subm = [" ".join(reg.split(" ")[:2])
                               for reg in linhas_arq_subm if "." in reg]
        else:
            linhas_arq_subm = predicao.keys()

        for lexelt in linhas_arq_subm:
            if lexelt in predicao:
                try:
                    respostas = predicao[lexelt][:max_sugestoes]
                    args = (lexelt, separador, ';'.join(respostas))
                    arquivo_saida.write("%s %s %s\n" % args)
                except Exception, e:
                    print(e)

    # Carregar o caso de entrada para gerar o ranking de sinonimos
    def carregar_caso_entrada(self, dir_arq_caso_entrada, padrao_se=False):
        todos_lexelts = dict()

        parser = etree.XMLParser(recover=True)
        arvore_xml = ET.parse(dir_arq_caso_entrada, parser)
        raiz = arvore_xml.getroot()

        for lex in raiz.getchildren():
            todos_lexelts[lex.values()[0]] = []
            for inst in lex.getchildren():
                codigo = str(inst.values()[0])
                context = inst.getchildren()[0]
                frase = "".join([e for e in context.itertext()]).strip()

                palavra = inst.getchildren()[0].getchildren()[0].text
                todos_lexelts[lex.values()[0]].append(
                    {'codigo': codigo, 'frase': frase, 'palavra': palavra})

        if padrao_se:
            todos_lexelts_tmp = dict(todos_lexelts)
            todos_lexelts = dict()

            for lexelt in todos_lexelts_tmp:
                for reg in todos_lexelts_tmp[lexelt]:
                    novo_lexelt = "%s %s" % (lexelt, str(reg['codigo']))
                    todos_lexelts[novo_lexelt] = reg

        return todos_lexelts

    def carregar_gabarito(self, dir_gold_file):
        arquivo_gold = open(dir_gold_file, 'r')
        todas_linhas = arquivo_gold.readlines()
        arquivo_gold.close()

        saida = dict()
        separador = " :: "

        todas_linhas = [linha for linha in todas_linhas if linha != "\n"]

        for linha in todas_linhas:
            resposta_linha = dict()
            try:
                chave, sugestoes = str(linha).replace(
                    '\n', '').split(separador)
                sugestoes = [s for s in sugestoes.split(';') if s]

                for sinonimo in sugestoes:
                    sinonimo_lista = str(sinonimo).split(' ')
                    votos = int(sinonimo_lista.pop())
                    sinonimo_final = ' '.join(sinonimo_lista)

                    resposta_linha[sinonimo_final] = votos
                saida[chave] = resposta_linha
            except:
                traceback.print_exc()

        return saida

    # Carregar arquivos Submissão SemEval 2007 (formatado com o padrao SemEval)
    def carregar_arquivo_submissao(self, cfgs, dir_arquivo,
                                   tarefa="oot", pos_filtradas=[], lexelts_filtrados=[]):

        arquivo_submetido = open(dir_arquivo, 'r')
        todas_linhas = arquivo_submetido.readlines()
        arquivo_submetido.close()

        # Predicao filtrada por POS-tags
        saida_filtrada_pos = dict()
        # Predicao filtrada pelos LEXELTS permitidos
        saida_filtrada_lexelt = dict()

        separador = cfgs['semeval2007']['tarefas']['separadores'][tarefa]
        separador = " " + separador + " "

        total_sugestoes = 0

        for linha in todas_linhas:
            resposta_linha = dict()
            try:
                chave, sugestoes = str(linha).replace(
                    '\n', '').split(separador)
                # "cry.v 893" => ["cry", "v", "893"]
                lema_tmp, pos_tmp, lema_id_tmp = re.split('[\.\s]', chave)

                if pos_tmp in pos_filtradas or pos_filtradas in [None, []]:
                    todos_candidatos = sugestoes.split(';')
                    indice = 0

                    for sinonimo in todos_candidatos:
                        if sinonimo != "":
                            sin_lista = sinonimo
                            votos = len(todos_candidatos)-indice
                            resposta_linha[sinonimo] = votos

                        indice += 1

                    saida_filtrada_pos[chave] = resposta_linha

                #raw_input("Total Lexelts filtrados: " + str(len(lexelts_filtrados)))

                # Filtro por Lexelt, None ou [ ] sao valores Default
                if chave in lexelts_filtrados or lexelts_filtrados == []:
                    saida_filtrada_lexelt[chave] = resposta_linha

            except:  # Se linha está sem predição
                pass

        saida = dict()

        for lexelt in set(saida_filtrada_pos) & set(saida_filtrada_lexelt):
            saida[lexelt] = saida_filtrada_pos[lexelt]

        return saida, todas_linhas

    """
        Carrega bases e gabarito ja no formato Python
        Retorna o par: <casos_testes_dict, gabarito_dict>
    """
    def carregar_bases(self, cfgs, tipo_base, pos_avaliadas=None):
        if pos_avaliadas in [None, []]:
            pos_avaliadas = cfgs['semeval2007']['todas_pos']

        casos_testes = gabarito = None
        validador = VlddrSemEval(cfgs)

        # Carrega a base Trial para fazer os testes
        dir_bases_se = cfgs['caminho_bases'] + \
            '/'+cfgs['semeval2007']['dir_raiz']
        dir_gabarito = dir_bases_se+'/' + \
            cfgs['semeval2007'][tipo_base]['gold_file']
        dir_entrada = dir_bases_se+'/'+cfgs['semeval2007'][tipo_base]['input']

        gabarito = validador.carregar_gabarito(dir_gabarito)
        casos_testes = validador.carregar_caso_entrada(dir_entrada)
        # gabarito_dict[lexelt cod] = [[palavra votos], [palavra votos], [palavra votos], ...]
        # casos_testes_dict[lexema cod] = [frase, palavra, pos]
        casos_testes_dict, gabarito_dict = {}, {}

        # Filtrando lexelts por chave
        chaves_casos_testes = []
        for lexelt_parcial in casos_testes:
            for reg in casos_testes[lexelt_parcial]:
                chaves_casos_testes.append(
                    "%s %s" % (lexelt_parcial, reg['codigo']))
        todos_lexelts = set(chaves_casos_testes) & set(gabarito)
        todos_lexelts = [l for l in todos_lexelts if re.split('[\.\s]', l)[
            1] in pos_avaliadas]

        for lexelt in todos_lexelts:
            lista = []
            for sugestao in gabarito[lexelt]:
                voto = gabarito[lexelt][sugestao]
                lista.append([sugestao, voto])
            gabarito_dict[lexelt] = lista

        for lexelt_iter in todos_lexelts:
            lexelt = lexelt_iter.split(" ")[0]  # 'scrap.n 104' => 'scrap.n'
            for registro in casos_testes[lexelt]:
                palavra, frase = registro['palavra'], registro['frase']
                pos = lexelt.split(".")[1]
                nova_chave = "%s %s" % (lexelt, registro['codigo'])
                casos_testes_dict[nova_chave] = [frase, palavra, pos]

        # Recomeçando as variaveis
        casos_testes, gabarito = [], []

        for lexelt in casos_testes_dict:
            if lexelt in gabarito_dict:
                casos_testes.append(casos_testes_dict[lexelt])
                gabarito.append(gabarito_dict[lexelt])

        return casos_testes_dict, gabarito_dict

    # Ordena o gabarito padrao anotado SemEval2007 por frequencia de votos
    def fltr_gabarito(self, gabarito):
        try:
            return sorted(gabarito, key=lambda x: x[1], reverse=True)
        except:
            return []

    @staticmethod
    def aplicar_gap(predicao, gabarito):
        import MelamudGAP

        resultados = {}

        for lexelt in predicao:
            ranklist = predicao[lexelt]
            gold_ranklist = [palavra for palavra, peso in gabarito[lexelt]]
            gold_weights = [peso for palavra, peso in gabarito[lexelt]]
            resultados[lexelt] = MelamudGAP.gap(ranklist, gold_ranklist, gold_weights)

        return resultados, Util.media(resultados.values())
