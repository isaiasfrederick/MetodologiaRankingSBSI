#! coding: utf-8
import gc
import itertools
import json
import os
import re
import signal
import statistics
import sys
import traceback 
from operator import itemgetter
from statistics import mean as media
from sys import argv

import nltk
from nltk.corpus import wordnet
from textblob import TextBlob
from Alvaro import Alvaro

# Experimentacao
from DesOx import DesOx
from ExtratorWikipedia import ExtratorWikipedia
from Indexador import Whoosh
from InterfaceBases import InterfaceBases
from OxAPI import BaseOx, CliOxAPI, ExtWeb
from RepresentacaoVetorial import RepVetorial
from SemEval2007 import VlddrSemEval

from BasesDesambiguacao import BasesDesambiguacao
from Utilitarios import Util
import string

# Fim pacotes da Experimentacao
wn = wordnet
sing = Util.singularize

resultados_discriminados = dict()

def exibir_bases(cfgs, fonte='wordnet', tipo='test', td_pos="anrv"):
    validador = VlddrSemEval(cfgs)

    if type(td_pos) == 'str': td_pos = list(td_pos)

    casos_testes_dict, gabarito_dict = carregar_bases(cfgs, tipo)
    palavras = set()

    alvaro = Alvaro.INSTANCE

    retorno = set()

    for lexelt in set(casos_testes_dict.keys()) & set(gabarito_dict.keys()):
        p = lexelt.split(".")[0]
        palavras.add(p)

    casos_validos = set(casos_testes_dict.keys()) & set(gabarito_dict.keys())

    for lexelt in [l for l in casos_validos if casos_testes_dict[l][2] in td_pos]:
        frase, palavra, pos = casos_testes_dict[lexelt]
        frase = Util.descontrair(frase).replace("  ", " ")
        palavra = lexelt.split(".")[0]

        if palavra in palavras:
            if pos in td_pos:
                nfrase = str(frase).replace(palavra, "(%s)" % palavra)
                nfrase = frase
                Util.verbose_ativado = True
                Util.print_formatado("%s" % frase)
                Util.print_formatado("Palavra: "+palavra)
                Util.print_formatado("POS: "+pos)
                Util.print_formatado("Resposta: " +
                                     str(validador.fltr_gabarito(gabarito_dict[lexelt])))
                cands = alvaro.selec_candidatos(palavra, pos, fontes=['wordnet', 'oxford'])
                # selec_candidatos(self, palavra, pos, fontes=['wordnet'], indice_definicao=-1):
                print("Candidatos: " + str(cands['uniao']))
                print("\n\n")

        retorno.add(str((palavra, pos)))

    return [eval(reg) for reg in retorno]

"""
    Recebe uma (frase, palavra target, pos)
    E calcula pmi para todas palavras da frase com a palavra target
    Rertorna uma lista ordenada daqueles de maior score
"""
def retornar_pmi_sentenca(cfgs, frase, palavra, pos):
    rem_vrbs_dlxc = cfgs['remover_delexical_verbs_sentenca']
    dlx_vrb = cfgs['delexical_verbs']

    validador = VlddrSemEval(cfgs)

    casos_testes_dict, gabarito_dict = carregar_bases(cfgs, "test")
    palavras = set()

    alvaro = Alvaro.INSTANCE

    todos_pmis = []

    # Convertendo values "anvr" -> ['a', 'n', 'r', 'v']
    filtros = cfgs['metodo_pmi']['filtro_sintatico']
    filtro_sintatico = dict([(k, list(v)) for (k, v) in filtros.items()])

    frase_tokenizada = nltk.pos_tag(nltk.word_tokenize(frase))

    if rem_vrbs_dlxc == True:
        frase_tokenizada = [(t, p) for (t, p) in frase_tokenizada if not t.lower() in dlx_vrb]

    for t, pos_iter in frase_tokenizada:
        pos_token = pos_iter[0].lower()
        pos_token = "a" if pos == "j" else pos_token # Se é adjetivo

        try:
            # Essa checagem do if é pra ver se nao é caracter de numero, que tem ZERO relevancia
            if set(t.lower())&set(string.ascii_lowercase):
                # Se o filtro sintatico admite a palavra
                if pos_token in filtro_sintatico[pos]:
                    if sing(t) != sing(palavra) and Util.stem(t) != Util.stem(palavra):
                        pos_tmp = "a" if pos == "j" else pos[0].lower() # Se é adjetivo

                        # Gerando chaves
                        k_list = Util.sort([(palavra, pos_tmp), (t, pos_token)], col=0, reverse=False)

                        if str(k_list) in Alvaro.PMI_TAG:
                            pmi = Alvaro.PMI_TAG[str(k_list)]
                            todos_pmis.append((t, pmi))
                        else:
                            try:
                                pmi = Alvaro.pmi(t, palavra)
                                todos_pmis.append((t, pmi))

                                Alvaro.PMI_TAG[str(k_list)] = pmi

                                reg = k_list + [pmi]
                            except:
                                print("\tExcecao no calculo da PMI para %s!"%str((t, palavra)))
        except Exception, e:
            print("\tExcecao no calculo PMI: " + str(e))

    # Formato [('palavra', pmi), ('palavra', pmi), ..., ('palavra', pmi)]
    return Util.sort(todos_pmis, col=1, reverse=True)

"""
    Assim como o metodo retornar_pmi_sentenca retorna as palavras
    da sentenca baseada na ocorrencia nas definicoes do dicionario.
    Palavras da sentenca muito ocorrentes do dicionario para determinada
    definicao recebe pontuacoes  maiores
"""
def retornar_representacao_definicoes(cfgs, representacao_frase, todos_cands, pos):
    validador = VlddrSemEval(cfgs)

    casos_testes_dict, gabarito_dict = carregar_bases(cfgs, "test")
    palavras = set()

    alvaro = Alvaro.INSTANCE

    todas_correlacoes = []

    # Convertendo values "anvr" -> ['a', 'n', 'r', 'v']
    filtros = cfgs['metodo_pmi']['filtro_sintatico']
    filtro_sintatico = dict([(k, list(v)) for (k, v) in filtros.items()])

    resultado = dict()

    try:
        for cand in todos_cands:
            vetores = Alvaro.vetor_exemplos(cand, pos, min_freq=2)
            vetores = vetores.items()

            for def_iter, vetor_frequencia_iter in vetores:
                if not def_iter in Alvaro.DEFINICOES_VETORIAIS:
                    palavras_exemplos_descritor = []                
                    lema, definicao = def_iter.split(":::")
                    
                    def_toke = nltk.pos_tag(nltk.word_tokenize(definicao.lower()))

                    # Convertendo pos-tags
                    def_toke_tmp = [(s, pt[0].lower() if pt[0].lower() != 'j' else 'a') for (s, pt) in def_toke]
                    def_toke = [s for (s, pt) in def_toke_tmp if pt == pos]

                    # Se a definicao nao contem sinonimos, extrair alguma outra palavra relacionada
                    if def_toke == []:
                        res = []

                        for (s, pt) in def_toke_tmp:
                            try: pmi = Alvaro.pmi(s, lema)
                            except: pmi = -1000 * 1000
                            res.append((s, pmi))
                        
                        res = Util.sort(res, col=1, reverse=True)

                        if res.__len__() > 0: def_toke = [res[0][0]]
                        else: def_toke = []

                    vetor_frequencia = list(vetor_frequencia_iter)
                    freq_min_exemplos = int(vetor_frequencia[0][1] * 0.6)
                    vetor_frequencia = [(p, f) for (p, f) in vetor_frequencia if f >= freq_min_exemplos and p != lema]

                    for palavra_frase, freq_palavra_frase in vetor_frequencia:
                        if palavra_frase + ':::' + pos in Alvaro.PALAVRAS_POS:
                            pass
                        else:
                            if wordnet.synsets(palavra_frase, pos):
                                palavras_exemplos_descritor.append(palavra_frase)

                    #print('\n')
                    #print((cand, def_iter))
                    #print('\n')
                    #print(palavras_exemplos_descritor)
                    #print('\n')
                    #print(def_toke)
                    #print('\n')

                    palavras_relacionadas = RepVetorial.obter_palavras_relacionadas
                    representacao_def = palavras_exemplos_descritor + def_toke
                    repvet_inst = RepVetorial.INSTANCE
                    palavras_derivadas = palavras_relacionadas(repvet_inst, positivos=representacao_def, pos=pos, topn=100)

                    Alvaro.DEFINICOES_VETORIAIS[def_iter] = palavras_derivadas

                else:
                    palavras_derivadas = Alvaro.DEFINICOES_VETORIAIS[def_iter]

                try:
                    #resultado[def_iter] = palavras_derivadas[0][0]
                    resultado[def_iter] = RepVetorial.cosseno(repvet_inst, palavras_derivadas[0][0], representacao_frase[0][0])
                except:
                    pass
                
            #for d in BaseOx.obter_definicoes(BaseOx.INSTANCE, cand, pos=pos):
            #    sins = BaseOx.obter_sins(BaseOx.INSTANCE, cand, d, pos=pos)
            #    sins = [s.lower() for s in sins]

            #    ex = BaseOx.obter_atributo(BaseOx.INSTANCE, cand, pos, d, 'exemplos')
            #    ex = [e.lower() for e in ex]
                
            #    vetor_definicao = []
            #    resultado[cand + ":::" + d] = vetor_definicao
    except Exception, e:
        pass
    
    # Formato [('palavra', cosseno), ('palavra', cosseno), ..., ('palavra', cosseno)]
    resultado = Util.sort(resultado.items(), col=1, reverse=True)    

    return resultado


def exibir_bases_pmi(cfgs, fonte='wordnet', tipo='test', td_pos=['a', 'n', 'r', 'v'], min_pmi=1.2):
    validador = VlddrSemEval(cfgs)

    casos_testes_dict, gabarito_dict = carregar_bases(cfgs, tipo)
    palavras = set()

    alvaro = Alvaro.INSTANCE

    retorno = []

    for lexelt in set(casos_testes_dict.keys()) & set(gabarito_dict.keys()):
        p = lexelt.split(".")[0]
        palavras.add(p)

    total_mudancas = 0

    casos_validos = set(casos_testes_dict.keys()) & set(gabarito_dict.keys())

    for lexelt in [l for l in casos_validos if casos_testes_dict[l][2] in td_pos]:
        frase, palavra, pos = casos_testes_dict[lexelt]
        frase = Util.descontrair(frase).replace("  ", " ")
        palavra = lexelt.split(".")[0]

        todos_pmis = []
        flag = False

        if palavra in palavras:
            if pos in td_pos:
                nfrase = str(frase).replace(palavra, "(%s)" % palavra)
                nfrase = frase

                print('\n')
                print((palavra, pos))
                print(nfrase)
                print(validador.fltr_gabarito(gabarito_dict[lexelt]))

                flag_mudanca = False

                for t, pos_iter in nltk.pos_tag(nltk.word_tokenize(nfrase)):
                    if t != palavra and pos_iter[0].lower() in ['n', 'v']:
                        try:
                            pmi = Alvaro.pmi(t, palavra)
                            todos_pmis.append((t, pmi))

                            print((t, pmi))

                            flag_mudanca = pmi > min_pmi

                        except KeyboardInterrupt, ke:
                            break
                        except Exception, e:
                            pass

                total_mudancas += int(flag_mudanca)

                print("\n\nTotal instancias validas: " + str(total_mudancas))

                try:
                    retorno.append((palavra, pos, nfrase, todos_pmis))
                except:
                    pass

    return retorno


def indexar_todos_candidatos(cfgs):
    casos_entrada, gabarito = carregar_bases(cfgs, "test")

    # Todos lexelts
    todos_lexelts = dict()

    for lexelt in casos_entrada:
        key = re.split('[\.\s]', lexelt)[0]
        if not key in todos_lexelts:
            todos_lexelts[key] = []
        todos_lexelts[key].append(lexelt)

    total_certos = 0
    total = 0

    # => 3446
    for key in todos_lexelts:
        lexelts_ineditos = todos_lexelts[key]

        for lexelt in lexelts_ineditos:
            frase, palavra, pos = casos_entrada[lexelt]
            palavra = palavra.lower()
            par_reg = Alvaro.selec_todos_candidatos(
                    Alvaro.INSTANCE, palavra, pos, fontes=['oxford', 'wordnet'])
            registro_generico, todos_candidatos = par_reg

            todos_candidatos = [p for p in todos_candidatos if  Util.e_mpalavra(p) == False]

            print("\n\n")
            print("\tTodos candidatos: " + str(todos_candidatos[:3]))
            print("\n\n")

            # Alvaro.indexar_exemplos(cand_iter, pos, fonte='oxford')
            for candidato in todos_candidatos:
                if set(candidato).intersection(set("-_ ")).__len__() == 0:
                    print("\tIndexando " + str((candidato, pos)))
                    if Alvaro.indexar_exemplos(candidato, pos, fonte='oxford'):
                        print("\tCerto!")
                        os.system('echo "%s %s" >> ~/palavras_indexadas.log'%(candidato, pos))
                        total_certos += 1
                    else:
                        print("\tErro!")
                    print("\tFim.")
                    print('\n')

        total += 1

    print("\n__ Fim da indexaca dos documentos __\n")
    return (total_certos, total)


def carregar_bases(cfgs, tipo_base, pos_avaliadas=None):
    return VlddrSemEval.carregar_bases(VlddrSemEval.INSTANCE,
                                       cfgs, tipo_base, pos_avaliadas=pos_avaliadas)

# Este metodo usa a abordagem do Alvaro sobre as bases do SemEval
# Ela constroi uma relacao (score) entre diferentes definicoes, possivelmente sinonimos
#   criterio = frequencia OU alvaro OU embeddings
def predizer_sins(
    cfgs,
    criterio='frequencia',
    usar_gabarito=False,
    lexelts_filtrados=None,
    fontes_def='oxford', tipo=None,
    max_ex=-1, usr_ex=False,
    pos_avaliadas=None,
    carregar_candidatos_disco=False
):

    verbose_geral = cfgs['verbose']['geral']
    separador = cfgs['separador']
    med_sim = cfgs['medida_similaridade']
    saida_contigencial = cfgs['saida_contig']['habilitar']

    if fontes_def != 'oxford':
        raise Exception("Esta fonte de definicoes nao é contem exemplos...")

    if pos_avaliadas in [None, []]:
        pos_avaliadas = cfgs['semeval2007']['todas_pos']
    if type(pos_avaliadas) != list:
        raise Exception("\n\nAs POS avaliadas devem ser uma list!\n\n")

    # Construtor com carregador de modelo
    dir_modelo = "%s/%s" % (cfgs['caminho_bases'], cfgs['modelos']['default'])
    rep_vet = RepVetorial.INSTANCE

    base_ox = BaseOx.INSTANCE
    alvaro = Alvaro.INSTANCE

    rep_vetorial = RepVetorial.INSTANCE
    des_ox = DesOx(cfgs, base_ox, rep_vetorial=rep_vetorial)

    if max_ex == -1:
        max_ex = sys.maxint

    todos_cands = dict()

    # Resultado de saida <lexelt : lista>
    predicao_final = dict()

    # Fonte para selecionar as definicoes e fonte para selecionar os candidatos
    # fontes_def, fontes_cands = raw_input("Digite a fonte para definicoes: "), ['oxford', 'wordnet']
    fontes_def, fontes_cands = fontes_def, cfgs['fontes_cands']
    casos_testes_dict, gabarito_dict = carregar_bases(
        cfgs, tipo, pos_avaliadas=pos_avaliadas)

    if lexelts_filtrados in [None, []]:
        casos_testes_dict_tmp = list(casos_testes_dict.keys())
    else:
        casos_testes_dict_tmp = set(
            casos_testes_dict.keys()) & set(lexelts_filtrados)
        casos_testes_dict_tmp = list(casos_testes_dict_tmp)

    vldr_se = VlddrSemEval(cfgs)
    todos_lexelts = list(set(casos_testes_dict_tmp)
                         & set(gabarito_dict.keys()))
    indices_lexelts = [i for i in range(len(todos_lexelts))]

    palavras_invalidas = []

    cache_rel_sinonimia = dict()
    cache_seletor_candidatos = dict()
    cache_resultado_desambiguador = dict()

    usar_frases_wikipedia = cfgs['wikipedia']['usar_frases']

    if Util.CONFIGS['ngram']['usar_seletor'] == False:
        Alvaro.NGRAMS_SIGNALMEDIA = dict()
        Alvaro.NGRAMS_LEIPZIG = dict()
        Alvaro.NGRAMS_COCA = dict()
    else:

        if "coca" in Util.CONFIGS['ngram']['fontes']:
            print("\n\t- Carregando n-grams COCA Corpus!")
            Alvaro.NGRAMS_COCA = Util.abrir_json(cfgs['ngram']['dir_coca_ngrams'])
            print("\t- n-grams COCA Corpus carregado!")

        if "signalmedia" in Util.CONFIGS['ngram']['fontes']:
            # Abrindo ngrams SignalMedia
            arq_ngrams_tmp = dict()

            with open(Util.CONFIGS['ngram']['signalmedia_5grams'], 'r') as todas_linhas:
                print("\n\t- Carregando n-grams SignalMedia Corpus!")

                for linha_ngram in todas_linhas:
                    try:
                        tokens = linha_ngram.split(":")
                        freq_ngram = int(tokens[-1])
                        ngram = str(
                            ":".join(tokens[:-1])).strip('\t').replace("\t", " ")
                        Alvaro.NGRAMS_SIGNALMEDIA[ngram] = freq_ngram
                    except:
                        pass

                print("\n\t- n-grams SignalMedia Corpus carregado!")

            print("\n\t- Derivando n-grams SignalMedia Corpus!")

            ngrams_signalmedia_derivados = Alvaro.derivar_ngrams_string(Alvaro.NGRAMS_SIGNALMEDIA,
                                                                        cfgs['ngram']['min'],
                                                                        cfgs['ngram']['max'])
            for ng in ngrams_signalmedia_derivados:
                Alvaro.NGRAMS_SIGNALMEDIA[ng] = ngrams_signalmedia_derivados[ng]

            print("\t- n-grams SignalMedia Corpus derivado!")

            ngrams_signalmedia_derivados = None

    dir_palavras_indexadas = Util.CONFIGS['corpora']['dir_palavras_indexadas_exemplos']
    try:
        Alvaro.PALAVRAS_EXEMPLOS_INDEXADOS = set(
            Util.abrir_json(dir_palavras_indexadas, criarsenaoexiste=True))
    except:
        Alvaro.PALAVRAS_EXEMPLOS_INDEXADOS = set()

    total_acertos = 0

    qtde_sugestoes_oot = Util.CONFIGS['params_exps']['qtde_sugestoes_oot'][0]
    qtde_sugestoes_oot = 200

    contador = 0

    if Alvaro.PMI_TAG in [None, dict()]:
        dir_contadores = Util.CONFIGS['metodo_pmi']['dir_contadores_tag']
        if Util.arq_existe(None, dir_contadores):
            Alvaro.PMI_TAG = Util.abrir_json(dir_contadores, criarsenaoexiste=False)            
            print("\n\tAbrindo PMIs: %d registros!\n"%len(Alvaro.PMI_TAG))

    if Alvaro.PALAVRAS_MONOSSEMICAS in [None, dict()]:
        dir_monossemicas = Util.CONFIGS['oxford']['dir_palavras_monossemicas']        
        if Util.arq_existe(None, dir_monossemicas):
            try:
                Alvaro.PALAVRAS_MONOSSEMICAS = set(Util.abrir_json(dir_monossemicas, criarsenaoexiste=False))           
                print("\n\tAbrindo palavras monossemicas: %d registros!\n\n"%len(Alvaro.PALAVRAS_MONOSSEMICAS))
            except Exception, e:
                print("\t" + str(e))

    for cont in list(indices_lexelts):
        lexelt = todos_lexelts[cont]

        if lexelt in Util.CONFIGS['lexelts_bloqueados']:
            indices_lexelts.remove(cont)

    # Se os embeddings retornar NADA, some.
    total_instancias_ambiguas = 0

    # Utiliza o BEST-PMI para rankear o Best
    desambiguar = Util.CONFIGS['desambiguar_embeddings']
    # Completa predicao com PMI aplicado ao dicionario
    completar_predicao_dicionarios = Util.CONFIGS['completar_predicao_dicionarios']

    for cont in indices_lexelts:
        lexelt = todos_lexelts[cont]
        frase, palavra, pos = casos_testes_dict[lexelt]
        frase = Util.descontrair(frase).replace("  ", " ")
        palavra = lexelt.split(".")[0]

        exemplos_ponderado = []

        if not palavra in palavras_invalidas:
            chave_seletor_candidatos = str((palavra, pos))
            interseccao_casos = list(
                set(casos_testes_dict_tmp) & set(gabarito_dict.keys()))
            gab_ordenado = Util.sort(gabarito_dict[lexelt], 1, reverse=True)

            print("\n\n\n")
            print("\t@@@ Processando a entrada " + str(lexelt))
            print("\t%d / %d\t" % (cont+1, len(interseccao_casos)))
            #print("\t" + str(pcent) + "%")
            print("\t*** FRASE: %s\n" % str((frase, palavra, pos)))
            print("\tGabarito: %s" % str(gab_ordenado))

            if usar_gabarito == True:
                if cfgs['gabarito']['tipo'] == 'default':
                    cands = [e[0] for e in gabarito_dict[lexelt]
                             if not Util.e_mpalavra(e[0])]
                elif cfgs['gabarito']['tipo'] == 'gap':
                    cands = []
                    for l in casos_testes_dict:
                        try:
                            frasetmp, palavratmp, postmp = casos_testes_dict[l]
                            if palavratmp == palavra:
                                cands += [p1 for p1, p2 in gabarito_dict[l]]
                        except:
                            pass

                    cands = list(set(cands))
                    cands = [c for c in cands if not Util.e_mpalavra(c)]

                top_unigrams = [(p, BaseOx.freq_modelo(
                    BaseOx.INSTANCE, p)) for p in cands]
                top_ngrams = list(top_unigrams)
                cands_brutos = list(top_unigrams)

            else:

                tripla_cands = alvaro.selec_candidatos_filtr_bkp(palavra, pos, fontes=fontes_cands)

                cands = list(set(tripla_cands['uniao']))
                cands_best = tripla_cands['best']
                cands_oot = tripla_cands['oot']
                
                confs_locais = Util.CONFIGS['alvaro']['politica_selecao']

                todos_cands[lexelt] = tripla_cands

                # Conjunto-universo de candidatos
                # selec_todos_candidatos(self, palavra, pos, fontes=['wordnet'], salvar=True)
                alv_inst = Alvaro.INSTANCE
                f_tmp = ['oxford', 'wordnet']

                cands_universo = Alvaro.selec_todos_candidatos(alv_inst, palavra, pos, fontes=f_tmp, salvar=True)
                cands_universo = cands_universo[0]['uniao_candidatos']
                cands_universo = [p for p in cands_universo if Util.e_mpalavra(p) == False]

                # Removendo palavra
                if palavra in cands:
                    cands.remove(palavra)

                cands_brutos = list(cands)

                # Removendo palavras nulas de frequencia nula
                top_unigrams = [(p, BaseOx.freq_modelo(BaseOx.INSTANCE, p)) for p in cands]
                top_unigrams = [r for r in sorted(
                    top_unigrams, key=lambda x: x[1], reverse=True) if r[1] > 0]

                if cfgs['ngram']['usar_seletor'] == True:
                    dir_ngramas = "./Bases/Candidatos/%s.json"%lexelt
                    ngramas_discriminados = Util.abrir_json(dir_ngramas, criarsenaoexiste=False)

                    if ngramas_discriminados in [None, { }]:
                        print("\t[%s] Gerando ngramas!\n"%lexelt)
                        ngramas_discriminados = Alvaro.selec_ngrams(
                            Alvaro.INSTANCE, palavra, frase, [p for (p, s) in top_unigrams])
                    else:
                        print("\t N-gramas de [%s] foram achados no disco!\n"%lexelt)

                    # =============================================================

                    dif_cands = list(set(cands_universo) - set(ngramas_discriminados.keys()))
                    dif_cands = [(p, BaseOx.freq_modelo(BaseOx.INSTANCE, p)) for p in dif_cands]
                    dif_cands = [r[0] for r in sorted(dif_cands, key=lambda x: x[1], reverse=True) if r[1] > 0]

                    # ngramas nao compreendidos no que veio do disco
                    ngramas_discriminados_dif = Alvaro.selec_ngrams(
                        Alvaro.INSTANCE, palavra, frase, dif_cands)

                    for c_iter in ngramas_discriminados_dif:
                        ngramas_discriminados[c_iter] = ngramas_discriminados_dif[c_iter]

                    if ngramas_discriminados.__len__() > 0:
                        d = "./Bases/Candidatos/%s.json"%lexelt
                        Util.salvar_json(d, ngramas_discriminados)
                        print("\n\tSalvando ngramas discriminados para %s."%lexelt)

                    # =============================================================

                    top_ngrams = dict()
                    for c_iter in ngramas_discriminados:
                        top_ngrams[c_iter] = 0
                        for base_ngrams in Util.CONFIGS['ngram']['fontes']:
                            if c_iter in ngramas_discriminados:
                                for n in ngramas_discriminados[c_iter][base_ngrams]:
                                    freq = ngramas_discriminados[c_iter][base_ngrams][n]
                                    top_ngrams[c_iter] += Alvaro.pont_colocacao(int(freq), int(n))

                    top_ngrams = top_ngrams.items()
                    top_ngrams = [(p, s) for (p, s) in top_ngrams if p in cands]
                                   
                    # Removendo frequencia == 0
                    top_ngrams = [registros for registros in Util.sort(
                        top_ngrams, 1, reverse=True) if registros[1] > 0.00]
                    top_ngrams = [(p, s) for (p, s) in top_ngrams if wordnet.synsets(p, pos)]

                else:
                    top_ngrams = []

                cands = [p for p in cands if Util.e_mpalavra(p) == False]

                # TOP-10 predicoes inicializados com NGRAMS
                cands = [
                    p for (p, s) in top_ngrams[:cfgs['ngram']['max_cands_filtro']]]
                # Inicializando com TOP unigramas
                cands += [p for (p, s) in top_unigrams if not p in cands]

            if criterio in ["assembled", "assemble"]:
                try:

                    if Util.CONFIGS['ngram']['usar_seletor'] == True:
                        cands_tmp = list(cands)
                        rem_delex_verbs = Util.CONFIGS['remover_delexical_verbs']

                        if rem_delex_verbs == True:
                            cands_tmp = [p for p in cands_tmp if not p.lower() in rem_delex_verbs]

                        cands_tmp = [p for p in cands_tmp if len(p) > 1][:qtde_sugestoes_oot]

                        if not 'n' in resultados_discriminados:
                            resultados_discriminados['n'] = dict()
                            
                        resultados_discriminados['n'][lexelt] = cands_tmp


                    if desambiguar == True:
                        palavras_correlatas = retornar_pmi_sentenca(
                            Util.CONFIGS, frase, palavra, pos)

                        palavras_ex, freq_palavras = BasesDesambiguacao.consultar_palavras_exemplos(palavra, pos)
                        intersec = set(palavras_ex).intersection(nltk.word_tokenize(frase.lower()))

                        if palavra.capitalize() in intersec: intersec.remove(palavra.capitalize())  
                        if palavra.lower() in intersec: intersec.remove(palavra.lower())
                        if palavra in intersec: intersec.remove(palavra)

                        stop_words = BasesDesambiguacao.SUPER_LISTA_STOPWORDS
                        intersec = [p for p in intersec if not p.lower() in stop_words]

                        # Pega a palavra mais bem relacionada e vê se,
                        # a partir dela, deriva-se a Best Prediction
                        if palavras_correlatas != []:
                            similares = []
                            indice_prox = 0

                            try:                                
                                melhor_correlata, importancia_pmi = palavras_correlatas[0]
                                pior_correlata, importancia_pmi = palavras_correlatas[-1]

                                inst = RepVetorial.INSTANCE
                                coordenadas = [palavra, melhor_correlata]
                                
                                similares = RepVetorial.obter_palavras_relacionadas(
                                inst, positivos=coordenadas, negativos=[pior_correlata], pos=pos, topn=100)

                                indice = 1

                                while not similares:
                                    melhor_correlata, importancia_pmi = palavras_correlatas[indice]
                                    inst = RepVetorial.INSTANCE
                                    coordenadas = [palavra, melhor_correlata]
                                    
                                    similares = RepVetorial.obter_palavras_relacionadas(
                                    inst, positivos=coordenadas, negativos=[pior_correlata], pos=pos, topn=100)
                                    indice += 1

                                    if similares:
                                        break
                                    else:
                                        break

                            except:
                                similares = []

                            if completar_predicao_dicionarios == True:
                                if similares == None:
                                    similares = [] 

                                # GAP: 0.122872128668
                                documentos = Whoosh.consultar_documentos(list(coordenadas), operador="AND", limite=200, dir_indexes=Whoosh.DIR_INDEXES_EXEMPLOS)

                                docs_filtrados = []
                                for doc in documentos:
                                    for cand_iter in cands:
                                        if cand_iter + "-" + pos + "." in doc['path']:
                                            doc_palavra, doc_pos, doc_definicao = cand_iter, pos, doc['title'].split(':::')[1]
                                            docs_filtrados.append((doc_palavra, doc_pos, doc_definicao))
                                            print("\t%s - %s"%(doc['title'], doc['path']))

                                if not docs_filtrados:
                                    print("\n\tNao ha docs filtrados...")
                                else:
                                    docs_usados = set()                                    
                                    for doc_palavra, doc_pos, doc_definicao in docs_filtrados:
                                        sins = BaseOx.obter_sins(BaseOx.INSTANCE, doc_palavra, doc_definicao, doc_pos)
                                        for s in sins:
                                            if not s in similares and s in cands:
                                                similares.append(s)
                                        
                            similares = [p for p, s in similares if p in cands]

                            for c in similares:
                                cands.remove(c)

                            print('\n')
                            print("\tSimilares %s: %s" % (str((palavra, melhor_correlata)), str(similares)))

                            if similares == []:
                                total_instancias_ambiguas += 1

                            cands = similares + cands
                                
                            print("\n\tSaida: " + str(cands) + "\n\n")

                            if Util.CONFIGS['ngram']['usar_seletor'] == False:
                                cands_tmp = list(cands)
                                rem_delex_verbs = Util.CONFIGS['remover_delexical_verbs']

                                if rem_delex_verbs == True:
                                    cands_tmp = [p for p in cands_tmp if not p.lower() in rem_delex_verbs]

                                cands_tmp = [p for p in cands_tmp if len(p) > 1][:qtde_sugestoes_oot]

                                if not 'e' in resultados_discriminados:
                                    resultados_discriminados['e'] = dict()

                                resultados_discriminados['e'][lexelt] = cands_tmp

                        else:
                            print("\tNao ha forte correlacao para a instancia!")

                except Exception, e:
                    pass

                try:
                    inst = Alvaro.INSTANCE

                    if Util.CONFIGS['remover_delexical_verbs'] == True:
                        cands = [p for p in cands if not p.lower() in Util.CONFIGS['delexical_verbs']]

                    predicao_final[lexelt] = [
                        p for p in cands if len(p) > 1][:qtde_sugestoes_oot]

                    if True == desambiguar and cfgs['ngram']['usar_seletor'] == True:
                        if not 'ne' in resultados_discriminados:                                
                            resultados_discriminados['ne'] = dict()

                        resultados_discriminados['ne'][lexelt] = predicao_final[lexelt]

                except Exception, e:
                    predicao_final[lexelt] = []

                if Alvaro.PONDERACAO_DEFINICOES != None:
                    Alvaro.salvar_base_ponderacao_definicoes()

            # Metodo de Baseline
            elif criterio == "word_move_distance":
                conj_predicao = []
                resultado_wmd = dict()

                for cand_iter in cands:
                    nova_frase = frase.replace(palavra, cand_iter)
                    pont = rep_vetorial.word_move_distance(frase, nova_frase)
                    if not pont in resultado_wmd:
                        resultado_wmd[pont] = []
                    resultado_wmd[pont].append(cand_iter)

                for pont in sorted(resultado_wmd.keys(), reverse=False):
                    conj_predicao += resultado_wmd[pont]

                predicao_final[lexelt] = [
                    p for p in conj_predicao if len(p) > 1][:qtde_sugestoes_oot]

            elif criterio == "similares_emb":
                rep_vet = RepVetorial.INSTANCE
                similares = RepVetorial.obter_palavras_relacionadas(rep_vet, positivos=[palavra], pos=None, topn=100)

                similares = [p for p, s in similares if Util.e_mpalavra(p) == False]

                predicao_final[lexelt] = [
                    p for p in similares if len(p) > 1][:qtde_sugestoes_oot]

                # Se o modelo nao é capaz de predizer o melhor substituto,
                # Entao, compute o erro
                if predicao_final[lexelt] == []:
                    predicao_final[lexelt] = palavra
            
            if cont + 1 < len(indices_lexelts):
                prox_lexelt = todos_lexelts[cont+1]
                prox_palavra = prox_lexelt.split(".")[0]
                if palavra != prox_palavra:
                    BaseOx.objs_unificados = None
                    BaseOx.objs_unificados = dict()

            if Alvaro.PONDERACAO_DEFINICOES != None:
                Util.salvar_json(
                    "./Bases/ponderacao_definicoes.json", Alvaro.PONDERACAO_DEFINICOES)

        if exemplos_ponderado:
            Util.salvar_json("./Bases/%s.frase.json" %
                             lexelt, exemplos_ponderado)

        gc.collect()
        print("\n\n\n\n\n\n\n")

    print("\nTOTAL INSTANCIAS AMBIGUAS: " + str(total_instancias_ambiguas) + "\n\n")

    if Alvaro.PMI_TAG != None:
        if Alvaro.PMI_TAG.__len__() > 0:
            dir_contadores = Util.CONFIGS['metodo_pmi']['dir_contadores_tag']
            res = Util.salvar_json(dir_contadores, Alvaro.PMI_TAG)
            print("\n\tSalvando PMIs: " + str(res) + "\n\n")

    if Alvaro.PALAVRAS_MONOSSEMICAS != None:
        if Alvaro.PALAVRAS_MONOSSEMICAS.__len__() > 0:
            dir_palavras_monossemicas = Util.CONFIGS['oxford']['dir_palavras_monossemicas']
            res = Util.salvar_json(dir_palavras_monossemicas, list(Alvaro.PALAVRAS_MONOSSEMICAS))
            print("\n\tSalvando palavras monossemicas: " + str(res) + "\n\n")

    # Para o Garbage Collector
    cache_rel_sinonimia = None
    cache_seletor_candidatos = None
    cache_resultado_desambiguador = None

    # Remover predicoes falhas
    predicao_final_copia = dict(predicao_final)

    for reg in predicao_final:
        if predicao_final[reg] in [[], ""]:
            del predicao_final_copia[reg]

    predicao_final = dict(predicao_final_copia)

    # Predicao, caso de entrada, gabarito
    return predicao_final, casos_testes_dict, gabarito_dict, None, todos_cands, None


def aplicar_abordagens(cfgs):
    from SemEval2007 import VlddrSemEval

    Util.verbose_ativado = False
    Util.cls()

    params_exps = cfgs['params_exps']

    Util.CONFIGS = cfgs

    # InterfaceBases.setup(cfgs)
    rep_vet = RepVetorial.INSTANCE

    ExtratorWikipedia.INSTANCE = ExtratorWikipedia(cfgs)

    caminho_bases = cfgs['caminho_bases']

    vldr_se = VlddrSemEval.INSTANCE  # Validador SemEval 2007
    rep_vet = RepVetorial.INSTANCE

    diretorio_saida_json = cfgs['dir_saida_json']
    dir_saida_abrdgm = cfgs['saida_experimentos']

    # GERANDO PARAMETROS
    todos_criterios = params_exps['todos_criterios']
    qtde_exemplos = params_exps['qtde_exemplos']
    qtde_sugestoes_best = params_exps['qtde_sugestoes_best']
    qtde_sugestoes_oot = params_exps['qtde_sugestoes_oot']
    todas_fontes_def = params_exps['todas_fontes_def']
    tipos_base = params_exps['tipos_base']

    flags_usar_gabarito = [v for v in params_exps['usar_gabarito']]
    flags_usar_exemplos = [v for v in params_exps['usar_exemplos']]

    pos_avaliadas = params_exps['pos_avaliadas'] if params_exps['pos_avaliadas'] else [
        None]
    max_indice_pos = cfgs['params_exps']['max_entradas_pos']

    parametrizacao = [
        todos_criterios,
        qtde_exemplos,
        todas_fontes_def,
        tipos_base,
        flags_usar_gabarito,
        flags_usar_exemplos,
        pos_avaliadas
    ]

    # Gerando todas combinacoes de parametros p/ reproduzir experimentos
    parametrizacao = list(itertools.product(*parametrizacao))

    # Filtra os casos de entrada por lexelt
    cfgs_se = cfgs["semeval2007"]
    caminho_raiz_semeval = "%s/%s" % (caminho_bases, cfgs_se["dir_raiz"])
    path_base_teste = "%s/%s" % (caminho_raiz_semeval,
                                 cfgs_se["test"]["input"])

    casos_filtrados_tmp = vldr_se.carregar_caso_entrada(
        path_base_teste, padrao_se=True)
    contadores = dict([(pos, 0) for pos in cfgs_se['todas_pos']])
    lexelts_filtrados = []

    for lexelt in casos_filtrados_tmp:
        pos_tmp = re.split('[\s\.]', lexelt)[1]  # 'war.n 0' => ('war', 'n', 0)
        if contadores[pos_tmp] < max_indice_pos:
            lexelts_filtrados.append(lexelt)
            contadores[pos_tmp] += 1
    # Fim do filtro de casos de entrada por lexelt

    lf = lexelts_filtrados
    res_best = vldr_se.aval_parts_orig(
        'best', pos_filtradas=pos_avaliadas[0], lexelts_filtrados=lf).values()
    res_oot = vldr_se.aval_parts_orig(
        'oot', pos_filtradas=pos_avaliadas[0], lexelts_filtrados=lf).values()

    cfg_ngram_ox = cfgs['ngram']['dir_exemplos']

    ch_ngram_plain = cfg_ngram_ox['oxford_plain']
    ch_ngram_tagueados = cfg_ngram_ox['oxford_tagueados']

    Alvaro.abrir_contadores_pmi()

    for parametros in parametrizacao:
        crit, max_ex, fontes_def, tipo, usr_gab, usr_ex, pos_avaliadas = parametros

        if type(pos_avaliadas) != list:
            raise Exception(
                "\n\nAs POS avaliadas devem ser expressas no formato de list!\n\n")

        predicao = set()
        casos = set()
        gabarito = set()

        try:
            pred_saida = predizer_sins(
                cfgs,
                lexelts_filtrados=lexelts_filtrados,
                usar_gabarito=usr_gab,
                criterio=crit,
                tipo=tipo,
                fontes_def=fontes_def,
                pos_avaliadas=pos_avaliadas,
                carregar_candidatos_disco=False
            )

            predicao, casos, gabarito, candidatos, todos_cands, predicao_final_discriminada = pred_saida

            todas_instancias, media = VlddrSemEval.aplicar_gap(predicao, gabarito)
            print("\n\nGAP: %s\n\n" % str(media))

        except Exception, e:
            import traceback
            print("\n")
            traceback.print_exc()
            print("\n%s\n" % str(e))

            Alvaro.salvar_contadores_pmi()

            if Alvaro.PONDERACAO_DEFINICOES != None:
                print("\nSalvando relacao entre sinonimos pontuadas via-WMD!\n")
                Alvaro.salvar_base_ponderacao_definicoes()

        Alvaro.salvar_contadores_pmi()

        # Gerando score
        for cont in qtde_sugestoes_oot:
            nome_abrdgm = cfgs['padrao_nome_submissao']  # '%d-%s-%s ... %s'
            reg = (crit, usr_gab, tipo, max_ex, usr_ex, fontes_def, 'oot')
            nome_abrdgm = nome_abrdgm % reg

            predicao_oot = dict()

            for lexelt in predicao:
                try:
                    sugestoes = [p for p in predicao[lexelt]
                                 if p in todos_cands[lexelt]['oot']]
                    predicao_oot[lexelt] = sugestoes
                except:
                    pass

            vldr_se.formtr_submissao(
                dir_saida_abrdgm+"/"+nome_abrdgm, predicao_oot, None, cont, ":::")

            print("\nSaida da sua abordagem: " +
                  dir_saida_abrdgm+"/"+nome_abrdgm+"\n")

            if Util.arq_existe(dir_saida_abrdgm, nome_abrdgm):
                try:
                    res_oot.append(vldr_se.obter_score(
                        dir_saida_abrdgm, nome_abrdgm))
                except Exception, reg:
                    print("\n@@@ Erro na geracao do score da abordagem '%s'" %
                          nome_abrdgm+"\n")

        nome_abrdgm = cfgs['padrao_nome_submissao']
        reg = (crit, usr_gab, tipo, max_ex, usr_ex, fontes_def, 'best')
        nome_abrdgm = nome_abrdgm % reg

        predicao_best = dict()

        for lexelt in predicao:
            if lexelt in predicao_best and lexelt in predicao_oot:
                if predicao_best[lexelt] != predicao_oot[lexelt]:
                    print("\n")
                    print((predicao_best[lexelt], predicao_oot[lexelt]))
                    print("\nSao diferentes!\n")

        for lexelt in predicao:
            try:
                sugestoes = [p for p in predicao[lexelt]
                             if p in todos_cands[lexelt]['best']]
                predicao_best[lexelt] = sugestoes
            except:
                pass

        print("Saida da sua abordagem: "+dir_saida_abrdgm+"/"+nome_abrdgm)
        vldr_se.formtr_submissao(
            dir_saida_abrdgm+"/"+nome_abrdgm, predicao_best, None, 1, "::")

        if Util.arq_existe(dir_saida_abrdgm, nome_abrdgm):
            try:
                res_best.append(vldr_se.obter_score(
                    dir_saida_abrdgm, nome_abrdgm))
            except:
                print("\n@@@ Erro na geracao do score da abordagem '%s'" %
                      nome_abrdgm)

    if Alvaro.PALAVRAS_EXEMPLOS_INDEXADOS != None:
        dir_saida = Util.CONFIGS['corpora']['dir_palavras_indexadas_exemplos']
        Util.salvar_json(dir_saida, list(Alvaro.PALAVRAS_EXEMPLOS_INDEXADOS))
        Alvaro.PALAVRAS_EXEMPLOS_INDEXADOS = None

    Alvaro.salvar_base_ponderacao_definicoes()

    res_tarefas = {'best': res_best, 'oot': res_oot}

    # Exibindo todas abordagens
    for nome_tarefa in res_tarefas:
        res_tarefa = res_tarefas[nome_tarefa]
        chave = ""

        cfgs_se = cfgs['semeval2007']

        exibir_resultados_automaticamente = cfgs_se['exibir_resultados_automaticamente']
        metrica_automatica = cfgs_se['metrica_automatica']

        if not exibir_resultados_automaticamente and not metrica_automatica in res_tarefa[0].keys():
            while not chave in res_tarefa[0].keys():
                msg = "\nEscolha a chave pra ordenar a saida "

                chave = raw_input(msg+nome_tarefa.upper()+": " +
                                str(res_tarefa[0].keys()) + ": ")
        else:
            chave = cfgs_se['metrica_automatica']

        res_tarefa = sorted(res_tarefa, key=itemgetter(chave), reverse=True)

        Util.salvar_json("%s/%s.%s" % (diretorio_saida_json,
                                       "000-SAIDA-" + nome_tarefa.upper(), nome_tarefa), res_tarefa)
        print("\n" + chave.upper() + "\t-----------------------")

        for reg in res_tarefa:
            print(reg['nome'])
            etmp = dict(reg)
            del etmp['nome']
            Util.exibir_json(etmp, bloquear=False)
            print('\n')

        print("\n")


if __name__ == '__main__':
    if len(argv) < 3:
        print('\nParametrizacao errada!\nTente py ./main <dir_config_file>\n\n')
        exit(0)
    elif argv[2] == 'ajuda':
        exit(0)

    cfgs = Util.carregar_cfgs(argv[1])
    InterfaceBases.setup(cfgs, dir_keys="./keys.json", funcao=None if argv[2] != 'aval_saida' else argv[2])

    inst = RepVetorial.INSTANCE

    BasesDesambiguacao.SUPER_LISTA_STOPWORDS

    stop_words = Util.mesclar_listas(Util.abrir_json(
        Util.CONFIGS['super_lista_stopwords']).values())
    BasesDesambiguacao.SUPER_LISTA_STOPWORDS = stop_words

    if argv[2] == 'indexar_todos_candidatos':
        print("\tIndexando exemplos para todos candidatos.")
        indexar_todos_candidatos(cfgs)
        print("\tFim da indexacao dos exemplos!")

    elif argv[2] == 'conjugar':
        raw_input(Util.conjugar('plays'))
        exit(0)

    elif argv[2] == 'aval_saida':
        folder = "/".join(argv[3].split('/')[:-1])
        dir_arq = argv[3].split('/')[-1]
             
        try: pos_filtradas = argv[4]
        except: pos_filtradas = None

        instance = VlddrSemEval.INSTANCE
        res = VlddrSemEval.aval_saida(
            instance, argv[3], folder, dir_arq, pos_filtradas=pos_filtradas, metrica='ModeRecall')

        print('\n')
        print(res)
        raw_input('\n\n<enter>')
        print('\n\n')

    elif argv[2] == 'aval_saida_files':            
        try: pos_filtradas = argv[4]
        except: pos_filtradas = None

        res = [ ]

        for arq_iter in open(argv[3], 'r').readlines():
            arq = arq_iter.replace('\n', '')

            folder = "/".join(arq.split('/')[:-1])
            dir_arq = arq.split('/')[-1]

            instance = VlddrSemEval.INSTANCE
            res.append(VlddrSemEval.aval_saida(
                instance, arq, folder, dir_arq, pos_filtradas=pos_filtradas, metrica='ModeRecall'))

        Util.exibir_json(res, bloquear=False)

    elif argv[2] == 'relacionadas':
        from functools import reduce

        palavras = argv[3:]
        inst = RepVetorial.INSTANCE
        relacionadas = RepVetorial.obter_palavras_relacionadas(
            inst, positivos=palavras, topn=100, remover_score=True)

        clusters = []
        for p in palavras:
            clusters.append(RepVetorial.obter_palavras_relacionadas(
                inst, positivos=[p], topn=100, remover_score=True))

        intersec = list(reduce(lambda i, j: i & j, (set(x) for x in clusters)))

        print("Soma: " + str(relacionadas))
        print("\n\n")
        print("Interseccao: " + str(intersec))

    elif argv[2] == 'sinonimos':
        palavra = argv[3]
        pos = argv[4]
        min_pmi = float(argv[5])

        try:
            derivar_palavras = eval(argv[6])
        except:
            derivar_palavras = False

        todas_definicoes = [(palavra, d) for d in BaseOx.obter_definicoes(
            BaseOx.INSTANCE, palavra, pos)]
        todos_sinonimos = Alvaro.obter_sinonimos_filtrados(
            todas_definicoes, pos, remover_sinonimos_replicados=True)

        for lexelt in todos_sinonimos.items():
            palavra_definicao, sinonimos = eval(lexelt[0]), lexelt[1]
            palavra, definicao = palavra_definicao

            print((palavra, definicao, sinonimos))

            inst_repvet = RepVetorial.INSTANCE

            for t in nltk.word_tokenize(definicao.lower()):
                try:
                    if not derivar_palavras:
                        soma = []
                        interseccao = []
                    else:
                        soma = RepVetorial.obter_palavras_relacionadas(
                            inst_repvet, positivos=[palavra, t], pos=pos, topn=200)
                        interseccao = Alvaro.interseccao_palavras(
                            t, palavra, pos)
                        soma = [p for p, s in soma]

                    pmi = Alvaro.pmi(palavra, t)
                    if pmi >= min_pmi:
                        print("\n")
                        print("\t%s: %f" % ((palavra, t), pmi))
                        print("\tSoma: " + str(soma))
                        print("\tInterseccao: " + str(interseccao))
                except:
                    pass

            print("\n\tSINONIMOS:")
            for t in sinonimos:
                try:
                    if not derivar_palavras:
                        soma = []
                        interseccao = []
                    else:
                        soma = RepVetorial.obter_palavras_relacionadas(
                            inst_repvet, positivos=[palavra, t], pos=pos, topn=200)
                        interseccao = Alvaro.interseccao_palavras(
                            t, palavra, pos)
                        soma = [p for p, s in soma]

                    pmi = Alvaro.pmi(palavra, t)
                    if pmi >= min_pmi:
                        print("\n")
                        print("\t%s: %f" % ((palavra, t), pmi))
                        print("\tSoma: " + str(soma))
                        print("\tInterseccao: " + str(interseccao))
                except:
                    pass

            print("\n")

    elif argv[2] == 'caso_unico':
        pass

    elif argv[2] == 'indexar':
        lista_arqs = "./Bases/Corpora/SignalMedia/arquivos.txt"
        textos_repetidos = "./Bases/Corpora/SignalMedia/textos_repetidos.txt"

        Whoosh.iniciar_indexacao_signalmedia(lista_arqs, textos_repetidos)

    elif argv[2] == 'ver_base':
        tipo = argv[3]

        if len(argv) == 5:
            td_pos = list(argv[4])
            exibir_bases(cfgs, tipo=tipo, td_pos=td_pos)
        else:
            b = exibir_bases(cfgs, tipo=tipo)

    elif argv[2] == 'ver_base_pmi':
        tipo = argv[3]

        if raw_input("Informar o <min_pmi>? s/n: ").lower() == 's':
            min_pmi = float(raw_input("Min PMI: "))
        else:
            min_pmi = 1.2

        if len(argv) == 5:
            td_pos = list(argv[4])
            base = exibir_bases_pmi(
                cfgs, tipo=tipo, td_pos=td_pos, min_pmi=min_pmi)
            Util.exibir_json(base, bloquear=True)
        else:
            base = exibir_bases_pmi(cfgs, tipo=tipo, min_pmi=min_pmi)
            Util.exibir_json(base, bloquear=True)

    elif argv[2] == 'pmi':
        palavra1 = argv[3]
        palavra2 = argv[4]

        pmi = Alvaro.pmi(palavra1, palavra2)
        print("\nPMI para o par (%s, %s): %f" % (palavra1, palavra2, pmi))

    elif argv[2] == 'ver_definicoes':
        palavra, pos = argv[3], argv[4]

        print("\t" + str((palavra, pos)).upper())
        for defiter in BaseOx.obter_definicoes(BaseOx.INSTANCE, palavra, pos):
            sins = BaseOx.obter_sins(BaseOx.INSTANCE, palavra, defiter, pos)
            print("\t\t" + defiter + ' - ' + str(sins))

    elif argv[2] == 'ver_definicoes2':
        todos_cands = dict()
        mgold = dict()

        path, path_gold, fonte = argv[3], argv[4], argv[5]

        total_gold = 0
        total_corretos = 0

        for linha_iter in open(path).readlines():
            reg = str(linha_iter).replace('\n', '').split('.')
            palavra = reg[0]
            tpos = reg[1:]

            todos_cands[palavra] = [ ]

            for pos in tpos:
                if fonte == 'oxford':
                    for defiter in BaseOx.obter_definicoes(BaseOx.INSTANCE, palavra, pos):
                        sins = BaseOx.obter_sins(BaseOx.INSTANCE, palavra, defiter, pos)
                        todos_cands[palavra] += sins
                elif fonte == 'wordnet':
                    for s in wn.synsets(palavra, pos):
                        h_set = []
                        for h in s.hypernyms():
                            h_set.append(h)
                            for h2 in h.hypernyms():
                                h_set.append(h2)
                        todos_cands[palavra] += s.lemma_names()
                        for h in h_set:
                            todos_cands[palavra] += h.lemma_names()

        for linha_iter in open(path_gold).readlines():
            key, values = str(linha_iter).replace('\n', '').split(' :: ')
            values = [' '.join(v.split(' ')[:-1]) for v in values.split(';') if v != '']
            mgold[key] = values
            gabarito = todos_cands[key.split(".")[0]]
            total_gold += len(values)
            intersec = set(gabarito).intersection(set(values))
            total_corretos += len(intersec)
            print((key, values))

        print("\n\n")
        print("\nVersao: " + str(wn.get_version()))
        print(float(total_corretos)/float(total_gold))
        print(total_corretos)
        raw_input(total_gold)
        print("\n\n")

    elif argv[2] == 'frases_wikipedia':
        palavra = argv[3]

        print("\t" + palavra.upper())
        for lema, descricao, url, f in ExtratorWikipedia.obter_frases_exemplo(palavra):
            print("\t" + str((descricao, f)))

    elif argv[2] == 'aplicar':
        gc.enable()

        try:
            aplicar_abordagens(cfgs)
        except KeyboardInterrupt:
            if Alvaro.FREQUENCIA_PMI != None:
                print("\tSalvando contadores PMI...")
                Alvaro.salvar_contadores_pmi()
                print("\tContadores PMI salvos...")

        if Alvaro.FREQUENCIA_PMI != None:
            Alvaro.salvar_contadores_pmi()

        if resultados_discriminados.__len__() > 0:
            nome_obj = "-".join(Util.CONFIGS['fontes_cands'])
            Util.salvar_json("./Bases/%s.json"%nome_obj, resultados_discriminados)



    print('\n\n\n\n\tFim do __main__\n\n\n\n')
