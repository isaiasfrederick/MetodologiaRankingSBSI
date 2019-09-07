# coding: utf-8
import itertools
import os
import re
import string
import sys
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from operator import itemgetter
from sys import argv

import lxml.html
import nltk
import numpy
from lxml import etree
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy import spatial

# from pattern.en import conjugate
from Alvaro import Alvaro
from Indexador import Whoosh
from InterfaceBases import InterfaceBases
from OxAPI import *
from RepresentacaoVetorial import RepVetorial as RV
from Utilitarios import Util

conjugate = None

mesclar_listas = Util.mesclar_listas
abrir_json = Util.abrir_json

class RepresentacaoConceito(object):
    STOP_WORDS = {}
    HIPEHIPO = {}

    def __init__(self, palavra, pos):
        conceito_nuclear = []
        conceitos_marginais = []
        sinonimos_exclusivos = []
        palavras_exemplo = []

    @staticmethod
    def filtrar_palavra(palavra):
        STOP_WORDS = RepresentacaoConceito.STOP_WORDS
        stop_tokens = STOP_WORDS + Util.CONFIGS['delexical_verbs'] + list(string.punctuation)

        return not palavra.lower() in stop_tokens and not palavra in stop_tokens

    @staticmethod
    def criar_repr_wordnet(palavra, pos):
        pos = pos.lower()
        delexical_verbs = Util.CONFIGS['delexical_verbs']
        altura = 6 # Altura taxonomia

        lista_exc = [palavra] + delexical_verbs
        chave = palavra + '@@@' + pos

        conceitos_nucleares = []

        sim_mono_dup = set()
        similares_dup = set()

        similares_mono = dict()
        similares = dict()

        uniao_sins = []
        sinonimos_exc = dict()

        resultado = dict()

        hiperonimos = dict()
        hiponimos = dict()

        if pos in 'nv':
            if not chave in RepresentacaoConceito.HIPEHIPO:
                RepresentacaoConceito.HIPEHIPO[chave] = list()

                for syn in wn.synsets(palavra, pos):
                    sinonimos_exc[syn.name()] = []
                    hiperonimos[syn.name()] = []
                    hiponimos[syn.name()] = []

                    similares[syn.name()] = []
                    similares_mono[syn.name()] = []

                    for hipe_path in syn.hypernym_paths():
                        for hipe in hipe_path[::-1][1:][:altura]:
                            for l in hipe.lemma_names()[:4]:
                                if Util.e_mpalavra(l) == False and not l in lista_exc:
                                    RepresentacaoConceito.HIPEHIPO[chave].append(l)
                                    hiperonimos[syn.name()].append(l)

                    for hipo in syn.hyponyms():
                        for l in hipo.lemma_names()[:2]:
                            if Util.e_mpalavra(l) == False and not l in lista_exc:
                                RepresentacaoConceito.HIPEHIPO[chave].append(l)
                                hiponimos[syn.name()].append(l)

                    for l in syn.lemma_names():
                        if Util.e_mpalavra(l) == False and not l in lista_exc:
                            sinonimos_exc[syn.name()].append(l)
                            uniao_sins.append(l)

                uniao_sins = dict([(k, v) for (k, v) in Counter(uniao_sins).items() if v == 1])

                for syn_name in sinonimos_exc:
                    sinonimos_exc[syn_name] = list(set(sinonimos_exc[syn_name])&set(uniao_sins.keys()))

        elif pos in 'ars':
            if not chave in RepresentacaoConceito.HIPEHIPO:
                RepresentacaoConceito.HIPEHIPO[chave] = dict()

                for syn in wn.synsets(palavra, pos):
                    similares_mono[syn.name()] = []
                    similares[syn.name()] = []

                    for st in syn.similar_tos():
                        similares_dup.add(st.name())
                        similares[syn.name()].append(st.name())                        

                        syns_tmp = wn.synsets(st.name().split('.')[0], pos)
                        # Se palavra similar_tos so tem uma definicao
                        if syns_tmp.__len__() == 1:
                            similares_mono[syn.name()].append(st.name())
                            sim_mono_dup.add(st.name())

                        syns_tmp = None

                    sinonimos_exc[syn.name()] = []

                    for l in syn.lemma_names():
                        if Util.e_mpalavra(l) == False and not l in lista_exc:
                            sinonimos_exc[syn.name()].append(l)
                            uniao_sins.append(l)

                uniao_sins = dict([(k, v) for (k, v) in Counter(uniao_sins).items() if v == 1])

                sim_mono_dup = dict([(k, v) for (k, v) in Counter(sim_mono_dup).items() if v == 1])
                similares_dup = dict([(k, v) for (k, v) in Counter(similares_dup).items() if v == 1])

                for syn_name in sinonimos_exc:
                    sinonimos_exc[syn_name] = list(set(sinonimos_exc[syn_name])&set(uniao_sins.keys()))

                for syn_name in similares:
                    conj1 = set(similares[syn_name])
                    conj2 = set(similares_dup.keys()) # Aqui sao similares que aparecem so uma vez

                    similares_dup[syn_name] = list(conj1.intersection(conj2))

                for syn_name in similares_mono:
                    conj1 = set(similares_mono[syn_name])
                    conj2 = set(sim_mono_dup.keys())

                    similares_mono[syn_name] = list(conj1.intersection(conj2))

        obj_saida = dict()

        todos_syn_names = set(sinonimos_exc)&set(hiperonimos)&set(hiponimos)&set(similares)&set(similares_mono)

        for syn_name in todos_syn_names:
            obj_saida[syn_name] = {
                'sinonimos': sinonimos_exc[syn_name],
                'hiperonimos': hiperonimos[syn_name],
                'hiponimos': hiponimos[syn_name],
                'similares': similares[syn_name], # Similares univocos
                'similares_mono': similares_mono[syn_name] # Similares monossemicos
            }

        return obj_saida

    @staticmethod
    def media_vetores(palavras, normalized_pmis):
        modelo = RV.INSTANCE.modelo

        palavras_tmp = []
        norm_pmis_tmp = []

        for i in range(len(palavras)):
            if palavras[i] in modelo.vocab:
                palavras_tmp.append(palavras[i])
                norm_pmis_tmp.append(normalized_pmis[i])

        palavras = list(palavras_tmp)
        normalized_pmis = list(norm_pmis_tmp)

        # remove out-of-vocabulary words
        palavras = [p for p in palavras if p in modelo.vocab]
        
        # numpy.average(a, axis=None, weights=None, returned=False)
        if len(palavras) >= 1:
            return numpy.average(modelo[palavras], weights=normalized_pmis, axis=0)
        else:
            return None

    @staticmethod
    def sort_by_freq(baseox_inst, palavras, retirar_freq=True):
        palavras = [(p, BaseOx.freq_modelo(baseox_inst, p)) for p in palavras]
        
        if retirar_freq == True:
            return [p for (p, freq) in Util.sort(palavras, col=1, reverse=True) if freq > 0]
        else:
            return Util.sort(palavras, col=1, reverse=True)

    @staticmethod
    def construir_objeto(palavra, pos):
        #    "dir_representacao_conceitos": "./Bases/Cache/RepresentacaoConceitos",
        #    "dir_conceitos_combinados": "./Bases/Cache/ConceitosCombinados"

        # Verificando CACHE
        key = 'dir_representacao_conceitos'
        dir_objeto = Util.CONFIGS['aplicacao'][key] + '/'
        dir_objeto += palavra + '-' + pos + '.json'

        objeto_construido = Util.abrir_json(dir_objeto, criarsenaoexiste=False)

        lista_exc = [palavra.lower()] + Util.CONFIGS['delexical_verbs']

        if objeto_construido != None:
            return objeto_construido
        # Fim da verificacao do CACHE

        objeto_saida = { }

        obter_palavras_exemplos = BasesDesambiguacao.obter_palavras_exemplos
        filtrar_palavra = RepresentacaoConceito.filtrar_palavra

        res = []

        todos_objetos_ox = []
        todos_sins = []
        index_sins = dict()

        identificar_nucleo = RepresentacaoConceito.criar_repr_wordnet
        representacoes_wordnet = identificar_nucleo(palavra, pos)

        lista_hiperonimos = []
        lista_hiponimos = []
        lista_sinonimos = []
        lista_similares = []
        lista_similares_mono = []

        sort_by_freq = RepresentacaoConceito.sort_by_freq

        for synset_name in representacoes_wordnet:
            registro = representacoes_wordnet[synset_name]

            registro['hiperonimos'] = sort_by_freq(BaseOx.INSTANCE, registro['hiperonimos'], retirar_freq=True)            
            lista_hiperonimos.append(registro['hiperonimos'])

            registro['hiponimos'] = sort_by_freq(BaseOx.INSTANCE, registro['hiponimos'], retirar_freq=True)            
            lista_hiponimos.append(registro['hiponimos'])

            registro['sinonimos'] = sort_by_freq(BaseOx.INSTANCE, registro['sinonimos'], retirar_freq=True)
            lista_sinonimos.append(registro['sinonimos'])

            registro['similares'] = sort_by_freq(BaseOx.INSTANCE, registro['similares'], retirar_freq=True)
            lista_similares.append(registro['similares'])

            registro['similares_mono'] = sort_by_freq(BaseOx.INSTANCE, registro['similares_mono'], retirar_freq=True)
            lista_similares_mono.append(registro['similares_mono'])

        for definicao_iter in BaseOx.obter_definicoes(BaseOx.INSTANCE, palavra, pos):
            sins = BaseOx.obter_sins(BaseOx.INSTANCE, palavra, definicao_iter, pos)
            todos_sins += sins
            reg = (definicao_iter, sins)
            todos_objetos_ox.append(reg)

        # Obter sinonimos exclusivos
        todos_sins = [k for (k, v) in dict(Counter(todos_sins)).items() if v == 1]
        resultado = dict()

        palavras_exc_definicoes_ox = [ ]

        for definicao_iter, sins in todos_objetos_ox:
            tokens_def_tmp = nltk.word_tokenize(definicao_iter.lower())
            palavras_exc_definicoes_ox += [p for p in tokens_def_tmp if not p in lista_exc]

        palavras_exc_definicoes_ox = [p for (p, freq) in Counter(palavras_exc_definicoes_ox).items()]

        pmi_max = None
        nucleos = None
        exemplos_completos = []

        for definicao_iter, sins in todos_objetos_ox:
            sins_exclusivos = list(set(sins)&set(todos_sins))
            sins_exclusivos = [s for s in sins_exclusivos if Util.e_mpalavra(s) == False]
            sins_exclusivos = RepresentacaoConceito.sort_by_freq(BaseOx.INSTANCE, sins_exclusivos, retirar_freq=True)

            tokens_definicao = nltk.word_tokenize(definicao_iter)
            definicao_tagueada = nltk.pos_tag(tokens_definicao)
            definicao_util = [p for p in tokens_definicao if filtrar_palavra(p)]

            exemplos_completos = []

            try:

                lista_exc = [Util.singularize(palavra), Util.pluralize(palavra)]
                
                exemplos = BaseOx.obter_atributo(BaseOx.INSTANCE, palavra, pos, definicao_iter, 'exemplos')
                exemplos = nltk.word_tokenize(" ".join(exemplos).lower())
                exemplos_completos = [(p, freq) for (p, freq) in Counter(exemplos).items() if filtrar_palavra(p)]
                exemplos = [(p, freq) for (p, freq) in Counter(exemplos).items() if freq > 1 and filtrar_palavra(p)]
                exemplos = [(p, freq) for (p, freq) in exemplos if not p in definicao_util + lista_exc]
                exemplos = sorted(exemplos, key=lambda x: x[1], reverse=True)
                exemplos_completos = sorted(exemplos_completos, key=lambda x: x[1], reverse=True)

                for p, freq in exemplos:
                    try:
                        pmi_tmp = Alvaro.consultar_pmi(palavra, None, p, None)
                    except: pass
            except Exception, e:
                pass

            if definicao_util:
                def_util_token = nltk.word_tokenize(" ".join(definicao_util).lower())

                nucleos = set(Util.mesclar_listas(lista_hiperonimos))&set(def_util_token)
                nucleos.update(list(set(Util.mesclar_listas(lista_hiponimos))&set(def_util_token)))

                if nucleos.__len__() == 0:
                    nucleos.update(sins_exclusivos)
                    nucleos.update(list(set(Util.mesclar_listas(lista_similares_mono))&set(def_util_token)))
                if nucleos.__len__() == 0:
                    nucleos.update(list(set(Util.mesclar_listas(lista_similares))&set(def_util_token)))
                if nucleos.__len__() == 0:
                    nucleos.update(list(set(Util.mesclar_listas(lista_sinonimos))&set(def_util_token)))
                if nucleos.__len__() == 0:
                    nucleos.update(set(def_util_token)&set(palavras_exc_definicoes_ox))

                lista_hiperonimos = set(Util.mesclar_listas(lista_hiperonimos))&set(def_util_token)
                lista_hiponimos = set(Util.mesclar_listas(lista_hiponimos))&set(def_util_token)
                lista_similares = set(Util.mesclar_listas(lista_similares))&set(def_util_token)
                lista_similares_mono = set(Util.mesclar_listas(lista_similares_mono))&set(def_util_token)
                lista_sinonimos = set(Util.mesclar_listas(lista_sinonimos))&set(def_util_token)

                todas_palavras = list(lista_hiperonimos)
                todas_palavras += list(lista_hiponimos)
                todas_palavras += list(lista_similares)
                todas_palavras += list(lista_similares_mono)
                todas_palavras += list(lista_sinonimos)

                todas_palavras += list(def_util_token)
                todas_palavras += [p for (p, freq) in exemplos]

                try:
                    pmi_max = Alvaro.consultar_pmi(palavra, None, palavra, None)
                except:
                    pmi_max = 5.00

                flexoes_target = [Util.pluralize(palavra.lower()), Util.singularize(palavra.lower())]

                todas_palavras = list(set(todas_palavras)-set(flexoes_target))

                correlacoes = []
                nucleo_pmi = []

                for relacionada in todas_palavras:
                    try:
                        try:
                            pont_pmi = Alvaro.consultar_pmi(palavra, None, relacionada, None)
                        except:
                            if relacionada in nucleos:
                                pont_pmi = pmi_max
                        if pont_pmi >= 0.00:
                            if relacionada in nucleos:
                                nucleo_pmi.append((relacionada.lower(), pont_pmi))
                            correlacoes.append((palavra, relacionada.lower(), pont_pmi))
                        else: pass
                    except Exception, e:
                        pass

                correlacoes = [eval(reg) for reg in set([str(reg) for reg in correlacoes])]
                correlacoes = sorted(correlacoes, key=lambda x: x[2], reverse=True)

                todas_palavras = []
                todos_pesos = []

                nucleos = [p for (p, pmi) in Util.sort(nucleo_pmi, col=1, reverse=True)][:1]
                soma_total = sum([pmi_max if correlata_iter in nucleos else pmi for (target_iter, correlata_iter, pmi) in correlacoes])

                for target_iter, correlata_iter, pmi in correlacoes:
                    if not correlata_iter in flexoes_target:
                        todas_palavras.append(correlata_iter)
                    
                        if correlata_iter in nucleos:
                            pcent = float(pmi_max)/soma_total
                            todos_pesos.append(pcent)
                        else:
                            pcent = float(pmi)/soma_total
                            todos_pesos.append(pcent)
            else:
                pass

            resultado[definicao_iter] = {
                'nucleos': list(nucleos),
                'lista_hiperonimos': list(lista_hiperonimos),
                'lista_hiponimos': list(lista_hiponimos),
                'todas_palavras': list(todas_palavras),
                'sins_exclusivos': list(sins_exclusivos),
                'lista_sinonimos': list(lista_sinonimos),
                'lista_similares': list(lista_similares),
                'lista_similares_mono': list(lista_similares_mono),
                'exemplos': list(exemplos_completos),
            }

        todos_sins = None
        try:
            max_frequencia = exemplos_completos[0][1]
        except:
            max_frequencia = 0

        for definicao_iter in resultado:
            nucleos = resultado[definicao_iter]['nucleos']
            todas_palavras = { }
            try:
                exemplos_ordenados = resultado[definicao_iter]['exemplos']
            except:
                exemplos_ordenados = []

            if exemplos_ordenados:
                lista_exc = [Util.singularize(palavra), Util.pluralize(palavra)]
                exemplos_ordenados = [(p, freq) for (p, freq) in exemplos_ordenados if freq > 1 and not p in lista_exc and filtrar_palavra(p)]

            lista_unificada = [reg for reg in Util.mesclar_listas(resultado[definicao_iter].values())]
            lista_unificada = [reg for reg in lista_unificada if type(reg) in [str, unicode] and not reg in nucleos]

            for n in nucleos:
                todas_palavras[n] = pmi_max
            for p in lista_unificada:
                if not p in todas_palavras:
                    try:
                        pmi_tmp = Alvaro.consultar_pmi(palavra, None, p, None)
                        todas_palavras[p] = pmi_tmp
                    except: pass
            for p, freq in exemplos_ordenados:
                if not p in todas_palavras:
                    try:
                        pmi_tmp = Alvaro.consultar_pmi(palavra, None, p, None)
                        pmi_tmp = pmi_tmp * (float(freq)/float(max_frequencia))
                        todas_palavras[p] = pmi_tmp
                    except: pass

            resultado[definicao_iter]['pmi'] = todas_palavras

        Util.salvar_json(dir_objeto, resultado)

        return resultado


class LexicalEmbeddings(object):
    LEMMATIZER = WordNetLemmatizer()
    STEMMER = PorterStemmer()
    REGRAS_POSTAGS = {
        "a": list("anv"),
        "n": list("nv"),
        "r": list("r"),
        "v": list("anv")
    }

    @staticmethod
    def conversor(palavra, definicao, lista_sinonimos):
        nucleo = raw_input("Nucleo: ")
        sec = raw_input("Secundaria: ")

        sec = re.sub("[,.:\s]", ";", sec).split(';')

        vetores = [RepVetorial.INSTANCE.modelo[w] for w in [nucleo] + sec]

        vaverage = numpy.average(vetores, axis=0)
        vsum = numpy.sum(vetores, axis=0)

        try:
            vaverage = RepVetorial.INSTANCE.modelo.similar_by_vector(vaverage, topn=20)
            vsum = RepVetorial.INSTANCE.modelo.similar_by_vector(vsum, topn=20)

            res = []
            for i in range(len(vsum)):
                if i < len(vaverage):
                    reg = [vsum[i], vaverage[i]]
                else:
                    reg = [vsum[i], None]
                res.append(tuple(reg))

            return res

        except Exception, e:
            print("\n\nExcecao! %s\n\n"%str(e))
            return []

    @staticmethod
    def representar_conceitos(palavra, pos):
        identificar_conceito_nuclear = LexicalEmbeddings.identificar_conceito_nuclear
        definicoes = list()

        for definicao in BaseOx.obter_definicoes(BaseOx.INSTANCE, palavra, pos):
            todos_pmis = identificar_conceito_nuclear(palavra, definicao.lower(), pos)
            definicoes.append((definicao, todos_pmis))

        return list(definicoes)


class Senseval2(object):
    @staticmethod
    def carregar(diretorio):
        resultado = []

        parser = etree.XMLParser(recover=True)
        arvore_xml = etree.parse(diretorio, parser)
        raiz = arvore_xml.getroot()

        for lexelt in raiz.getchildren():
            for instance in lexelt.getchildren():
                id_reg = instance.attrib['id']
                remove_tag_inserido = False
                for context in instance.getchildren():
                    target = str(context.xpath("./head/text()")[0])
                    lema, pos = [v for k, v in lexelt.items() if k ==
                                 'item'][0].split('.')
                    nodos_aninhados = context.xpath("./node()")

                    xml_aninhado = []  # = innerHTML construido manualmente

                    for n in nodos_aninhados:
                        if type(n) == lxml.etree._Element:
                            if remove_tag_inserido == False:
                                nodo = "%s %s" % ("REMOVETAG", n.text)
                                xml_aninhado.append(nodo)
                                remove_tag_inserido = True
                            else:
                                xml_aninhado.append(n.text)
                        else:
                            xml_aninhado.append(n)

                    xml_aninhado = " ".join(xml_aninhado)
                    xml_aninhado.replace("\n", " ")
                    xml_aninhado = re.sub(' +', ' ', xml_aninhado)

                    resultado.append((id_reg, xml_aninhado, target, lema, pos))

        return resultado    

    #   Recebe o arquivo formatado com as predicoes para as instancias
    #   Chama uma versao modificada do script original
    @staticmethod
    def pontuar(dir_predicao):
        # Versao adaptada do script original
        from senseval2_scorer import calcular_pontuacao
    
        # referencia ao configs.json
        index = 'senseval2-lexicalSample'

        dir_gold = Util.CONFIGS[index]['gold']
        dir_sensemap =  Util.CONFIGS[index]['sensemap']

        args = ["metodologia", "-f", dir_predicao, dir_gold, dir_sensemap]

        return calcular_pontuacao(args)

class Senseval3(object):
    @staticmethod
    def carregar(diretorio):
        resultado = []

        parser = etree.XMLParser(recover=True)
        arvore_xml = etree.parse(diretorio, parser)
        lexelts = arvore_xml.xpath('lexelt')

        for lexelt in lexelts:
            for instance in lexelt.getchildren():
                for context in instance.getchildren():
                    target = str(context.xpath("./head/text()")[0])
                    lema, pos = [v for k, v in lexelt.items() if k ==
                                 'item'][0].split('.')
                    nodos_aninhados = context.xpath("./node()")

                    xml_aninhado = []  # = innerHTML construido manualmente

                    for n in nodos_aninhados:
                        if type(n) == lxml.etree._Element:
                            nodo = "%s %s" % ("REMOVETAG", n.text)
                            xml_aninhado.append(nodo)
                        else:
                            xml_aninhado.append(n)

                    xml_aninhado = " ".join(xml_aninhado)
                    xml_aninhado.replace("\n", " ")
                    xml_aninhado = re.sub(' +', ' ', xml_aninhado)

                    resultado.append((xml_aninhado, target, lema, pos))

        return resultado

    #   Recebe o arquivo formatado com as predicoes para as instancias
    @staticmethod
    def pontuar(dir_predicao):
        return []

class BasesDesambiguacao(object):
    SYNSETS = dict()
    PORTER_STEMMER = PorterStemmer()
    LEMMATIZER = WordNetLemmatizer()

    SUPER_LISTA_STOPWORDS = []
    CACHE_PALAVRAS_EXEMPLO = dict()
    CONTEXTO_TAGUEADO = None

    @staticmethod
    def abrir_contextos_tagueados():
        try:
            diretorio = Util.CONFIGS['desambiguacao']['diretorio_contexto_tagueado']
            BasesDesambiguacao.CONTEXTO_TAGUEADO = Util.abrir_json(diretorio, criarsenaoexiste=True)
            return True
        except Exception, e:
            pass

        return False

    @staticmethod
    def salvar_contextos_tagueados():
        if BasesDesambiguacao.CONTEXTO_TAGUEADO != None:
            if BasesDesambiguacao.CONTEXTO_TAGUEADO.__len__() > 0:
                diretorio = Util.CONFIGS['desambiguacao']['diretorio_contexto_tagueado']
                return Util.salvar_json(diretorio, BasesDesambiguacao.CONTEXTO_TAGUEADO)

        return False

    '''
        Retorna uma lista de palavras presentes nos exemplos
        no formato <lista<palavras>, Lista<palavra, freq>>.
        É um CONJUNTO-UNIAO de palavras de exemplos de definicoes
    '''
    @staticmethod
    def obter_palavras_exemplos(palavra, pos):
        inst = BaseOx.INSTANCE
        palavras = []
        exemplos = []

        if BasesDesambiguacao.SUPER_LISTA_STOPWORDS.__len__() > 0:
            sw = list(string.punctuation) + BasesDesambiguacao.SUPER_LISTA_STOPWORDS
            sw = [p.lower() for p in sw]
        else:
            raise Exception("A lista de stopwords esta vazia!")

        for defi in BaseOx.obter_definicoes(inst, palavra, pos):
            exemplos += [f.lower() for f in BaseOx.obter_atributo(inst,
                                                                  palavra, pos, defi, 'exemplos')]

        for ex in exemplos:
            palavras += [p for p in nltk.word_tokenize(ex) if not p in sw]

        return palavras, Counter(palavras)

    @staticmethod
    def mesclar_inventarios(palavra, pos, inventario_base='wordnet', usar_lemas=False):
        key = 'dir_conceitos_combinados'
        dir_objeto = Util.CONFIGS['aplicacao'][key] + '/'
        dir_objeto += palavra + '-' + pos + '.json'

        obj_retorno = Util.abrir_json(dir_objeto, criarsenaoexiste=False)

        if obj_retorno: return obj_retorno
	else: obj_retorno = { }

        inventario_base='wordnet'

        PORTER_STEMMER = BasesDesambiguacao.PORTER_STEMMER
        LEMMATIZER = BasesDesambiguacao.LEMMATIZER

        lista_retorno = list()

        sw = BasesDesambiguacao.SUPER_LISTA_STOPWORDS + \
            [p.lower() for p in BasesDesambiguacao.SUPER_LISTA_STOPWORDS]
        sw += list(string.punctuation)

        inventario_wn = dict() # <definicao (synset.name()): lista de tokens da definicao>
        inventario_ox = dict() # <definicao : lista de tokens da definicao>

        repvet_inst = RepVetorial.INSTANCE

        casamentos = dict()

        todos_sins = dict() # <definicao : lista de sinonimos>

        for d_ox in BaseOx.obter_definicoes(BaseOx.INSTANCE, palavra, pos):
            sins = BaseOx.obter_sins(BaseOx.INSTANCE, palavra, d_ox, pos)
            inventario_ox[d_ox.lower()] = [w for w in nltk.word_tokenize(
                d_ox.lower()) if not w in sw]
            todos_sins[d_ox.lower()] = sins

            # Alias
            if len(inventario_ox[d_ox.lower()]) == 1:
                alias = inventario_ox[d_ox.lower()][0]
                for d_ox in BaseOx.obter_definicoes(BaseOx.INSTANCE, alias, pos):
                    sins = BaseOx.obter_sins(BaseOx.INSTANCE, alias, d_ox, pos)
                    inventario_ox[d_ox.lower()] = [w for w in nltk.word_tokenize(
                        d_ox.lower()) if not w in sw]
                    todos_sins[d_ox.lower()] = sins

        for s in wn.synsets(palavra, pos):
            d_wn = [w for w in nltk.word_tokenize(
                s.definition().lower()) if not w in sw]

            inventario_wn[s.name()] = d_wn

            for d_ox in inventario_ox:
                vet1_tmp = [PORTER_STEMMER.stem(w)
                            for w in inventario_ox[d_ox]]
                vet2_tmp = [PORTER_STEMMER.stem(w) for w in d_wn]

                # Mede a sobreposicao entre palavras da definicao
                cosseno = Util.cosseno(vet1_tmp, vet2_tmp, tipo='list')
                jaccard = Util.jaccard(vet1_tmp, vet2_tmp, tipo='list')

                sins_wn = s.lemma_names() + \
                    Util.mesclar_listas([h.lemma_names()
                                         for h in s.hypernyms() + s.hyponyms()])
                sins_wn = list(set(sins_wn))

                if palavra in sins_wn:
                    sins_wn.remove(palavra)
                if palavra in todos_sins[d_ox]:
                    todos_sins[d_ox].remove(palavra)

                try:
                    # Mede a sobreposicao entre sinonimos da definicao
                    lesk_lemma = float(len(set(todos_sins[d_ox]) & set(
                        sins_wn)))/max(len(todos_sins[d_ox]), len(sins_wn))
                except ZeroDivisionError, e:
                    lesk_lemma = 0

                wmd = RepVetorial.word_move_distance(
                    repvet_inst, d_ox.lower(), s.definition().lower())

                if wmd > Util.CONFIGS['wmd_max']:
                    wmd = Util.CONFIGS['wmd_max']

                wmd_percent = float(Util.CONFIGS['wmd_max'] - wmd) / Util.CONFIGS['wmd_max']

                lesk_lemma_def1 = len(
                    set(todos_sins[d_ox]) & set(d_wn))/float(len(d_wn))
                lesk_lemma_def2 = len(set(sins_wn) & set(
                    inventario_ox[d_ox]))/float(len(inventario_ox[d_ox]))
                lesk_lemma_def = max(lesk_lemma_def1, lesk_lemma_def2)

                #if cosseno + jaccard + lesk_lemma + lesk_lemma_def != 0.00 and wmd != float('inf'):
                reg = {
                    'cosseno': cosseno,
                    'jaccard': jaccard,
                    'lesk_lema': lesk_lemma,
                    'lesk_lemma_def': lesk_lemma_def,
                    'wmd_percent': wmd_percent,
                    'wmd': wmd
                }

                reg['definicoes'] = {
                    'wordnet': s.definition().lower(),
                    'oxford': d_ox.lower()
                }

                try:
                    media = sum(reg.values()) / float(len(reg.values()))
                    reg['media'] = media

                    lista_retorno.append(reg)
                except:
                    print("\t@@@ Excecao na mescla de inventarios: " + str((palavra, pos)))

        Util.salvar_json(dir_objeto, lista_retorno)

        return lista_retorno

    @staticmethod
    def lesk(palavras_tagueadas): # Pares (palavra, pos)
        for palavra, pos in palavras_tagueadas:
            pass
        return []

    @staticmethod
    def consultar_palavras_exemplos(target, pos):
        BasesDes = BasesDesambiguacao

        if not "%s-%s"%(target, pos) in BasesDes.CACHE_PALAVRAS_EXEMPLO:
            BasesDes.CACHE_PALAVRAS_EXEMPLO["%s-%s"%(target, pos)] = None
            BasesDes.CACHE_PALAVRAS_EXEMPLO["%s-%s"%(target, pos)] = { }

            reg_tmp = BasesDes.obter_palavras_exemplos(target, pos)
            BasesDes.CACHE_PALAVRAS_EXEMPLO["%s-%s"%(target, pos)] = reg_tmp

        try:
            return BasesDes.CACHE_PALAVRAS_EXEMPLO["%s-%s"%(target, pos)]
        except:
            return list(), dict()

    """
        Base de desambiguacao - Base lexical
        ID da instancia - ID da instancia
    """
    @staticmethod
    def desambiguar(nome_base, chave, contexto, target, pos, marcador):
        RC = RepresentacaoConceito

        sing = Util.singularize
        plural = Util.pluralize

        if nome_base == None or nome_base == "":
            raise Exception("\tNome base invalido!")
        if chave == None or chave == "":
            raise Exception("\tNome base invalido!")

        stop_words = BasesDesambiguacao.SUPER_LISTA_STOPWORDS + \
            list(string.punctuation)

        stop_words += Util.CONFIGS['delexical_verbs']
        stop_words += [w.lower() for w in stop_words]

        punctuation = string.punctuation

        # Cria uma representacao do conceito baseado nas relacoes linguisticas 
        representacao_conceitos = RC.construir_objeto(Util.singularize(target.lower()), pos)
        inventario_embedd = dict()

        vetor_conceitos = dict()

        for reg in representacao_conceitos:
            pmis_norm = []
            vetor_conceitos[reg] = pmis_norm
            todos_pmis_tmp = representacao_conceitos[reg]['pmi']
            sum_pmi = sum([pmi_iter for pmi_iter in todos_pmis_tmp.values() if pmi_iter >= 0.00])

            for palavra, pmi_iter in todos_pmis_tmp.items():
                if pmi_iter >= 0.00:
                    pmis_norm.append((palavra, pmi_iter/sum_pmi))

            vetor_conceitos[reg] = pmis_norm

            palavras_conceito = [k for (k, v) in pmis_norm]
            pesos_pmi = [v for (k, v) in pmis_norm]

            vetor_conceito = RepresentacaoConceito.media_vetores(palavras_conceito, pesos_pmi)
            inventario_embedd[reg] = vetor_conceito

        # Tamanho janela
        max_contexto = 20

        regras_associacao = {
            "a": list("anrv"),
            "n": list("anrv"),
            "r": list("anrv"),
            "v": list("anrv"),
        }

        janela_esquerda = []
        janela_direita = []

        contexto = contexto.lower()
        target = target.lower()

        chave_registro = nome_base + '@@@' + chave
        
        if chave_registro in BasesDesambiguacao.CONTEXTO_TAGUEADO:
            tokens_tagueados = BasesDesambiguacao.CONTEXTO_TAGUEADO[chave_registro]
        else:
            tokens_tagueados = nltk.pos_tag(nltk.word_tokenize(contexto.lower()))
            BasesDesambiguacao.CONTEXTO_TAGUEADO[chave_registro] = tokens_tagueados

        contexto_tagueado = [(t, pt[0].lower() if pt[0] != "J" else "a")
                            for (t, pt) in tokens_tagueados if not t in stop_words]

        tokens_filtrados_dicionario = [t for t, pt in contexto_tagueado]
        indice_marcador = tokens_filtrados_dicionario.index(marcador.lower())

        palavras_ex, palavras_freq = BasesDesambiguacao.consultar_palavras_exemplos(target, pos)

        target_sing, target_plural = sing(target), plural(target)

        if target_sing in palavras_freq:
            del palavras_freq[target_sing]
        if target_plural in palavras_freq:
            del palavras_freq[target_plural]

        palavras_ex = palavras_freq.keys()

        del tokens_filtrados_dicionario[indice_marcador]

        # Janela esquerda
        colisoes = set()
        indice = indice_marcador - 1

        # Criando a janela esquerda
        while indice != 0 and len(janela_esquerda) < max_contexto:
            t, pt = contexto_tagueado[indice]
            if pt in regras_associacao[pos] and not t.lower() in stop_words:
                try:
                    colisoes.add(t)
                    janela_esquerda.append(
                        (target, pos, t, pt, Alvaro.consultar_pmi(target, pos, t, pt)))
                except (ZeroDivisionError, ValueError), e:
                    pass
            indice -= 1

        janela_esquerda = janela_esquerda[:max_contexto][::-1]

        # Janela direita (+2 = skip tag + target)
        colisoes = set()
        indice = indice_marcador + 2

        # Criando a janela direita
        while indice < len(contexto_tagueado) and len(janela_direita) < max_contexto:
            t, pt = contexto_tagueado[indice]
            if pt in regras_associacao[pos] and not t.lower() in stop_words:
                try:
                    colisoes.add(t)
                    janela_direita.append(
                        (target, pos, t, pt, Alvaro.consultar_pmi(target, pos, t, pt)))
                except (ZeroDivisionError, ValueError), e:
                    pass
            indice += 1

        janela_direita = janela_direita[:max_contexto]

        # As duas janelas, a esquerda e a direita estao formadas
        ranking_pmi = Util.sort(
            janela_esquerda + janela_direita, col=4, reverse=True)

        if palavras_ex:
            target_flex = [sing(target), plural(target)]
            lista_pmi = [t for (target, pos, t, pt, pmi) in ranking_pmi if not t in target_flex]
            intersec_pmiex = list(set(palavras_ex).intersection(set(lista_pmi)))
        else: 
            intersec_pmiex = []

        tokens = [p for (p, pt) in tokens_tagueados]

        intersec_ctxex = list(set(palavras_ex).intersection(set(tokens)))

        intersec_pmiex = [p.lower() for p in intersec_pmiex]
        intersec_ctxex = [p.lower() for p in intersec_ctxex]

        intersec_pmiex = list(set(intersec_pmiex) - set(stop_words))
        intersec_ctxex_tmp = list(set(intersec_ctxex) - set(stop_words))
        intersec_ctxex = []

        for palavra in intersec_ctxex_tmp:
            try:
                pmi_tmp = Alvaro.consultar_pmi(target, None, palavra, None)
                intersec_ctxex.append((target, palavra, pmi_tmp))
            except Exception, e:
                pass

        ranking_pmi = [reg for reg in ranking_pmi if sing(reg[2]) != sing(target)]
        somatorio_pesos = 0.00

        todas_palavras = { }

        intersec_pmiex_tmp = []
        for rel in intersec_pmiex:
            try: intersec_pmiex_tmp.append((target, rel, Alvaro.consultar_pmi(target, None, rel, None)))
            except: pass

        colecoes = [
            ([(t, rel, pmi_iter) for (t, ptiter, rel, prel, pmi_iter) in ranking_pmi], 1.00),
            (intersec_ctxex, 4.00),
            (intersec_pmiex_tmp, 0.1)
        ]

        intersec_pmiex_tmp = None

        for colecao, peso_colecao in colecoes:
            for reg_iter in colecao:
                target_iter, relacionada_iter, pmi_iter = reg_iter
                if pmi_iter > 0.00 and not target_iter.lower() in relacionada_iter.lower():
                    if not relacionada_iter in todas_palavras:
                        pmi_iter_ponderado = pmi_iter / peso_colecao
                        somatorio_pesos += pmi_iter_ponderado
                        todas_palavras[relacionada_iter] = pmi_iter_ponderado

        for relacionada_iter in todas_palavras:
            todas_palavras[relacionada_iter] = (todas_palavras[relacionada_iter])/float(somatorio_pesos)

        intersec = dict()
        consultar_documentos = Whoosh.consultar_documentos

        filtrar_contextos_promissores = False
        if filtrar_contextos_promissores == True:
            for args_cons in list(itertools.product([target.lower()], [p[0] for p in Util.sort(todas_palavras.items(), col=1, reverse=True)][:20])):
                palavras_docs = list(args_cons)
                todos_docs = consultar_documentos(palavras_docs, operador='AND', limite=10000, dir_indexes=Whoosh.DIR_INDEXES)

                todos_docs = " ".join([doc['content'] for doc in todos_docs])
                tokens_docs = nltk.word_tokenize(todos_docs.lower())
                todos_docs = [(k, v) for (k, v) in Counter(tokens_docs).items() if v > 2 and not k in stop_words]
                todos_docs = Util.sort(todos_docs, col=1, reverse=True)

                for palavra_iter, freq_palavra in todos_docs:
                    if not palavra_iter.lower() in intersec:
                        intersec[palavra_iter.lower()] = 0
                    intersec[palavra_iter.lower()] += 1

                todos_docs, tokens_docs = None, None

            print('\n\n')
            intersec = Util.sort([(k, v) for (k, v) in intersec.items() if v > 1 and not k in stop_words], col=1, reverse=True)
            print("CONTEXTOS PROMISSORES:\n")
            print(intersec)
            raw_input('\n\n<enter>')
            print('\n\n')

        todas_palavras_tmp = todas_palavras.items()
        palavras_ctx = [k for (k, v) in todas_palavras_tmp]
        pesos_palavas = [v for (k, v) in todas_palavras_tmp]

        vetor_contexto = RC.media_vetores(palavras_ctx, pesos_palavas)
        resultado = []

        try:
            for definic_ox in inventario_embedd:
                cosine = spatial.distance.cosine
                dist_cos = 1.00 - cosine(inventario_embedd[definic_ox], vetor_contexto)
                resultado.append((definic_ox, dist_cos))
        except:
            return (ranking_pmi, intersec_pmiex, intersec_ctxex, todas_palavras, [])

        resultado = Util.sort(resultado, col=1, reverse=True)

        # Mescla inventarios Oxford + Wordnet
        # baseado em medidas como: Lesk, WMD, etc...
        mesc_invent = BasesDesambiguacao.mesclar_inventarios        
        invent_mesclado_target = mesc_invent(sing(target.lower()), pos, inventario_base='wordnet')
        invent_mesclado = dict()

        for registros_iter in invent_mesclado_target:
            def_ox = registros_iter['definicoes']['oxford']
            if not def_ox in invent_mesclado:
                invent_mesclado[def_ox] = []
            if registros_iter['media'] > 0.00:
                invent_mesclado[def_ox].append(registros_iter)

        resultado_final = { }

        for definic_ox in invent_mesclado:
            defs_casadas = invent_mesclado[definic_ox]
            defs_casadas = sorted(defs_casadas, key=itemgetter('media'), reverse=True)
            defs_casadas = [d['definicoes']['wordnet'] for d in defs_casadas]

            resultado_final[definic_ox] = defs_casadas

        resultado_final = dict([(k.lower(), v) for (k, v) in resultado_final.items()])

        res = []
        for def_ox, cos in resultado:
            try:
                for def_wn in resultado_final[def_ox.lower()]:
                    res.append((def_ox, cos, def_wn))
            except: 
                return (ranking_pmi, intersec_pmiex, intersec_ctxex, todas_palavras, [])

        res_tmp = list(res)
        res = dict()

        syn_keys = { }
        for s in wordnet.synsets(target, pos):
            try:
                def_tmp= s.definition().lower()
                syn_keys[def_tmp] = s.lemmas()[0].key()
            except: pass

        for def_ox, cos, def_wn in res_tmp:
            if not def_wn in res:
                res[def_wn] = (cos, syn_keys[def_wn.lower()])
    
        res = Util.sort(res.items(), col=1, reverse=True)

        # Ranking PMI = PMI -> Pequenos contextos
        # intersec = Intersecao de exemplos e ranking_pmi
        # intersec_ctxex = Intersecao de exemplos e contexto amplo
        # todas_palavras = {palavra : peso_pmi_normalizado}

        return (ranking_pmi, intersec_pmiex, intersec_ctxex, todas_palavras, res)


if os.path.basename(__file__) == argv[0]:
    cfgs = Util.carregar_cfgs(argv[1])
    Util.CONFIGS = cfgs

    BasesDesambiguacao.abrir_contextos_tagueados()

    stop_words = mesclar_listas(abrir_json(
        Util.CONFIGS['super_lista_stopwords']).values())
    BasesDesambiguacao.SUPER_LISTA_STOPWORDS = stop_words

    stop_words = Util.abrir_json(Util.CONFIGS['super_lista_stopwords']).values()

    RepresentacaoConceito.STOP_WORDS = Util.mesclar_listas(stop_words)
    RepresentacaoConceito.STOP_WORDS = [p.lower() for p in RepresentacaoConceito.STOP_WORDS]
    RepresentacaoConceito.STOP_WORDS = [p.strip() for p in RepresentacaoConceito.STOP_WORDS]

    try:
        print('\n')
        print(Senseval2.pontuar('./saida_teste.txt'))
    except:
        pass

    if argv.__len__() >= 3:
        funcao = None if argv[2] != 'aval_saida' else argv[2]
    else:
        funcao = None

    InterfaceBases.setup(cfgs, dir_keys="./keys.json", funcao=funcao)

    desambiguar = BasesDesambiguacao.desambiguar

    indice = 1
    registros_senseval = Senseval2.carregar("./teste.xml")

    todas_palavras = [ ]
    for r in registros_senseval:
        palavra, pos = r[::-1][:2][::-1]
        todas_palavras.append((palavra, pos))

    # reg = BasesDesambiguacao.mesclar_inventarios(palavra, pos, inventario_base='wordnet')

    instancia_exemplos = 0
    instancia_exemplos_pmi = 0

    if os.path.basename(__file__) == argv[0]:
        if Alvaro.PMI_TAG == None or Alvaro.PMI_TAG == dict():
            Alvaro.PMI_TAG = Util.abrir_json(
                Util.CONFIGS['metodo_pmi']['dir_contadores_tag'])

        saida_generica = dict()

        try:
            print("\n\n")
            for id_reg, ctx, target, lema, pos in registros_senseval:
                try:
                    if indice >= 0:
                        contexto = Util.descontrair(ctx)
                        contexto = "".join(
                            [" " if c in string.punctuation else c for c in ctx])
                        contexto = re.sub(' +', ' ', contexto)

                        # Ranking PMI = PMI -> Pequenos contextos
                        # intersec = Intersecao de exemplos e ranking_pmi
                        # intersec_ctxex = Intersecao de exemplos e contexto amplo
                        # return (ranking_pmi, intersec_pmiex, intersec_ctxex, palavras_ponderadas)
                    
                        print("\n\n*** Instancia " + str(id_reg))
                        # palavras_ponderadas = {palavra : peso_pmi_normalizado}
                        saida_des = desambiguar('senseval2', id_reg, contexto, target, pos, "REMOVETAG")
                        ranking_pmi, intersec, intersec_ctxex, palavras_ponderadas, resultado = saida_des
                        ranking_ppmi = [(target, pos, t, pt, pmi) for (target, pos, t, pt, pmi) in ranking_pmi]
                        intersec_ctxex = [(target, t, pmi) for (target, t, pmi) in intersec_ctxex]

                        # work work.159 work%2:41:11::

                        lema = unicode(id_reg.split('.')[0])

                        if len(resultado) > 0:
                            lema_synset = unicode(resultado[0][1][1])
                            os.system('echo "%s %s %s" >> saida_teste.txt'%(lema, unicode(id_reg), lema_synset))

                        pcent = float(indice) / float(len(registros_senseval))
                        print("\n" + str((target, pos)))
                        print(str(pcent) + "%\t" + str(indice) +
                            "/" + str(len(registros_senseval)) + " - " + contexto)

                        print("\t=> Ranking PPMI")
                        for reg in ranking_ppmi:
                            print("\t- " + unicode(reg))
                        print("\t=> Ranking Contexto-Exemplos")
                        for reg in intersec_ctxex:
                            print("\t- " + unicode(reg))
                        print("\t=> Ranking Contexto-Exemplos-PMI")
                        for reg in intersec:
                            print("\t- " + unicode(reg))

                        print("\n\n\n")

                        instancia_exemplos += 1 if intersec_ctxex else 0
                        instancia_exemplos_pmi += 1 if intersec else 0

                except KeyboardInterrupt, ke:
                    raise KeyboardInterrupt("A execucao foi interrompida...")
                except Exception, e:
                    traceback.print_exc()

                indice += 1

        except KeyboardInterrupt, ke:
            pass

    print("\n\tINSTANCIA COM EXEMPLOS: " + str(instancia_exemplos) + "\n\n")
    print("\n\tINSTANCIA COM SOBREPOSICAO PMI: " + str(instancia_exemplos_pmi) + "\n\n")    

    print("\n\n\n")
    print("\tSalvando no CACHE os contextos tagueados...")
    BasesDesambiguacao.salvar_contextos_tagueados()
    print("\tContextos tagueados salvos no CACHE...")

    # Salvando contadores PMI
    if Alvaro.PMI_TAG != None:
        dir_pmi_tag = Util.CONFIGS['metodo_pmi']['dir_contadores_tag']
        if Alvaro.PMI_TAG.__len__() > 0:
            print("\n\tSalvando objeto PMI...")
            Util.salvar_json(dir_pmi_tag, Alvaro.PMI_TAG)
            print("\tObjeto PMI salvo!\n")
