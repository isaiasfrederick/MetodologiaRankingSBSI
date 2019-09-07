# -*- coding: utf-8 -*-
import glob
import hashlib
import json
import math
import os
import random
import re
import string
import unicodedata
from collections import Counter
from os import system
from sys import version_info
from unicodedata import normalize

#from pattern.en import conjugate
import sys

import bencode
import nltk
import requests
import textblob
from nltk.corpus import stopwords, wordnet
from pywsd.utils import lemmatize, lemmatize_sentence, porter

wn = wordnet

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

class Util(object):
    MAX_WMD = 1000
    configs = None
    # Contadores Corpus
    contadores = None
    verbose_ativado = True

    URLS_INVALIDAS = set()
    CONFIGS = None

    @staticmethod
    def get(url):
        return None

    @staticmethod
    def sort(colecao, col, reverse=False):
        return sorted(colecao, key=lambda x: x[col], reverse=reverse)

    @staticmethod
    def stem(palavra):
        return ps.stem(palavra.lower())

    @staticmethod
    def media(vetor):
        try:
            return sum(vetor)/len(vetor)
        except: return 0.00

    @staticmethod
    def norm_palavra(palavra):
        return palavra

    @staticmethod
    def requisicao_http(url, headers=None, admite_404=False):
        if url in Util.URLS_INVALIDAS:
            return None

        print("\nRequerindo URL: %s\n"%url)

        if headers:
            res = requests.get(url, headers = headers)
        else:
            res = requests.get(url)

        print("\nResultado: %s\n"%str(res.status_code))
        print("Admite 404: " + str(admite_404))

        if res.status_code == 200 or admite_404 == True:
            return res
        else:
            Util.URLS_INVALIDAS.add(url)
            return None

    @staticmethod
    def pontuacao_valida(operando_variavel, medida_similaridade):
        if medida_similaridade == 'word_move_distance':
            return True
        elif medida_similaridade == 'cosine' and (operando_variavel > 0.00):
            return True
        return False

    @staticmethod
    def singularize(palavra):
        return textblob.Word(palavra).singularize().replace("'", "")

    @staticmethod
    def pluralize(palavra):
        return textblob.Word(palavra).pluralize().replace("'", "")

    @staticmethod
    def md5sum(objeto):
        import hashlib
        import bencode
        data_md5 = hashlib.md5(bencode.bencode(objeto)).hexdigest()
        return data_md5

    @staticmethod
    def md5sum_string(s):
        import hashlib
        m = hashlib.md5()
        m.update(s)
        return m.hexdigest()

    @staticmethod
    def cvrsr_pos_wn_oxford(pos):
        #ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
        
        if pos == 'n': pos = 'Noun'
        elif pos == 'v': pos = 'Verb'
        elif pos == 'r': pos = 'Adverb'
        elif pos == 'a': pos = 'Adjective'

        return pos

    @staticmethod
    def cvsr_pos_semeval_ox(pos):
        return Util.cvrsr_pos_wn_oxford(pos)

    @staticmethod
    def cvsr_pos_semeval_wn(pos):
        if pos == 'a': return 's'
        
        return pos

    @staticmethod
    def conversor_pos_oxford_wn(pos):
        if pos in ['Noun', 'Verb', 'Adjective']:
            return pos[0].lower()
        elif pos == 'Adverb':
            return 'r'
        elif pos == 'Conjunction':
            return 'r'

        return pos

    @staticmethod
    def descontrair(txt):
        # specific
        txt = re.sub(r"won't", "will not", txt)
        txt = re.sub(r"can\'t", "can not", txt)

        # general
        txt = re.sub(r"n\'t", " not", txt)
        txt = re.sub(r"\'re", " are", txt)
        txt = re.sub(r"\'s", " is", txt)
        txt = re.sub(r"\'d", " would", txt)
        txt = re.sub(r"\'ll", " will", txt)
        txt = re.sub(r"\'t", " not", txt)
        txt = re.sub(r"\'ve", " have", txt)
        txt = re.sub(r"\'m", " am", txt)
        
        return txt

    @staticmethod
    def e_mpalavra(palavra):
        return '-' in palavra or ' ' in palavra or '_' in palavra

    @staticmethod
    def remover_multipalavras(lista):
        return [e for e in lista if Util.e_mpalavra(e) == False]

    @staticmethod
    def print_log(msg):
        pass
        
    @staticmethod
    def carregar_cfgs(dir_configs):
        arq = open(dir_configs, 'r')
        obj = json.loads(arq.read())
        arq.close()
        
        Util.configs = obj

        return obj

    @staticmethod
    def limpar_arquivo(caminho_completo_arquivo):
        try:
            os.system('rm '+caminho_completo_arquivo)
        except: pass

    @staticmethod
    def cosseno(doc1, doc2, tipo='str'):
        if tipo == 'str':
            vec1 = Util.doc_para_vetor(doc1.lower())
            vec2 = Util.doc_para_vetor(doc2.lower())
        elif tipo == 'list':
            vec1 = Util.doc_para_vetor(" ".join(doc1).lower())
            vec2 = Util.doc_para_vetor(" ".join(doc2).lower())

        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if denominator:
            return float(numerator) / denominator
        else:
            return 0.0

    @staticmethod
    def jaccard(doc1, doc2, tipo='str'):
        if tipo == 'str':
            doc1 = set(doc1.split())
            doc2 = set(doc2.split())
        elif tipo == 'list':
            doc1 = set(" ".join(doc1).split())
            doc2 = set(" ".join(doc2).split())

        return float(len(doc1 & doc2)) / len(doc1 | doc2)

    @staticmethod
    def juntar_tokens(array):
        saida = ""

        for token in array:
            saida += re.sub('[_-]', ' ', token) + ' '

        return list(set(saida[:-1].split(' ')))

    @staticmethod
    def doc_para_vetor(text):
        WORD = re.compile(r'\w+')
        words = WORD.findall(text)
        words = text.split(' ')

        return Counter(words)

    @staticmethod
    def abrir_json(diretorio, criarsenaoexiste=False):
        if os.path.exists(diretorio) == False:
            if criarsenaoexiste == True:
                os.system('echo "{ }" > ' + diretorio)
                return { }
            elif criarsenaoexiste == False:
                return None
        else:
            arq = open(diretorio, 'r')
            try:
                obj = json.loads(arq.read())
            except:
                return None
            arq.close()
            return obj

        return None

    @staticmethod
    def deletar_arquivo(dir_arquivo):
        if os.path.exists(dir_arquivo):
            system("rm " + dir_arquivo)

    @staticmethod
    def limpar_diretorio_temporarios(configs):
        os.system('rm ' + configs['dir_temporarios'] + '/*')

    @staticmethod
    def limpar_diretorio(configs, diretorio):
        os.system('rm ' + diretorio + '/*')

    @staticmethod
    def obter_synsets(palavra, pos_semeval):
        if pos_semeval != 'a':
            return wordnet.synsets(palavra, pos_semeval)
        else:
            saida = [ ]

            for pos in ['a', 's']: # adjective e adjective sattelite
                saida += wordnet.synsets(palavra, pos)

            return saida

    # RETIRA AS PONDERACOES DE ACORDO COM A POS DESEJADA DA WORDNET
    @staticmethod
    def filtrar_ponderacoes(pos_semeval, ponderacoes):
        lista_pos = ['s', 'a'] if pos_semeval == 'a' else [pos_semeval]

        return [e for e in ponderacoes if e[0].pos() in lista_pos]

    @staticmethod
    def salvar_json(diretorio, obj):
        if type(obj) == set:
            obj = list(obj)
        if not type(diretorio) in [str, unicode]:
            raise TypeError("O diretorio (argumento 1) deve ser str!")
            
        try:
            arq = open(diretorio, 'w+')
            obj_serializado = json.dumps(obj, indent=4)
            arq.write(obj_serializado)
            arq.close()

            return True
        except Exception, e:
            os.system("rm " + diretorio)
            return False

    @staticmethod
    def print_formatado(mensagem, visivel=True):
        if Util.verbose_ativado and visivel:
            print(mensagem)

    # Recebe uma lista de listas e retorna-as concatenadas em uma unica
    # [[1], [2], [3,4]] => [1, 2, 3, 4]
    @staticmethod
    def mesclar_listas(lista):
        import itertools
        return list(itertools.chain.from_iterable(lista))

    @staticmethod
    def extrair_sins_cands_def(definicao, pos):
        #ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
        import itertools

        if not type(pos) in [str, unicode]:
            print('\n\nTipo POS: ' + str(type(pos)))
            traceback.print_stack()
            sys.exit(1)

        wn = wordnet

        if len(pos) > 1:
            pos = Util.conversor_pos_oxford_wn(pos)
        
        try:
            sw = stopwords.words('english')
            resultado_tmp = [ ]
            for (palavra, pos_tmp) in nltk.pos_tag(Util.tokenize(definicao.lower())):
                todos_lemas = list(itertools.chain(*[s.lemma_names() for s in wn.synsets(palavra, pos)]))
                if palavra in todos_lemas and False:
                    resultado_tmp.append((palavra, pos))
                elif not palavra in sw and pos_tmp[0].lower() == pos:
                    resultado_tmp.append((palavra, pos_tmp))
                elif not palavra in sw and pos_tmp[0].lower() == 'j' and pos in ['a', 's']:
                    resultado_tmp.append((palavra, pos_tmp))
        except: pass

        resultado = [ ]

        try:
            for l, pos_iter in resultado_tmp:
                if wn.synsets(l, pos_iter):
                    resultado.append(l)
        except:
            # retirando pontuacao
            tmp = [p[0] for p in resultado_tmp if len(p[0]) > 1]

            for l in tmp:
                try:
                    if wn.synsets(l, pos):
                        resultado.append(l)
                except: resultado.append(l)

        if not resultado:
            # retirando pontuacao
            tmp = [p[0] for p in resultado_tmp]
            resultado = [p for p in tmp if len(p) > 1]

        return resultado

    # Retorna todos arquivos da pasta. SOMENTE arquivos
    @staticmethod
    def list_arqs(dir_arqs, caminho_completo=True):
        if caminho_completo:
            return [f for f in os.listdir(dir_arqs) if os.path.isfile(os.path.join(dir_arqs, f))]
        else:
            return os.listdir(dir_arqs)

    @staticmethod
    def cls():
        system('clear')

    @staticmethod
    def retornar_valida(frase):
        frase = Util.completa_normalizacao(frase)
        frase = re.sub('[?!,;]', '', frase)
        frase = frase.replace("\'", " ")
        frase = frase.replace("-", " ")
        frase = frase.replace("\'", "")
        frase = frase.replace("\\`", "")
        frase = frase.replace("\"", "")
        frase = frase.replace("\n", " ")

        return frase.strip().lower()

    @staticmethod
    def gethostname():
        import socket
        try: return socket.gethostname()
        except: return ""

    @staticmethod
    def arq_existe(pasta, nome_arquivo):
        if pasta == None:
            # Aqui, supoe-se que <nome_arquivo> é fullpath do arquivo
            return os.path.isfile(nome_arquivo)
        if pasta[-1] != "/":
            pasta = pasta + "/"

        return os.path.isfile(pasta + nome_arquivo) 

    @staticmethod
    def normalizar_ctx(lista_ctx, stop=True, lematizar=True, stem=True):
        if stop:
            lista_ctx = [i for i in lista_ctx if i not in stopwords.words('english')]
        if lematizar:
            lista_ctx = [lemmatize(i) for i in lista_ctx]
        if stem:
            lista_ctx = [porter.stem(i) for i in lista_ctx]

        return lista_ctx

    @staticmethod
    def processar_ctx(ctx, stop=True, lematizar=True, stem=True, resolver_en=True):
        pass

    @staticmethod
    def tokenize(sentenca):        
        return [t for t in re.split('[!?.,;:\-_\s]', sentenca) if t != ""]
        #return nltk.word_tokenize(sentenca)

    @staticmethod
    def resolver_en(sentenca):
        return sentenca

    @staticmethod
    def exibir_json(obj, bloquear=False):
        try:
            if type(obj) == set: obj = list(obj)
            print(json.dumps(obj, indent=4))
        except Exception, e:
            print("@ Excecao na exibicao do objeto!")
            print(e)

        if bloquear: raw_input("\n\n<enter>")

    @staticmethod
    def retornar_valida(frase, lower=True, strip=True):
        frase = Util.completa_normalizacao(frase)
        frase = re.sub('[?!,;]', '', frase)
        frase = frase.replace("\'", " ")
        frase = frase.replace("-", " ")
        frase = frase.replace("\'", "")
        frase = frase.replace("\\`", "")
        frase = frase.replace("\"", "")
        frase = frase.replace("\n", " ")

        if lower: frase = frase.lower()
        if strip: frase = frase.strip()

        return frase

    @staticmethod
    def completa_normalizacao(cadeia, codif='utf-8'):
        if version_info[0] == 2:
            try:
                return normalize('NFKD', cadeia.decode(codif)).encode('ASCII','ignore')
            except: pass
        elif version_info[0] == 3:
            try:
                return normalize('NFKD', cadeia).encode('ASCII', 'ignore').decode('ASCII')
            except: pass

        return cadeia.encode('ASCII','ignore')

    @staticmethod
    def retornar_valida_pra_indexar(frase):
        frase = Util.completa_normalizacao(frase)
        frase = re.sub('['+string.punctuation+']', ' ', frase)        
        frase = ''.join(e for e in frase if (e.isalnum() and not e.isdigit()) or e == ' ')

        return frase.strip().lower()

    @staticmethod
    def obter_peso_frase(frase):
        frequencias = Util.obter_frequencias_frase(frase)
        soma = sum([e[1] for e in frequencias])

        return (soma / len(frequencias), frequencias)

    @staticmethod
    def is_stop_word(p):
        return p in stopwords.words('english')

    @staticmethod
    def ordenar_palavras(todas_palavras):
        dir_contadores = Util.configs['leipzig']['dir_contadores']

        if Util.contadores == None:
            contadores = Util.abrir_json(dir_contadores)
            Util.contadores = contadores
        else:
            contadores = Util.contadores

        palavras_indexadas = dict()
        palavras_ordenadas = [ ]
        
        for palavra in todas_palavras:
            try:
                if not contadores[palavra] in palavras_indexadas:
                    palavras_indexadas[contadores[palavra]] = [ ]
            except:
                palavras_indexadas[0] = [ ]

            try:
                palavras_indexadas[contadores[palavra]].append(palavra)
            except:
                palavras_indexadas[0].append(palavra)

        chaves = palavras_indexadas.keys()
        chaves.sort(reverse=True)

        for chave in chaves:
            palavras_ordenadas += list(set(palavras_indexadas[chave]))

        return palavras_ordenadas

    @staticmethod
    def conjugar(verbo):
        resultado=[ ]

        tense=['infinitive', 'present', 'past']
        person=[None, 1, 2, 3]
        number=[None, 'singular', 'plural']
        mood=[None, 'indicative']
        aspect=[None, 'imperfective', 'progressive']
        negated=[True, False]

        for t in tense:
            for p in person:
                for n in number:
                    for m in mood:
                        for a in aspect:
                            for neg in negated:
                                try:
                                    c=conjugate(verbo, tense=t, person=p,\
                                        number=n, mood=m, aspect=a, negated=neg)
                                    if c != None:
                                        resultado.append(c)
                                        resultado=list(set(resultado))
                                except:
                                    pass

        return resultado