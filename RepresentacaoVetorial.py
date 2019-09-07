#! coding: utf-8

import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from Utilitarios import Util
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from sys import argv
import re
import io

import nltk

# Para temporizar a medida WMD
import signal

wn = wordnet

# Esta classe trabalha com Wordembeddings para realizar a tarefa de prediçao de sinonimos
class RepVetorial(object):
    INSTANCE = None
    MODELO_CARREGADO = False

    STOP_WORDS = nltk.corpus.stopwords.words('english')

    def __init__(self, configs, diretorio_modelo=None, binario=True):
        self.modelo = None
        self.cfgs = configs
        self.stemmer = PorterStemmer()

        if self.modelo:
            self.modelo.init_sims(replace=True)  # normalizar vetores

        if diretorio_modelo:
            self.carregar_modelo(diretorio_modelo, binario)

    # Distancia de cosseno entre embeddings de duas palavras
    def cosseno(self, w1, w2):
        return self.modelo.similarity(w1, w2)

    def sinal_timeout_wmd(self, signum, frame):
        raise Exception("Timed out!")

    def word_move_distance(self, doc1_str, doc2_str):
        signal.signal(signal.SIGALRM, self.sinal_timeout_wmd)
        signal.alarm(100)   # 100 segundos

        stopwords = RepVetorial.STOP_WORDS

        try:            
            doc1_str, doc2_str = doc1_str.split(" "),  doc2_str.split(" ")

            doc1_str = [p for p in doc1_str if not p in stopwords]
            doc2_str = [p for p in doc2_str if not p in stopwords]

            valor = self.modelo.wmdistance(doc1_str, doc2_str)
            signal.alarm(0)
            return valor
        except Exception, e:
            try: signal.alarm(0)
            except: pass
            return Util.MAX_WMD

    def relacionadas(self, palavra, pos, topn=1):
        return self.obter_palavras_relacionadas(positivos=[palavra], pos=pos, topn=topn)

    def obter_palavras_relacionadas_vetor(self, vetor, pos=None, topn=1, remover_score=False):
        res = self.modelo.similar_by_vector(vetor, topn=topn)

        return [(p, s) for (p, s) in res if wn.synsets(p, pos)][:topn]

    def obter_palavras_relacionadas(self, positivos=None, negativos=None, pos=None, topn=1, remover_score=False):
        retorno = []

        if topn == 0: topn = 1

        try:
            if positivos in ["", [ ]]: positivos = None
            if negativos in ["", [ ]]: negativos = None

            if positivos != None:
                positivos = [p for p in positivos if p in self.modelo]
            if negativos != None:
                negativos = [p for p in negativos if p in self.modelo]

            res = self.modelo.most_similar_cosmul(positive=positivos, topn=topn)            

            if pos != None:
                retorno = [(palavra, score) for (palavra, score) in res if wn.synsets(palavra, pos) != [ ]]
            else:
                retorno = [(palavra, score) for palavra, score in res]
        except KeyError, ke:
            return [ ]

        if remover_score:
            return [palavra for (palavra, score) in retorno]
        else:
            return retorno

    def iniciar_processo(self, palavra_arg, pos_semeval, contexto, topn=10):
        if self.modelo == None:
            raw_input("Nao ha modelo carregado!")

        todos_conjuntos_synsets = [ ]

        registros = self.obter_palavras_relacionadas(
            positivos=[palavra_arg], topn=topn)
        saida = [ ]

        for palavra_flexionada, pontuacao in registros:
            # palavra = self.stemmer.stem(palavra_flexionada)
            palavra = palavra_flexionada
            synsets = Util.obter_synsets(palavra, pos_semeval)

            if not synsets in todos_conjuntos_synsets:
                if synsets.__len__() > 0:
                    lema_synset = synsets[0].name().split('.')[0]
                    saida.append((lema_synset, pontuacao))
                todos_conjuntos_synsets.append(synsets)

        return [r[0] for r in saida]

    def palavra_diferente(self, lista_palavras):
        if type(lista_palavras) == str:
            lista_palavras = lista_palavras.split(" ")

        return self.modelo.doesnt_match(lista_palavras)

    # Retirado de https://fasttext.cc/docs/en/english-vectors.html
    def load_vectors(self, fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = { }
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return data

    # Retirado de https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    def carregar_modelo(self, diretorio, binario=True):
        try:
            if binario == False:
                arq_glove = diretorio

                dir_arquivo_tmp = diretorio.split('/').pop()
                arq_tmp = "./Bases/" + self.cfgs['dir_temporarios'] + \
                    '/' + dir_arquivo_tmp + '.tmp'

                print("\n")
                print("\tConvertendo embeddings W2C para C2V...")
                glove2word2vec(arq_glove, arq_tmp)
                print("\tConvertendo embeddings W2C para C2V: feito.")
                print("\tConvertendo embeddings W2C para C2V: carregando...")
                self.modelo = KeyedVectors.load_word2vec_format(arq_tmp)
                print("\tConvertendo embeddings W2C para C2V: carregando!")
                print("\n")

            else:
                self.modelo = KeyedVectors.load_word2vec_format(
                    diretorio, binary=True)
        except Exception, e:
            # Carregando vetores fasttext
            self.modelo = self.load_vectors(diretorio)
    
        RepVetorial.MODELO_CARREGADO = True

    @staticmethod
    def bow_embeddings_definicao(palavra, pos):
        from OxAPI import BaseOx
        from Alvaro import Alvaro

        descricao_definicoes = {  }
        uniao_definicoes = set()

        for d in BaseOx.obter_definicoes(BaseOx.INSTANCE, palavra, pos=pos):
            d_sins = BaseOx.obter_sins(BaseOx.INSTANCE, palavra, d, pos=pos)

            correlatas = [ ]
            for s in d_sins:
                similares_tmp = [reg[0] for reg in Alvaro.palavras_similares(s, pos)]
                uniao_definicoes.update(similares_tmp)
                correlatas.append(similares_tmp)

            # Interseccao entre palavras da mesma definicao
            interseccao = set(correlatas[0])
            for c in correlatas[:1]:
                interseccao = set(interseccao)&set(c)

            descricao_definicoes[d] = interseccao

        #for d in descricao_definicoes:
        #    descricao_definicoes[d] = list(set(descricao_definicoes[d]) - set(uniao_definicoes))
        descricao_definicoes_tmp = {  }
        for d in descricao_definicoes:
            uniao_outros = set()
            for outro in descricao_definicoes:
                if outro != d: uniao_outros.update(descricao_definicoes[outro])
            descricao_definicoes_tmp[d] = set(descricao_definicoes[d]) - uniao_outros
            descricao_definicoes_tmp[d] = list(descricao_definicoes_tmp[d])
            uniao_outros = None

        return descricao_definicoes_tmp




class WordnetEmbeddings(object):
    def __init__():
        pass

    @staticmethod
    def representar_synset(nome_synset):
        from nltk.corpus import wordnet

        pass

    @staticmethod
    def representacao_definicao(definicao):
        import RepresentacaoVetorial
        tokens = nltk.word_tokenize(definicao.lower())

        rep_vet_inst = RepresentacaoVetorial.RepVetorial.INSTANCE

        resultados = { }

        for t in tokens:
            res = RepresentacaoVetorial.RepVetorial.obter_palavras_relacionadas(rep_vet_inst, positivos=[t], topn=40)

            print(t)
            for r in res:
                print("\t" + str(r))

            print("\n")


