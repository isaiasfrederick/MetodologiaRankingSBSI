from Utilitarios import Util

from OxAPI import BaseOx
from nltk.corpus import wordnet
from Utilitarios import Util

wn = wordnet

class Baselines(object):
    @staticmethod
    def predicao_usual(frase, palavra, pos, fonte='oxford'):
        conjunto = []

        if fonte == 'oxford':
            inst = BaseOx.INSTANCE

            for d in BaseOx.obter_definicoes(inst, palavra, pos):
                for s in BaseOx.obter_sins(inst, palavra, d, pos):
                    if s != palavra:
                        conjunto.append(s)

        elif fonte == 'wordnet':
            for s in wn.synsets(palavra, pos):
                for l in s.lemma_names():
                    if palavra != l:
                        if Util.e_mpalavra(l):
                            conjunto.append(l)

        return [p for p in conjunto if Util.e_mpalavra(p) == False]


class Word2Vec(object):
    @staticmethod
    def setup():
        pass

    @staticmethod
    def finalizar():
        pass

    @staticmethod
    def predicao_usual(frase, palavra, pos):
        pass

class Context2Vec(object):
    @staticmethod
    def setup():
        pass

    @staticmethod
    def finalizar():
        pass

    @staticmethod
    def predicao_usual(frase, palavra, pos):
        pass

