import os.path
import string
import traceback

import whoosh.index
from unidecode import unidecode
from whoosh.fields import *
from whoosh.index import create_in
from whoosh.qparser import QueryParser

from Utilitarios import Util

import json
import re


class Whoosh(object):
    DIR_INDEXES = "./Bases/Corpora/indexes"
    DIR_INDEXES_EXEMPLOS = "./Bases/Corpora/indexes-exemplos"

    SCHEMA_CORPORA = Schema(title=TEXT(stored=True), path=ID(
        stored=True), content=TEXT(stored=True))
    SCHEMA_EXEMPLOS = Schema(title=TEXT(stored=True), path=ID(
        stored=True), content=TEXT(stored=True))

    # Gera a frequencia de uma palavra no Corpora utilizado
    @staticmethod
    def count(keys, indexes):
        if type(keys) == str:
            keys = [keys]

        obter_docs = Whoosh.consultar_documentos
        docs_corpora_cand_par = obter_docs(
            keys, operador="AND", dir_indexes=indexes)
        freq = len(docs_corpora_cand_par)
        docs_corpora_cand_par = None

        return freq

    @staticmethod
    def remover_docs(docnums):
        ix = whoosh.index.open_dir(Whoosh.DIR_INDEXES)
        writer = ix.writer()

        for docnum in docnums:
            writer.delete_document(docnum, delete=True)

        writer.commit()

    @staticmethod
    def searcher(indexes):
        ix = whoosh.index.open_dir(indexes)
        return ix.searcher()

    @staticmethod
    def buscar_docnum_bypath(path):
        ix = whoosh.index.open_dir(Whoosh.DIR_INDEXES)
        return ix.searcher().document_number(path=unicode(path))

    @staticmethod
    def deletar_bypattern(campo, pattern):
        q = whoosh.query.Wildcard(campo, pattern)
        ix = whoosh.index.open_dir(Whoosh.DIR_INDEXES)
        r = ix.delete_by_query(q)
        ix.writer().commit()
        return r

    @staticmethod
    def buscar_padrao(campo, pattern, dir_indexes):
        q = whoosh.query.Wildcard(campo, pattern)
        ix = whoosh.index.open_dir(dir_indexes)

        res = ix.searcher().search(q)
        return res

    @staticmethod
    def documentos(indexes):
        ix = whoosh.index.open_dir(indexes)
        return ix.searcher().documents()

    @staticmethod
    def indexar_definicoes_palavra_ox(palavra):
        from OxAPI import BaseOx

        documentos = []
        numero = 1

        for d, pos in BaseOx.obter_definicoes(BaseOx.INSTANCE, palavra, retornar_pos=True):
            exemplos = BaseOx.obter_atributo(
                BaseOx.INSTANCE, palavra, None, d, 'exemplos')
            exemplos = ":::".join(exemplos)

            path = '%s-%s.json-%d' % (palavra, pos[0].lower(), numero)
            reg = (palavra + ":::" + d, path, exemplos)
            documentos.append(reg)

            numero += 1

        Whoosh.iniciar_indexacao_exemplos(documentos)
        return documentos

    @staticmethod
    def iniciar_indexacao(dir_lista_arquivos):
        if not os.path.exists(Whoosh.DIR_INDEXES):
            os.mkdir(Whoosh.DIR_INDEXES)
            indexer = create_in(Whoosh.DIR_INDEXES, Whoosh.SCHEMA_CORPORA)
        else:
            indexer = whoosh.index.open_dir(Whoosh.DIR_INDEXES)

        writer = indexer.writer()

        arquivo_lista = open(dir_lista_arquivos, 'r')
        todos_arquivos = [e.replace('\n', '')
                          for e in arquivo_lista.readlines()]
        arquivo_lista.close()

        indice_arquivo = 1
        for arquivo in todos_arquivos:
            indice_linha = 1
            with open(arquivo) as arq:
                for linha_arq in arq:
                    try:
                                        #conteudo = unicode(str(linha_arq).decode('utf-8'))
                                        #conteudo = re.sub(r'[^\x00-\x7F]+',' ', conteudo)
                        conteudo = str(linha_arq)
                        conteudo = "".join(
                            [i if ord(i) < 128 else " " for i in conteudo])
                        nome_arquivo = arquivo+'-'+str(indice_linha)

                        title = unicode(nome_arquivo)
                        path = unicode(nome_arquivo)
                        content = unicode(conteudo)

                        writer.add_document(
                            title=title, path=path, content=content)
                    except Exception, e:
                        print("\n")
                        traceback.print_exc()
                        print("\n")

                    print('\tArquivo %d - Linha %d' %
                          (indice_arquivo, indice_linha))
                    indice_linha += 1
            indice_arquivo += 1

        print('Realizando commit...')
        writer.commit()
        print('Commit realizado...')

    @staticmethod
    def iniciar_indexacao_signalmedia(dir_lista_arquivos, arquivos_duplicados):
        arquivos_duplicados = Util.abrir_json(
            arquivos_duplicados, criarsenaoexiste=False)
        arquivos_duplicados_set = set()

        for reg_str in arquivos_duplicados:
            path, _id_ = eval(reg_str)
            arquivos_duplicados_set.add(_id_)

        Whoosh.DIR_INDEXES = raw_input("Diretorio indexes: ")

        if not os.path.exists(Whoosh.DIR_INDEXES):
            os.mkdir(Whoosh.DIR_INDEXES)
            indexer = create_in(Whoosh.DIR_INDEXES, Whoosh.SCHEMA_CORPORA)
        else:
            indexer = whoosh.index.open_dir(Whoosh.DIR_INDEXES)

        writer = indexer.writer()

        arquivo_lista = open(dir_lista_arquivos, 'r')
        todos_arquivos = [e.replace('\n', '')
                          for e in arquivo_lista.readlines()]
        arquivo_lista.close()

        indice_arquivo = 1
        for _arquivo_ in todos_arquivos:
            if writer.is_closed:
                writer = None
                writer = indexer.writer()

            indice_linha = 1
            arquivo = "./Bases/Corpora/SignalMedia/" + _arquivo_

            print('\tArquivo %d' % (indice_arquivo))

            with open(arquivo) as _arq_:
                for linha_tmp in _arq_:
                    try:
                        obj_json = json.loads(linha_tmp)

                        if not obj_json["id"] in arquivos_duplicados_set:
                            titulo_doc_sm = obj_json["title"]

                            arq = obj_json["content"]

                            arq.replace(". \n\n", ". ")
                            arq.replace("? \n\n", "? ")
                            arq.replace("! \n\n", "! ")

                            arq.replace(". \n \n", ". ")
                            arq.replace("? \n \n", "? ")
                            arq.replace("! \n \n", "! ")

                            arq = re.split('[.?!]', arq)
                            arq = [l for l in arq if l.strip() != '']

                        else:
                            arq = []

                    except:
                        arq = []

                    for linha_arq in arq:
                        try:
                            #conteudo = unicode(str(linha_arq).decode('utf-8'))
                            #conteudo = re.sub(r'[^\x00-\x7F]+',' ', conteudo)
                            conteudo = str(linha_arq)
                            conteudo = "".join(
                                [i if ord(i) < 128 else " " for i in conteudo])

                            import random
                            nome_arquivo = arquivo + '-' + \
                                str(Util.md5sum_string(conteudo))

                            title = unicode(nome_arquivo)
                            path = unicode(nome_arquivo)
                            content = unicode(conteudo)

                            try:
                                if content.strip(" ") != "":
                                    if writer.is_closed:
                                        writer = indexer.writer()
                                        if writer.is_closed == False:
                                            raw_input("Deu certo!")
                                        else:
                                            raw_input("Deu errado!")

                                writer.add_document(
                                    title=title, path=path, content=content)
                            except Exception, e:
                                print("Content: " + str((content, title, path)))
                                import traceback
                                traceback.print_stack()
                                raw_input("\nExcecao aqui: %s\n" % str(e))

                        except Exception, e:
                            import traceback
                            traceback.print_stack()
                            print("\nException: " + str(e) + "\n")

                       #"""print('\tArquivo %d - Linha %d' % (indice_arquivo, indice_linha))"""
                        indice_linha += 1
            indice_arquivo += 1

            print('Realizando commit. Vai gastar um bom tempo!')
            try:
                if writer.is_closed:
                    raw_input("\nWriter estava fechado na hora do commit!\n")
                writer.commit()
                print("@@@ " + _arquivo_)
                print('Commit realizado...')
            except Exception, e:
                import traceback
                print("@@@ " + _arquivo_)
                print("\nException: " + str(e) + "\n")
                print('Commit NAO realizado...')
                traceback.print_stack()
                print("===============================")

    """
        Indexa exemplos das definicoes.
        Recebe uma lista de registros <palavra:::def; path; exemplos juntos por :::>
    """
    @staticmethod
    def iniciar_indexacao_exemplos(documentos):
        if not os.path.exists(Whoosh.DIR_INDEXES_EXEMPLOS):
            os.mkdir(Whoosh.DIR_INDEXES_EXEMPLOS)
            indexer = create_in(Whoosh.DIR_INDEXES_EXEMPLOS,
                                Whoosh.SCHEMA_EXEMPLOS)
        else:
            indexer = whoosh.index.open_dir(Whoosh.DIR_INDEXES_EXEMPLOS)

        writer = indexer.writer()

        cont_doc = 1
        print("\n")

        for palavra_def, path_arquivo, conteudo_iter in documentos:
            try:

                        #conteudo = unicode(str(linha_arq).decode('utf-8'))
                        #conteudo = re.sub(r'[^\x00-\x7F]+',' ', conteudo)

                conteudo = unicode(conteudo_iter)
                conteudo = "".join(
                    [i if ord(i) < 128 else " " for i in conteudo])

                title = palavra_def
                path = unicode(path_arquivo)
                content = unicode(conteudo)

                writer.add_document(title=title, path=path +
                                    '-' + str(cont_doc), content=content)
                cont_doc += 1
                print("\tIndexando exemplos da palavra '%s'" % path_arquivo)

            except Exception, e:
                print("\n")
                traceback.print_exc()
                print("\n")

        print("\n")
        print('\tRealizando commit de exemplos...')

        try:
            writer.commit()
            print('\tCommit de exemplos realizado...')
        except Exception, e:
            print('\tCommit de exemplos NAO realizado...')
            print('\tExcecoes: ' + str(e))

    @staticmethod
    def consultar_documentos(lista_palavras, operador="AND", limite=None, dir_indexes=None):
        operador = operador.upper()

        if type(lista_palavras) in [str, unicode]:
            lista_palavras = [lista_palavras]
        elif type(lista_palavras) == tuple:
            lista_palavras = list(lista_palavras)

        if type(lista_palavras) != list:
            raise Exception("Indexador deve receber uma lista, tupla ou string!")

        indexes = whoosh.index.open_dir(dir_indexes)
        searcher = indexes.searcher()
        parser = QueryParser("content", indexes.schema)

        consultar = ""
        operador = " " + operador + " "

        for arg in lista_palavras:
            consultar += arg+operador

        consultar = parser.parse(consultar[:-len(operador)])
        res_busca = searcher.search(consultar, limit=limite)

        resultado = [doc for doc in res_busca]

        return resultado

    @staticmethod
    def buscar_docnum(path):
        indexes = whoosh.index.open_dir(Whoosh.DIR_INDEXES)
        searcher = indexes.searcher()

        return searcher.document_number(path=unicode(path))

    @staticmethod
    def get_regex():
        rgx1 = u"\.|\,|\;|\s|\"|\-|\?|\!|\:|\t|\`|\_|\\(|\\)|\\[|\\]|@|\*"
        rgx2 = u"\u2019|\u201d|\u201f|\u2013|\u2014|\u2018"
        rgx3 = u"\u00e2|\u20ac|\u2122|\u0080"

        return rgx1+'|'+rgx2+'|'+rgx3

    @staticmethod
    def limpar(string):
        regex = get_regex()
        return ' '.join(re.split(regex, string))
