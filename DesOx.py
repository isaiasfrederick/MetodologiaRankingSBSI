#!coding: utf-8
from pywsd.utils import lemmatize, porter, lemmatize_sentence
from pywsd.cosine import cosine_similarity as cos_sim
from nltk.corpus import stopwords, wordnet
from collections import Counter
from textblob import TextBlob
from Utilitarios import Util
from nltk import pos_tag
import Indexador
from OxAPI import BaseOx
import itertools
import nltk
import inspect
import Alvaro
import sys
import re
import os


class DesOx(object):
    INSTANCE = None    
    cache_objs_json = { }

    def __init__(self, cfgs, base_ox, rep_vetorial=None):
        self.cfgs = cfgs
        self.base_ox = base_ox
        self.rep_conceitos = None

        self.rep_vetorial = rep_vetorial

        self.usar_cache = True
        self.dir_cache = cfgs['oxford']['cache']['desambiguador']
        self.tam_max_ngram = 4

    def assinatura_significado_aux(self, lema, pos, definicao, lista_exemplos):
        retornar_valida = Util.retornar_valida_pra_indexar
        assinatura = retornar_valida(definicao.replace('.', '')).lower()
        assinatura = [p for p in Util.tokenize(assinatura) if not p in [',', ';', '.']]
        if lista_exemplos:
            assinatura += list(itertools.chain(*[retornar_valida(ex).split() for ex in lista_exemplos]))
        assinatura += lema
        assinatura = [p for p in assinatura if len(p) > 1]
        return assinatura

    # "lematizar,True::nbest,True::stop,True::ctx,frase.::pos,r::usar_ontologia,False::
    # stem,True::usar_exemplos,True::busca_ampla,False"
    def construir_chave_cache(self, vars):
        vars = [",".join((unicode(k), unicode(v))) for k, v in vars.iteritems()]
        return "::".join(vars)

    # Metodo Cosseno feito para o dicionario de Oxford
    # Retorna dados ordenados de forma decrescente
    def desambiguar(self, ctx, ambigua, pos, nbest=True,\
                    lematizar=True, stem=True, stop=True,\
                    usar_ontologia=False, usr_ex=False,\
                    busca_ampla=False, med_sim='cosine', cands=[ ]):

        # Para gerar chave do CACHE
        vars_locais = dict(locals())

        dir_cache_tmp = None
        dir_bases = self.cfgs['caminho_bases']

        self.usar_cache = False

        if self.usar_cache:
            obj_dir_cache_des = dir_bases + '/' + self.cfgs['oxford']['cache']['desambiguador']

            del vars_locais['self']; del vars_locais['ambigua']

            k_vars = self.construir_chave_cache(vars_locais)

            dir_completo_obj = "%s/%s.json" % (obj_dir_cache_des, ambigua)
            Util.abrir_json(dir_completo_obj, criarsenaoexiste=True)

            if k_vars in obj_cache:
                return obj_cache[k_vars]

        if len(pos) == 1:
            pos = Util.cvrsr_pos_wn_oxford(pos)

        lem, st = (med_sim == 'cosine'), (med_sim == 'cosine')

        todas_assinaturas = [ ]

        try:
            todas_assinaturas = self.assinatura_significado(ambigua, usar_exemplos=usr_ex, lematizar=lem, stem=st)

            for candidato_iter in cands:
                todas_assinaturas += self.assinatura_significado(candidato_iter, usar_exemplos=usr_ex, lematizar=lem, stem=st)

            todas_assinaturas = [assi for assi in todas_assinaturas if pos == assi[0].split('.')[1]]
            # Tirando palavras de tamanho 1
            ctx = [p for p in Util.tokenize(ctx.lower()) if len(p) > 1]
            ctx = Util.normalizar_ctx(ctx, stop=stop, lematizar=lem, stem=st)
        except KeyboardInterrupt, ke:
            pass

        pontuacao = [ ]

        for assi in todas_assinaturas:
            ass_definicao = Util.normalizar_ctx(assi[3], stop=stop, lematizar=lematizar, stem=stem)

            label_def, desc_def, frase_def, ass_def = assi
            reg_def = (label_def, desc_def, frase_def)

            if med_sim == 'cosine':
                func_sim = cos_sim
            elif med_sim == 'word_move_distance':
                func_sim = self.rep_vetorial.word_move_distance

            dist_simi = func_sim(" ".join(ctx), " ".join(ass_definicao))

            # Colocando valor infinito
            if dist_simi == float('inf'):
                dist_simi = float(Util.MAX_WMD)

            if dist_simi == 'cosine' and dist_simi == 0.00:
                pass
            else:
                pontuacao.append((dist_simi, reg_def))            

        # Ordenacao da saida do desambiguador (cosine=decrescente, wmd=crescente)
        ordem_crescente = (med_sim == 'cosine')
        res_des = [(s, p) for p, s in sorted(pontuacao, reverse=ordem_crescente)]

        if self.usar_cache:
            obj_cache[k_vars] = res_des
            Util.salvar_json(dir_completo_obj, obj_cache)

        return res_des

    # DESAMBIGUA BASEADO EM FRASES DE EXEMPLO
    def des_exemplos(self, ctx,\
                    ambigua, pos, nbest=True,\
                    lematizar=True, stem=True, stop=True,\
                    normalizar_pont=True):

        cfgs = self.cfgs
        dir_bases = self.cfgs['caminho_bases']
        base_ox = self.base_ox
        
        rep_vet = self.rep_vetorial
        alvaro = Alvaro.Alvaro.INSTANCE

        dir_cache_rel_sinonimia = cfgs['caminho_bases']+'/'+cfgs['oxford']['cache']['sinonimia']
        chave_cache_relacao_sin = "%s-%s.json"%(ambigua, pos)
        dir_obj = dir_cache_rel_sinonimia+'/'+chave_cache_relacao_sin

        if not chave_cache_relacao_sin in Util.list_arqs(dir_cache_rel_sinonimia):
            rel_definicoes = alvaro.construir_relacao_definicoes(ambigua, pos, fontes='oxford')
            Util.salvar_json(dir_obj, rel_definicoes)
        else:            
            rel_definicoes = Util.abrir_json(dir_obj, criarsenaoexiste=False)

        res_des_tmp = [ ]
        pontuacao_somada = 0.00

        for def_ambigua in rel_definicoes:
            uniao_palavras_sem_duplicatas = set()
            uniao_palavras_com_duplicatas = list()
            exemplos_blob = [ ]

            palavras_tf = { }

            try:
                maximo_exemplos = self.cfgs['params_exps']['qtde_exemplos'][0]
                lista_exemplos = BaseOx.obter_atributo(ambigua, pos, def_ambigua, 'exemplos')
                # Adicionando lemas
                lista_exemplos.append(" ".join(BaseOx.obter_sins(ambigua, def_ambigua, pos)))
                # Adicionando definicao
                lista_exemplos.append(def_ambigua)

                for ex in lista_exemplos[:maximo_exemplos]:
                    ex_blob = TextBlob(ex)
                    exemplos_blob.append(ex_blob)
                    for token in ex_blob.words:
                        if Util.is_stop_word(token.lower()) == False:
                            token_lematizado = lemmatize(token)
                            uniao_palavras_sem_duplicatas.add(token_lematizado)
                            uniao_palavras_com_duplicatas.append(token_lematizado)

            except Exception, caminho:
                exemplos = [ ]

            textblob_vocab = TextBlob(" ".join(uniao_palavras_com_duplicatas))

            palavras_latentes = [ ]
            for p in textblob_vocab.word_counts:
                if textblob_vocab.word_counts[p] > 1:
                    palavras_latentes.append(p)

            palavras_derivadas = [ ]

            for p in uniao_palavras_sem_duplicatas:
                tf = alvaro.tf(p, textblob_vocab)
                palavras_tf[p] = tf

            pontuacao = 0.00

            for t in Util.tokenize(Util.resolver_en(ctx).lower()):
                try:
                    pontuacao += palavras_tf[t]
                except: pontuacao += 0.00

            pontuacao_somada += pontuacao

            try:
                if normalizar_pont:
                    reg_pont = pontuacao/sum(palavras_tf.values())
                else: reg_pont = pontuacao
            except ZeroDivisionError, zde: reg_pont = 0.00

            res_des_tmp.append(((ambigua, def_ambigua, [ ]), reg_pont))

        return sorted(res_des_tmp, key=lambda x: x[1], reverse=True)


    def desambiguar_exemplos(self,\
                    ctx,\
                    ambigua, pos,\
                    lematizar=True,\
                    stem=True,\
                    stop=True,\
                    normalizar_pont=True,\
                    profundidade=1,\
                    candidatos=[ ]):

        alvaro = Alvaro.Alvaro.INSTANCE
        todas_arvores = Alvaro.Alvaro.construir_arvores_definicoes(alvaro, ambigua, pos, profundidade, candidatos)
       
        caminhos_arvore = [ ]

        for arvore_sinonimia in todas_arvores:
            for caminho in arvore_sinonimia.percorrer():
                try:
                    cam_tmp = [tuple(reg.split(':::')) for reg in caminho.split("/")]
                    cam_tmp = [p for (p, def_p) in cam_tmp if p in candidatos or candidatos == [ ]]

                    conts_corretos = [1 for i in range(len(Counter(cam_tmp).values()))]

                    # Se todas palavras só ocorrem uma vez, entao nao existe ciclos
                    if Counter(cam_tmp).values() == conts_corretos:
                        if not caminho in caminhos_arvore:
                            caminhos_arvore.append(caminho)
                            
                except ValueError, ve:
                    pass

        cfgs = self.cfgs
        dir_bases = self.cfgs['caminho_bases']
        base_ox = self.base_ox
        
        rep_vet = self.rep_vetorial

        res_des_tmp = [ ]
        pontuacao_somada = 0.00

        #print("\nArvore para %s:"%str((ambigua, pos)))
        #Util.exibir_json(caminhos_arvore, bloquear=True)

        # Filtrar caminhos aqui
        for reg_caminhos in caminhos_arvore:
            uniao_palavras_sem_duplicatas = set()
            uniao_palavras_com_duplicatas = list()
            exemplos_blob = [ ]

            palavras_tf = { }

            try:
                maximo_exemplos = self.cfgs['params_exps']['qtde_exemplos'][0]

                lista_exemplos = [ ]
                definicoes_caminho = [tuple(r.split(":::")) for r in reg_caminhos.split("/")]
                
                # Percorrendo todos caminhos nas arvores de sinonimos
                for lema_caminho, def_caminho in definicoes_caminho:
                    try: # Adicionando caminho 
                        novos_ex = BaseOx.obter_atributo(BaseOx.INSTANCE,\
                                        lema_caminho, pos, def_caminho, 'exemplos')
                        novos_ex = list(novos_ex)                        
                        lista_exemplos += novos_ex
                    except: lista_exemplos = [ ]

                    # Adicionando lemas
                    sins_defcaminho = BaseOx.obter_sins(BaseOx.INSTANCE, lema_caminho, def_caminho, pos)

                    if sins_defcaminho:
                        lista_exemplos.append(" ".join(sins_defcaminho))

                    # Adicionando definicao
                    lista_exemplos.append(def_caminho)
                    
                for ex_iter in lista_exemplos[:maximo_exemplos]:
                    ex = ex_iter

                    ex_blob = TextBlob(ex)
                    exemplos_blob.append(ex_blob)

                    for token in ex_blob.words:
                        if Util.is_stop_word(token.lower()) == False:
                            token_lematizado = lemmatize(token)
                            uniao_palavras_sem_duplicatas.add(token_lematizado)
                            uniao_palavras_com_duplicatas.append(token_lematizado)

            except Exception, caminho:
                import traceback

                traceback.print_stack()
                traceback.print_exc()

                exemplos = [ ]

            tb_vocab_duplicatas = TextBlob(" ".join(uniao_palavras_com_duplicatas))

            for p in uniao_palavras_sem_duplicatas:
                tf = Alvaro.Alvaro.tf(alvaro, p, tb_vocab_duplicatas)
                palavras_tf[p] = tf

            pontuacao = 0.00

            for t in Util.tokenize(Util.resolver_en(ctx).lower()):
                try:
                    pontuacao += palavras_tf[t]
                except: pontuacao += 0.00

            pontuacao_somada += pontuacao

            try:
                if normalizar_pont:
                    reg_pont = pontuacao / sum(palavras_tf.values())
                else: reg_pont = pontuacao
            except ZeroDivisionError, zde:
                reg_pont = 0.00

            ambigua, def_ambigua = definicoes_caminho[0]
            novo_reg = ((ambigua, def_ambigua, [ ]), reg_pont)
            res_des_tmp.append(novo_reg)

        return sorted(res_des_tmp, key=lambda x: x[1], reverse=True)


    def desambiguar_adjetivos(self,\
                    ctx,\
                    ambigua, pos,\
                    lematizar=True,\
                    stem=True,\
                    stop=True,\
                    normalizar_pont=True,\
                    profundidade=1,\
                    candidatos=[ ]):
        if pos!='a': raise Exception("Esta pos nao é aceita!")

        ctx_blob = TextBlob(ctx)
        ngrams = { }
        ngrams_derivados = { }
        ngram_uniao = { }

        for n in range(3, 6):
            ngrams[n] = [ng for ng in ctx_blob.ngrams(n=n) if ambigua in ng]
            ngrams_derivados[n] = [ ]

            for c in candidatos:
                ngrams_derivados[n] += [" ".join(l).replace(ambigua, c).split(" ") for l in ngrams[n]]

        ngram_uniao = dict(ngrams)
        for n in ngrams_derivados:
            ngram_uniao[n] += ngrams_derivados[n]

        total_docs = set()

        cont_cands = dict([(c, 0) for c in candidatos])

        for n in ngram_uniao:
            for ngram_ in ngram_uniao[n]:
                ngram = list(ngram_)
                for doc in Indexador.Whoosh.consultar_documentos(ngram, "AND"):
                    if " ".join(ngram) in doc['content']:
                        try:
                            c = list(set(candidatos+[ambigua]).intersection(set(ngram)))[0]
                            cont_cands[c] += 1
                        except: pass

                        total_docs.add(doc['path'])
                        print(candidatos)
                        print(ngram)
                        print(ctx)
                        print(doc['content'])
                        print("\n")

        print(cont_cands)
        if 'cow disease' in ctx:
            raw_input("TOTAL DOCS: " + str(len(total_docs)))

        #Util.exibir_json(ngram_uniao, bloquear=True)
        #raw_input("\n\n<enter>")
        
        return [ ]



    def adapted_lesk(self, ctx, ambigua, pos, nbest=True,
                     lematizar=True, stem=True, stop=True,
                     usr_ex=False, janela=2):

        if len(pos) == 1:
            pos = Util.cvrsr_pos_wn_oxford(pos)

        limiar_polissemia = 10

        # Casamentos cartesianos das diferentes definicoes
        solucoes_candidatas = [ ]

        # Todas definicoes da palavra ambigua
        definicoes = [
            def_ox for def_ox in BaseOx.obter_definicoes(ambigua, pos)]

        ctx_blob = TextBlob(ctx)

        tags_validas = self.cfgs['pos_tags_treebank']
        tokens_validos = [(token, tag) for (token, tag)
                          in ctx_blob.tags if tag in tags_validas]

        tokens_validos_tmp = [ ]

        # [('The', 'DT'), ('titular', 'JJ'), ('threat', 'NN'), ('of', 'IN'), ...]
        for token, tag in tokens_validos:
            pos_ox = Util.cvrsr_pos_wn_oxford(tag[0].lower())

            defs_token = BaseOx.obter_definicoes(token, pos_ox)
            if not defs_token in [[ ], None]:
                tokens_validos_tmp.append((token, tag))
                solucoes_candidatas.append(defs_token)

        tokens_validos = tokens_validos_tmp
        tokens_validos_tmp = None

        indices_tokens_validos = [ ]

        if len(tokens_validos) != len(solucoes_candidatas):
            raise Exception("\nTAMANHOS DIFERENTES!\n")

        i = 0
        for tk, tag in list(tokens_validos):
            if tk == ambigua:
                cont = 0
                for e in sorted(range(0, i), reverse=True):
                    if len(solucoes_candidatas[e]) < limiar_polissemia:
                        indices_tokens_validos.append(e)
                        cont += 1
                    if cont == janela:
                        break

                indices_tokens_validos.append(i)

                cont = 0
                for d in range(i+1, len(tokens_validos)):
                    if len(solucoes_candidatas[d]) < limiar_polissemia:
                        indices_tokens_validos.append(d)
                        cont += 1
                    if cont == janela:
                        break

            i += 1

        tokens_validos = [tokens_validos[i] for i in indices_tokens_validos]

        print("\n")
        print("AMBIGUA: '%s'" % ambigua)
        print("CONTEXTO: '%s'\n" % ctx)
        print("TOKENS VALIDOS: "+str([(token, tag) for (token, tag) in tokens_validos]))
        prod = 1
        print("\n\n")
        print([len(solucoes_candidatas[i]) for i in indices_tokens_validos])
        for e in [solucoes_candidatas[i] for i in indices_tokens_validos]:
            prod *= len(e)
        print("Produtorio: "+str(prod))
        raw_input("\n")

        for dtmp in definicoes:
            d = str(dtmp).lower()

            todos_tamanhos_ngram = sorted(range(1, self.tam_max_ngram+1), reverse=True)

            for n in todos_tamanhos_ngram:
                ctx_blob = TextBlob(ctx.lower())
                todos_ngrams = ctx_blob.ngrams(n=n)

                for ngram in todos_ngrams:
                    ngram_str = " ".join(ngram)
                    freq = d.count(ngram_str)
                    pontuacao = freq*(n**2)

                    if freq:
                        d = d.replace(ngram_str, '')

        return 0.00

    """ Gera uma assinatura de um significado Oxford para aplicar Cosseno """
    def assinatura_significado(self, lema, lematizar=True,\
                            stem=False, stop=True,\
                            extrair_relacao_semantica=False,\
                            usar_exemplos=False):

        resultado = BaseOx.construir_objeto_unificado(self.base_ox, lema)

        if not resultado:
            resultado = { }

        lema = lemmatize(lema)

        assinaturas_significados = [ ]  # (nome, definicao, exemplos)

        for pos in resultado.keys():
            todos_significados = resultado[pos].keys()

            indice = 1
            for sig in todos_significados:
                nome_sig = "%s.%s.%d" % (lema, pos, indice)
                indice += 1

                if usar_exemplos:
                    exemplos = resultado[pos][sig]['exemplos']
                else:
                    exemplos = [ ]

                # nome, definicao, exemplos, assinatura
                definicao_corrente = [nome_sig, sig, exemplos, [ ]]
                assinaturas_significados.append(definicao_corrente)

                # Colocando exemplos na assinatura
                definicao_corrente[len(
                    definicao_corrente)-1] += self.assinatura_significado_aux(lema, pos, sig, exemplos)

                sig_secundarios = resultado[pos][sig]['def_secs']

                for ss in sig_secundarios:
                    nome_sig_sec = "%s.%s.%d" % (lema, pos, indice)

                    if usar_exemplos:
                        exemplos_secundarios = resultado[pos][sig]['def_secs'][ss]['exemplos']
                    else:
                        exemplos_secundarios = [ ]

                    definicao_corrente_sec = [
                        nome_sig_sec, ss, exemplos_secundarios, [ ]]
                    assinaturas_significados.append(definicao_corrente_sec)

                    definicao_corrente_sec[len(
                        definicao_corrente)-1] += self.assinatura_significado_aux(lema, pos, ss, exemplos_secundarios)

                    indice += 1

        for sig in assinaturas_significados:
            sig[3] = Util.normalizar_ctx(sig[3], stop=True, lematizar=True, stem=True)

        return [tuple(a) for a in assinaturas_significados]

    def retornar_valida(self, frase):
        return Util.retornar_valida(frase)

    def extrair_sinonimos(self, ctx, palavra, pos=None,
                          usar_exemplos=False, busca_ampla=False,
                          repetir=False, coletar_todos=True):

        max_sinonimos = 10

        obter_objeto_unificado_oxford = BaseOx.construir_objeto_unificado
        obter_sinonimos_oxford = BaseOx.obter_sinonimos_fonte_obj_unificado

        try:
            resultado = self.desambiguar(ctx, palavra, pos, usr_ex=usar_exemplos, busca_ampla=busca_ampla)
        except Exception, e:
            resultado = [ ]

        sinonimos = [ ]

        try:
            if resultado[0][1] == 0:
                resultado = [resultado[0]]
                repetir = False
            else:
                resultado = [item for item in resultado if item[1] > 0]
        except:
            resultado = [ ]

        continuar = bool(resultado)

        while len(sinonimos) < max_sinonimos and continuar:
            len_sinonimos = len(sinonimos)

            for item in resultado:
                definicao, pontuacao = item[0][1], item[1]

                if len(sinonimos) < max_sinonimos:
                    try:
                        obj_unificado = obter_objeto_unificado_oxford(palavra)

                        sinonimos_tmp = obter_sinonimos_oxford(
                            pos, definicao, obj_unificado)
                        sinonimos_tmp = [
                            s for s in sinonimos_tmp if not Util.e_mpalavra(s)]
                        sinonimos_tmp = list(
                            set(sinonimos_tmp) - set(sinonimos))

                        if coletar_todos:
                            sinonimos += sinonimos_tmp
                        elif sinonimos_tmp:
                            sinonimos += [sinonimos_tmp[0]]

                    except:
                        pass
                else:
                    continuar = False

            if repetir == False:
                continuar = False
            elif len_sinonimos == len(sinonimos):
                continuar = False

        return sinonimos[:max_sinonimos]
