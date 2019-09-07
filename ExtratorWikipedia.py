import requests
import lxml.html
import re

from nltk.corpus import wordnet
from Utilitarios import Util
import nltk

from OxAPI import BaseOx

class ExtratorWikipedia(object):
    INSTANCE = None
    
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.prefixo_url = ""

    def obter_pagina(self, url, codigo=False):
        page = requests.get(url)        
        tree = lxml.html.fromstring(page.content)

        if codigo == False:
            return tree
        else:
            return tree, page.status_code

    @staticmethod
    def completa_normalizacao(cadeia, codif='utf-8'):
        return Util.completa_normalizacao(cadeia, codif=codif)

    def buscar_tabela(self):
        url = ""
        tree = self.obter_pagina(url)
        xpath_lista_topo = '//*[@id="mw-content-text"]/div/dl[2]/dd/dl/dd/table'

        try:
            requests.session().cookies.clear()
        except: pass

        return tree.xpath(xpath_lista_topo)

    def retornar_lista(self, tabela):
        return [self.prefixo_url + t.get('href') for t in tabela.findall("tr/td[2]/a")]

    def obter_texto(self, url, obter_referencias=False):
        caminho_xpath_conteudo_principal = '//*[@id="mw-content-text"]/div/p'

        tree = self.obter_pagina(url)
        elemento = tree.xpath(caminho_xpath_conteudo_principal)

        #print("\nURL: %s\n"%url)

        texto = [e.text_content() for e in elemento]

        texto = re.sub("(\[[a-zA-Z0-9]+\])", "", "".join(texto)).encode('utf-8')
        texto = re.sub("(\s[A-Z]\.\s)", " ", texto)
        texto = re.sub("(\[[a-z\s]+\])", "", texto)
        texto = re.sub("(\s[A-Z]\.)", "", texto)
        texto = re.sub("(\s[Ss][t]\.)", "Saint", texto)

        regex = "(?=([\d]*))(\.)(?=([\d]+\s*))"
        texto = re.sub(regex, ",", texto)

        #print('Documento %s coletado!' % url)

        if obter_referencias:
            xpath_ref = '//*[@class="references"]/li/span/cite/a/@href'

            try:
                links_refs = [e for e in tree.xpath(xpath_ref) if not '#' in e]
                links_refs = [e for e in links_refs if not 'wiki' in e]
                links_refs = [e.replace(' ', '_') for e in links_refs if not 'dictionar' in e.lower()]
            except:
                links_refs = [ ]

            return texto, links_refs

        return texto, [ ]

    # Obtem todas as URLs da desambiguacao de um verbete da Wikipedia
    def obter_links_desambiguacao(self, url):
        url_base = "https://en.wikipedia.org/wiki/"
        tree = self.obter_pagina(url)

        # Xpath para retornar todas as URLs de uma pagina de desambiguacao
        xpath = '//*[@id="mw-content-text"]/div/ul/li/a[@href]'
        xpath_primario = '//*[@id="mw-content-text"]/div/p//a[@href]'

        secundarios = [url_base + e.text_content().replace(' ', '_') for e in tree.xpath(xpath)]
        primario = [url_base + e.text_content().replace(' ', '_') for e in tree.xpath(xpath_primario)]

        return primario + secundarios

    def url_entidade_nomeada(self, url):
        url = url.split("/")[-1]
        cont_capitalizados = 0
        for t in re.split("[_()]", url):
            if len(t):
                if t[0].isupper(): cont_capitalizados += 1
        return cont_capitalizados > 1

    def obter_links_relevantes_desambiguacao(self, url, palavra):
        if not palavra:
            return None

        tree, codigo = self.obter_pagina(url, codigo=True)

        if codigo == 404:
            return [ ]

        url_base = "https://en.wikipedia.org"
        palavra = palavra.lower()

        args = (palavra.upper(), palavra.lower(), palavra)

        xpath = '//*[@id="mw-content-text"]/div//a/@href'

        links_tmp = [e for e in tree.xpath(xpath) if not '#' in e]
        links_tmp = [e for e in links_tmp if '/wiki' in e]
        links_tmp = [e for e in links_tmp if not 'disambiguation' in e.lower()]
        links_tmp = [e for e in links_tmp if not 'wiktionar' in e.lower()]
        links_tmp = [e for e in links_tmp if not self.url_entidade_nomeada(e)]
        links_tmp = [e.replace(' ', '_') for e in links_tmp if not '#' in e]

        links_tmp = [url_base + e if not 'http' in e else e for e in links_tmp]
        links = [  ]

        palavras_exclusao = ['.svg', 'File:', 'intitle:', 'Special:']
        links_excluidos = [ ]

        # Removendo entidades nomeadas
        en = [
            "(band)",
            "(film)",
            "(_film)",
            "_film",
            "(novel)",
            "(song)",
            "(company)",
            "(magazine)",
            "(surname)",
            "(perfume)",
            "(album)",
            "(season_",
            "_(name)",
            ".svg",
            "File:",
            "Special:",
            "Wikipedia:",
            "film_series"
        ]
      
        r = [l for l in links_tmp if not any(ss in l for ss in en)]
        resultado = [ ]

        for l in r:
            try:
                if not l.split("/")[-1][1].isupper():
                    resultado.append(l)
            except: pass

        return resultado

    def extrair_frases_pagina_comum(self, url, palavra):
        xpath = '/html/body'
    
        try:
            tree = self.obter_pagina(url)
            
            text_content = tree.xpath(xpath)[0].text_content()
            text_content = text_content.replace("\t", " ")
            text_content = text_content.replace("\n", " ")
            text_content = unicode(' '.join(text_content.split()))

            text_content = ExtratorWikipedia.completa_normalizacao(text_content)
            resultados = [ ]

            todas_frases = re.split("[.!?]", text_content)

            for f in [f.strip(' ') for f in todas_frases]:
                nova_frase = ExtratorWikipedia.filtrar_frase_bruta(f, palavra)
                if nova_frase != None:
                    resultados.append(nova_frase)

        except Exception, e:
            return [ ]

        return resultados

    @staticmethod
    def filtrar_frase_bruta(frase, palavra):
        tem_verbo = False
        total_substantivos = 0
        tem_palavra = False

        strings_invalidas = set("<>{ }")

        if set(frase).intersection(strings_invalidas) or '":"' in frase:
            return None

        if len(frase) > 200:
            return None

        tagueados = nltk.pos_tag(nltk.word_tokenize(frase))

        try:
            if tagueados[0][1] == 'CD': tagueados = tagueados[1:]
        except: return None

        for t, pt in tagueados:
            if pt[0].upper() == 'V': tem_verbo = True
            if pt[0].upper() == 'N': total_substantivos += 1

            if palavra.lower() in t.lower():
                s1 = wordnet.synsets(palavra.lower())
                s2 = wordnet.synsets(t.lower())

                if set(s1).intersection(s2):
                    tem_palavra = True

            if tem_verbo and total_substantivos > 1 and tem_palavra:
                return " ".join([t for (t, pt) in tagueados])

        if tem_verbo and total_substantivos > 1 and tem_palavra:
            return " ".join([t for (t, pt) in tagueados])
        else:
            return None

    @staticmethod
    def obter_frases_exemplo(palavra):
        ext = ExtratorWikipedia(None)

        url = Util.CONFIGS['wikipedia']['url_desambiguacao']%palavra.capitalize()

        duplicatas = set()

        frases_desambiguadads = { }
        todas_definicoes = { }

        instance = BaseOx.INSTANCE

        links_relevantes_tmp = ext.obter_links_relevantes_desambiguacao(url, palavra)

        pagina_desambiguacao = not links_relevantes_tmp in [None, [ ], { }]

        if pagina_desambiguacao == False:
            url = Util.CONFIGS['wikipedia']['url_base_verbete']+'/'+palavra.lower()
            links_relevantes_tmp = [url]

        links_relevantes_tmp = [l for l in links_relevantes_tmp if not "dictiona" in l]
        links_relevantes = [ ]

        todos_sins = [ ]

        lemas = {  }
        url_lema = {  }

        for def_iter in BaseOx.obter_definicoes(instance, palavra, pos=None):
            conj_sins = BaseOx.obter_sins(instance, palavra, def_iter, pos=None)
            todos_sins += conj_sins

        todos_sins = [palavra] + list(set(todos_sins))

        for l in links_relevantes_tmp:
            for s in todos_sins:
                if s.lower() in l.lower() and not l in links_relevantes:
                    links_relevantes.append(l)
                    if not s.lower() in lemas: lemas[s.lower()] = [ ]
                    lemas[s.lower()].append(l)
                    url_lema[l] = s.lower()

        links_relevantes_tmp = None

        registros = [ ]

        for url in links_relevantes:
            if ".wikipedia." in url:
                texto_wikipedia, todas_urls_referencias = ext.obter_texto(url, obter_referencias=True)
                plain_text = Util.completa_normalizacao(texto_wikipedia).lower()

                if set(nltk.word_tokenize(plain_text)).intersection(set(todos_sins)):
                    todas_sentencas = re.split('[.?!]', plain_text)
                    descricao = todas_sentencas[0].replace('\n', ' ').strip(' ')
                    todas_sentencas = todas_sentencas[1:]
                    todas_sentencas = [re.sub('[\s\n\t]', ' ', s) for s in todas_sentencas]
                    todas_sentencas = [s.strip() for s in todas_sentencas if s.count(' ') > 1]

                    for frase in todas_sentencas:
                        if len(set(frase).intersection(set("<>{ }"))) > 0 or not '":"' in frase:   
                            print((frase, url_lema[url]))
                            if ExtratorWikipedia.filtrar_frase_bruta(frase, url_lema[url]) != None:
                                nova_frase = frase.replace('\n', ' ').strip(' ')
                                if not nova_frase in duplicatas:
                                    reg = palavra, descricao, url, nova_frase
                                    registros.append(reg)
                                    duplicatas.add(nova_frase)

                # Iterando referencias recuperadas
                for url_ref in todas_urls_referencias:
                    for frase in ext.extrair_frases_pagina_comum(todas_urls_referencias, palavra):
                        nova_frase = frase.replace('\n', ' ').strip(' ')
                        if not nova_frase in duplicatas:
                            reg = palavra, descricao, url + '@@@@' + url_ref, nova_frase
                            registros.append(reg)
                            duplicatas.add(nova_frase)
                    
        return registros
