from Alvaro import Alvaro
from RepresentacaoVetorial import *
from OxAPI import *
from DesOx import DesOx
from SemEval2007 import VlddrSemEval
import os

class InterfaceBases():
    OBJETOS = { }
    CFGS = None
    INICIALIZADO = False

    @staticmethod
    def setup(cfgs, dir_keys=None, funcao=None):
        if InterfaceBases.INICIALIZADO == True:
            return

        if dir_keys != None:
            app_cfg = Util.abrir_json("./keys.json")
            cfgs['oxford']['app_id'] = app_cfg['app_id']
            cfgs['oxford']['app_key'] = app_cfg['app_key']

        Util.deletar_arquivo("./Bases/ngrams.tmp")
        Util.CONFIGS = cfgs

        CliOxAPI.CLI = CliOxAPI(cfgs)
        ExtWeb.EXT = ExtWeb(cfgs)

        BaseOx.INSTANCE = BaseOx(cfgs, CliOxAPI.CLI, ExtWeb.EXT)
        RepVetorial.INSTANCE = RepVetorial(cfgs, None, True)
        VlddrSemEval.INSTANCE = VlddrSemEval(cfgs)

        Alvaro.INSTANCE = Alvaro(cfgs, BaseOx.INSTANCE, None, RepVetorial.INSTANCE)

        dir_palavras_monossemicas = Util.CONFIGS['oxford']['dir_palavras_monossemicas']
        Alvaro.PALAVRAS_MONOSSEMICAS = Util.abrir_json(dir_palavras_monossemicas, criarsenaoexiste=True)

        dir_modelo_default = cfgs["caminho_bases"]+"/"+cfgs["modelos"]["default"]

        tamanho_modelo_mb = os.path.getsize(dir_modelo_default) / 1024.0 / 1024.0

        if Alvaro.CANDIDATOS == None or Alvaro.CANDIDATOS == dict():
            dir_base = "./Bases/" + Util.CONFIGS['arquivo_candidatos']
            Alvaro.CANDIDATOS = Util.abrir_json(dir_base, criarsenaoexiste=True)

        modelo_binario = Util.CONFIGS['tipos_modelo'] == 'binario'

        if cfgs['carregar_modelo'] and funcao == None:
            print("\nCarregando modelo '%s'"%dir_modelo_default)
            print("\nTamanho do modelo: %s MBs" % str(tamanho_modelo_mb))
            RepVetorial.INSTANCE.carregar_modelo(dir_modelo_default, binario=modelo_binario)
            print("Modelo carregado!\n")
        else:
            RepVetorial.INSTANCE = None
            print("Modelo NAO carregado!\n")

        DesOx.INSTANCE = DesOx(cfgs, BaseOx.INSTANCE, RepVetorial.INSTANCE)

        InterfaceBases.CFGS = cfgs

        InterfaceBases.OBJETOS[Alvaro.__name__] = Alvaro.INSTANCE
        InterfaceBases.OBJETOS[DesOx.__name__] = DesOx.INSTANCE
        InterfaceBases.OBJETOS[BaseOx.__name__] = BaseOx.INSTANCE
        InterfaceBases.OBJETOS[CliOxAPI.__name__] = CliOxAPI.CLI
        InterfaceBases.OBJETOS[ExtWeb.__name__] = ExtWeb.EXT
        InterfaceBases.OBJETOS[RepVetorial.__name__] = RepVetorial.INSTANCE

        InterfaceBases.INICIALIZADO = True
    
    @staticmethod
    def encerrar_ambiente(cfgs, dir_keys=None):
        pass