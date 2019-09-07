from os import system

system('git add .')
mensagem = raw_input('Digite a mensagem do commit: ')
system('git commit -m "%s"' % mensagem)
system('git push origin master')
