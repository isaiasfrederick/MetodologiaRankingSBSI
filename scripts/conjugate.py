from pattern.en import conjugate
import sys

resultado=[ ]

tense=['infinitive', 'present', 'past']
person=[None, 1, 2, 3]
number=[None, 'singular', 'plural']
mood=[None, 'indicative']
aspect=[None, 'imperfective', 'progressive']
negated=[True, False]

verb=sys.argv[1]

for verb in ["have", "take", "make", "give", "go", "do", "set", "keep", "hold"]:
	for t in tense:
		for p in person:
			for n in number:
				for m in mood:
					for a in aspect:
						for neg in negated:
							try:
								c=conjugate(verb, tense=t, person=p, number=n, mood=m, aspect=a, negated=neg)
								if c != None:
									resultado.append(c)
									resultado=list(set(resultado))
							except:
								pass

raw_input(resultado)
