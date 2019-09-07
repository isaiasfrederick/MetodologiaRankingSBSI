#!/usr/bin/env python
"""
usage is

  score [vcmfh] <answer> <key> <sensemap>

  -v (or --verbose)
	  verbose -- print out every entry
  -c (or --coarse)
	  coarse grained scoring
  -m (or --mixed)
	  mixed grain scoring
  -f (or --fine)
	  fine grain scoring (default)
  -h (or --help)
	  print out this message, and some information on file formats
"""

detailed_help = """
This is a scorer for senseval2, 2001.  answers are one per line in the form

	<item> <instance> <answer> [ <answer> [ ... ]] [<comment>]

<item> is a lexical item, sometimes just a word, sometimes it notes a
word plus a part of speech.  for example, "fish" or "fish-v"

<instance> is an arbitrary identitier -- cannot contain whitespace.

<answer> is in the form [^ \\t/]+ followed by an optional "/" plus
weight.

a <comment> is in the form "!![^\\n]*"

answers will be compared against a key.  both answers and keys are in
the format described above. additionally, a sense map may be supplied
which provides a subsumption table of the scores.

a sensemap is in the form

<subsummee>[ <numsubsumed> <subsumer>]

HISTORY: a replica of Joseph Rosenzweig's scoring software, rewritten to be
more robust about the format of answers.

Author: Scott Cotton, cotton@linc.cis.upenn.edu

"""

import re
import sys
import getopt


DEBUG=0
#
# answers are in the form (item, instance): answer list
#
INSTANCES_KEYED=0
answer_key = {}
# sum of all the answer weights 
answer_ttl_weight = 0.0


# defaults
VERBOSE=0
FINE=1
MIXED=0
COARSE=0


def parse_answer_line(ln, key=None):
	global INSTANCES_KEYED, answer_ttl_weight
	prg = re.compile(r"^(?P<item>[^ \t]+)\s+(?P<instance>[^ \t]+)\s+(?P<answers>.*)")
	match = re.match(prg, ln)
	if not match:
		print >> sys.stderr, "bad answer line", ln
		return None
	str_answers = match.group("answers")
	# chop off comments
	if str_answers.find("!!") != -1:
	  str_answers = str_answers[:str_answers.find("!!")]
	item = match.group("item")
	instance =  match.group("instance")
	answers = []
	ttl_weight = 0
	weight_specified = 0
	answer_ttl_weight_diff = 0
	for str_answer in str_answers.split():
		if not str_answer: continue
		if str_answer.count('/') > 0:
			weight_specified = 1
			i = str_answer.index('/')
			try:
				answerid, weight = str_answer[:i], float(str_answer[i+1:])
			except ValueError:
				print >> sys.stderr, "bad answer line", ln
				return None
			# take all weights > 1 (usually out of 100) and
			# normalize them to a value between 0 and 1.
			while weight > 1.0: weight /= 100.0
			answers.append((answerid, weight))
			answer_ttl_weight_diff += weight
			ttl_weight += weight
		else:
			answers.append((str_answer, 1.0))
			if not key:
				ttl_weight += 1.0
	# can't deal with these
	if not len(answers):
		return None
	answer_ttl_weight += answer_ttl_weight_diff
	#
	# If the answer line presents no weights, the
	# total weight tried will be normalized to 1.0
	#
	if not key and not weight_specified:
		answer_ttl_weight += 1.0
	#
	# normalize the weights
	#
	if round(ttl_weight) > 1.0:
		wt = 1.0 / len(answers)
		answers_normalized = []
		for answ, old_wt in answers:
			answers_normalized.append((answ, wt))
			ttl_weight += wt
	else:
		answers_normalized = answers
	if key:
		answer_key[(item, instance)] = answers_normalized
		INSTANCES_KEYED += 1
	return item, instance, answers_normalized

#
# these hold the subsumption table information and just the senses.
#
senses_subsumed = {}
senses_subsuming = {}
subsum_ttl = {}

#
# this adds an entry, coordinating the structure of the data in the
# above dicts
#
def add_entry(subsumed, num=0, subsummer=None):
	if subsummer is not None:
		if senses_subsumed.has_key(subsumed):
			senses_subsumed[subsumed].append(subsummer)
			subsum_ttl[subsummer] = num
		else:
			senses_subsumed[subsumed] = [subsummer]
			subsum_ttl[subsummer] = num
		if senses_subsuming.has_key(subsummer):
			senses_subsuming[subsummer].append(subsumed)

		else:
			senses_subsuming[subsummer] = [subsumed]
		senses_subsuming[subsummer].extend(senses_subsuming.get(subsumed, []))		
			

#
# this parses a sensemap file line
#
def parse_senses_line(ln):
	ln = ln.strip()
	spl = ln.split()
	if len(spl) == 1:
		add_entry(ln)
		return
	for i in range(0, len(spl) - 1, 2):
		try:
			subsumee = spl[i]
			num = int(spl[i+1])
			subsumer = spl[i + 2]
			add_entry(subsumee, num, subsumer)
		except IndexError:
			print >> sys.stderr, "bad subsumption table entry: '%s'" % ln			
		except ValueError: # int() failed
			print >> sys.stderr, "bad subsumption table entry: '%s'" % ln

#
# these scores a given answer. XXX note that this assumes
# that the answers have been keyed and the sensemap file has
# been parsed.
#

# for when there's no answer available
class NoScore(Exception): pass

#
# verbose by instance output
#
def fmt_verbose(item, instance, ttl, keyids, answers):
	print 'score for "%s_%s": %0.3f' % (item, instance, ttl)
	print ' key   =',
	print ' '.join(keyids)
	print ' guess =',
	outs = []
	for id, wt in answers:
		outs.append("%s/%0.3f" % (id, wt))
	print ' '.join(outs)
	print

#
# fine grained scoring
#
def score_fine(item, instance, answers):
	keys = answer_key.get((item, instance))
	if keys is None:
		print >> sys.stderr, "no answer key for item %s instance %s" % (item, instance)
		raise NoScore()
	ttl = 0.0
	keyids = []
	for answerid, weight in answers:
		for keyid, keyweight in keys:
			keyids.append(keyid)
			if answerid == keyid:
				ttl += weight
	if VERBOSE:
		fmt_verbose(item, instance, ttl, keyids, answers)
	return ttl

#
# mixed grained scoring
#
def score_mixed(item, instance, answers):
	keys = answer_key.get((item, instance))
	if keys is None:
		print >> sys.stderr, "no answer key for item %s instance %s" % (item, instance)
		raise NoScore()
	ttl = 0.0
	kids = {}
	scored = {}
	# to get the best possible score, check for answers in decreasing level of
	# quality:  same, key subsumes answer, and answer subsumes key
	for aid, awt in answers:
		try:
			for kid, unused_kwt in keys:
				kids[kid] = 1
				if aid == kid:
					ttl += awt
					if DEBUG:
						print "k %s = a %s" % (kid, aid)
					raise "Answered"
			for kid, unused_kwt in keys:
				kids[kid] = 1
				if senses_subsuming.has_key(kid) and \
					 aid in senses_subsuming[kid]:
					ttl += awt
					if DEBUG:
						print "k %s subsumes a %s" % (kid, aid)
					raise "Answered"
			#
			# in ths case we sum all the keys, as for some reason
			# joseph's does.
			#
			for kid, unused_kwt in keys:
				kids[kid] = 1
				if senses_subsuming.has_key(aid) and \
					 kid in senses_subsuming[aid]:
					if DEBUG:
						print "a %s subsumes k %s" % (aid, kid)
					num_subsumed = subsum_ttl[aid]
					ttl += awt * (1.0 / num_subsumed)
		except "Answered":
			pass
	if VERBOSE:
		fmt_verbose(item, instance, ttl, kids.keys(), answers)
	return ttl

#
# return the set of collected (grouped) answers given a particular one
# the set is in the form of a dict.  good thing these relations aren't too
# big
#
def resolve_answer_to_group(answerid):
	dict = {answerid: 1}
	if not senses_subsumed.has_key(answerid):
		return dict
	for k in senses_subsumed[answerid]:
		dict[k] = 1
		dict.update(resolve_answer_to_group(k))
	return dict

#
# coarse grained scoring -- expand both the answer and the
# keys to their respective groups and see if any of the answers
# are in the keys, adding that answers weight as appropriate
#
def score_coarse(item, instance, answers):
	keys = answer_key.get((item, instance))
	if keys is None:
		print >> sys.stderr, "no answer key for item %s instance %s" % (item, instance)
		raise NoScore()
	ttl = 0.0
	prkeys = {}
	for answerid, weight in answers:
		answ_group = resolve_answer_to_group(answerid)
		try:
			for keyid, keyweight in keys:
				prkeys[keyid] = 1
				key_group = resolve_answer_to_group(keyid)
				for kg_answ in key_group.keys():
					if answ_group.has_key(kg_answ):
						ttl += weight
						raise "Answered"
		except "Answered":
			pass
	if VERBOSE:
		fmt_verbose(item, instance, ttl, prkeys.keys(), answers)		
	return ttl
	

#
# print out the answers ... exactly like the c program
#
def summarize(score_ttl, ihandled, ikeyed):
	prec = score_ttl / answer_ttl_weight
	rec = score_ttl / ikeyed

	obj = dict()
	obj['prec'] = prec
	obj['rec'] = rec
	obj['acertos'] = score_ttl
	obj['total_prec'] = answer_ttl_weight
	obj['total_rec'] = answer_ttl_weight

	return obj
																	

def calcular_pontuacao(todos_args):
	#
	# main flow control from here down
	#
	usage = __doc__

	# argument processing
	try:
		optlist, args = getopt.getopt(todos_args[1:], "vcmfh", ["coarse",
															   "mixed",
															   "fine",
															   "help",
															   "verbose"])
	except getopt.error, rest:
		raise Exception("\nParametros errados!\n")

	for opt, val in optlist:
		if opt in ("-v", "--verbose"):
			VERBOSE=1
		elif opt in ("-c", "--coarse"):
			COARSE=1;FINE=0;MIXED=0
		elif opt in ("-m", "--mixed"):
			MIXED=1;COARSE=0;FINE=0
		elif opt in ("-f", "--fine"):
			FINE=1;MIXED=0; COARSE=0
		elif opt in ("-h", "--help"):
			print usage
			print detailed_help
			sys.exit(0)
		else:
			raise Exception('What the hell happened? that\'s s\'poseed to be all the options')
	
	if len(args) != 3:
		raise Exception("\nParametros errados!\n")

	answers_f, key_f, sensemap_f = args

	try:
		# read in the answer key
		key_in = open(key_f)
		for line in key_in.readlines():
			parse_answer_line(line, key=1)
		key_in.close()

		# read in the sense map
		sense_in = open(sensemap_f)
		for line in sense_in.readlines():
			parse_senses_line(line)
		sense_in.close()

		# read in the answers, keeping track of ttl answers, sum score
		score_sum = 0
		instances_handled = 0
		answer_in = open(answers_f)

		for line in answer_in.readlines():
			p = parse_answer_line(line)
			if not p:
				continue
			item, instance, answers = p
			instances_handled += 1
			try:
				if FINE:
					score_sum += score_fine(item, instance, answers)
				elif MIXED:
					score_sum += score_mixed(item, instance, answers)
				elif COARSE:
					score_sum += score_coarse(item, instance, answers)
				else:
					raise Exception("\nParametros errados! %s\n" % "NO SCORING REQUESTED (impossible!)")
 
			except NoScore: # error already reported
				continue

		# print things out just like joseph's original
		#print
		#if FINE:
		#	print "Fine-grained score for",
		#elif MIXED:
		#	print "Mixed-grained score for",
		#elif COARSE:
		#	print "Coarse-grained score for",

		#print '"%s" using key "%s":' % (answers_f, key_f)

		return summarize(score_sum, instances_handled, INSTANCES_KEYED)

	except Exception, rest:
		raise Exception("\nParametros errados!\n")

	return None