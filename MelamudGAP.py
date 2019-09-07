# This file contains methods that implement Generalized Average Precision
import sys
import io

def constructX(my_ranklist, gold_ranklist):
  x = [ ];
  for item in my_ranklist:
    x.append(int(item in gold_ranklist));
  return x;

def precision(x, i):
  return sum(x[:i])/(i + 0.0);

def average_precision(my_ranklist, gold_ranklist):
  result = 0.0;
  x = constructX(my_ranklist, gold_ranklist);
  for i in range(1, len(my_ranklist) + 1):
    #print >>sys.stderr, 'i = ' + str(i) + ' : precision(x, i) = ' + str(precision(x, i));
    result += x[i - 1] * precision(x, i);
    #print >>sys.stderr, 'result += ' + str(x[i-1]) + ' * ' + str(precision(x,i)) + '( = ' + str(x[i-1] * precision(x,i)) + '), result = ' + str(result);
  return result / len(gold_ranklist);

def I(val):
  return int(val > 0)

def average(arr):
  return sum(arr) / (len(arr) + 0.);

def gap(my_ranklist, gold_ranklist, gold_weights):
  x = constructX(my_ranklist, gold_ranklist);
  result = 0.;
  for i in range(1, len(my_ranklist) + 1):
    result += I(x[i - 1]) * precision(x, i);
  denominator = 0.;
  for i in range(len(gold_ranklist)):
    denominator += I(gold_weights[i]) * average(gold_weights[:i+1]);
  return result / denominator;
