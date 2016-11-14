import re

resultats = {}
ips = []
alphas = []
ns = []
regex = re.compile(r'n=(?P<n>[0-9]), i=\((?P<i>\d+\.\d+, \d+\.\d+)\), a=(?P<a>\d+\.\d+), err=(?P<err>\d+\.\d+)')
with open("resultats.txt", 'r') as f:
  line = f.readline()
  while line != "":
    tpl = regex.search(line)
    if tpl is not None:
      n = int(tpl.group('n'))
      i = tuple([float(x) for x in tpl.group('i').split(",")])
      a = float(tpl.group('a'))
      err = float(tpl.group('err'))
      print(n, i, a, err)
      if n not in ns:
        ns.append(n)
      if i not in ips:
        ips.append(i)
      if a not in alphas:
        alphas.append(a)
      if n not in resultats:
        resultats[n] = {ips.index(i): {alphas.index(a): err}}
      elif ips.index(i) not in resultats[n]:
        resultats[n][ips.index(i)] = {alphas.index(a): err}
      elif alphas.index(a) not in resultats[n][ips.index(i)]:
        resultats[n][ips.index(i)][alphas.index(a)] = err
    line = f.readline()

print(repr(resultats))
with open("resultats.dict.py", 'w') as f:
  f.write("result = " + repr(resultats) + "\n")
  f.write("alphas = " + repr(alphas) + "\n")
  f.write("ips = " + repr(ips) + "\n")
  f.write("ns = " + repr(ns) + "\n")

