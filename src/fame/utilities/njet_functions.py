# Code adapted from NJet Library

import getopt
import itertools
import os
import re
import signal
import sys
import time
from functools import cmp_to_key
from math import pi, sqrt
from tqdm.auto import tqdm

from fame.utilities import runparams

signal.signal(signal.SIGINT, signal.SIG_DFL)

# you will need to install njet and point to it here
NJET_DIR = '/mt/home/htruong/software/njet/local'
sys.path.append(NJET_DIR)

try:  # for python3
    from importlib.machinery import SourceFileLoader

    njet = SourceFileLoader('njet', os.path.join(os.path.dirname(__file__), NJET_DIR + "/bin/njet.py" )).load_module()
except ImportError:  # for python2
    import imp

    njet = imp.load_source('njet', os.path.join(os.path.dirname(__file__),NJET_DIR + "/bin/njet.py"))

OLP = njet.OLP

DEBUG = False
SLCTEST = None
CCTEST = None
NPOINTS = 50000000
VIEW = 'NJ'
if sys.platform.startswith('linux'):
    LIBNJET = os.path.join(os.path.dirname(__file__), NJET_DIR + '/lib/libnjet2.so')
elif sys.platform.startswith('darwin'):
    LIBNJET = os.path.join(os.path.dirname(__file__), NJET_DIR + '/lib/libnjet2.dylib')
else:
    print("Warning: unknown system '%s'. Library will probably fail to load." % sys.platform)
    LIBNJET = os.path.join(os.path.dirname(__file__), NJET_DIR + '/lib/libnjet2.dll')

factorial = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]  # n!

ORDER_TPL = """
# fixed
CorrectionType          QCD
BLHA1TwoCouplings       yes

# changable
#Extra NJetMultiPrec 2
#Extra Precision 1e-2
Extra NJetPrintStats yes

# test-specific
%s

# process list
"""


def njet_init(order):
    if os.path.exists(LIBNJET):
        status = OLP.OLP_Start(order, libnjet=LIBNJET)
    else:
        print("Warning: '%s' not found, trying system default." % LIBNJET)
        status = OLP.OLP_Start(order)
    if DEBUG:
        if status:
            print(OLP.contract)
        else:
            print(order)
    return status


def relerr(a, b):
    if abs(a + b) != 0.:
        return abs(2. * (a - b) / (a + b))
    else:
        return abs(a - b)


def nis(i, j):
    return i + j * (j - 1) / 2 if i <= j else j + i * (i - 1) / 2


def calculate_treeval(mom, params, data):
    alphas = params.get('as', 1.)
    alpha = params.get('ae', 1.)
    mur = params.get('mur', 1.)
    legs = len(mom[0])

    for p in data:
        mcn = p['mcn']
        if 'name' in p:
            name = p['name']
        else:
            name = repr(p['inc'] + p['out'])
        npoints = min(NPOINTS, len(mom))
        print("-------- channel %s -------- (%d points)" % (name, npoints))
        treevals = []
        for j in tqdm(range(npoints)):
            retval = OLP.OLP_EvalSubProcess(mcn+2, mom[j], alphas=alphas, alpha=alpha, mur=mur)
            born = retval[0]
            treevals.append(born)
            if born == 0:
                print("ERROR born = 0")
                born = 1
    return treevals


def chan_has_lc(p):
    channel = njet.Channel([njet.Process.cross_flavour(i) for i in p['inc']] + p['out'])
    chanmatches = njet.Process.canonical.get(channel.canon_list, None)
    if chanmatches:
        return chanmatches[0].has_lc
    return False


def add_to_order(mcn, order, test):
    new = ["\n"]
    params = test['params']
    if CCTEST:
        params['type'] = CCTEST
    ptype = params.get('type', 'CC')
    new.append("AlphasPower %d" % params.get('aspow', 0))
    new.append("AlphaPower  %d" % params.get('aepow', 0))
    if ptype == 'DS':
        new.append("AmplitudeType LoopDS")
    else:
        new.append("AmplitudeType Loop")
    new.append(params.get('order', ''))
    for p in test['data']:
        mcn += 1
        p['mcn'] = mcn
        procline = "%s -> %s" % (' '.join(map(str, p['inc'])), ' '.join(map(str, p['out'])))
        new.append(procline)
        if ptype == 'NORMAL':
            p['has_lc'] = chan_has_lc(p)
            if p['has_lc']:
                new.append("Process %d AmplitudeType LoopLC" % (mcn + 1))
                new.append(procline)
                new.append("Process %d AmplitudeType LoopSLC" % (mcn + 2))
                new.append(procline)
                mcn += 2
        elif ptype == 'CC':
            new.append("Process %d AmplitudeType ccTree" % (mcn + 1))
            new.append(procline)
            new.append("Process %d AmplitudeType Tree" % (mcn + 2))
            new.append(procline)
            mcn += 2
        elif ptype == 'SC':
            new.append("Process %d AmplitudeType scTree" % (mcn + 1))
            new.append(procline)
            new.append("Process %d AmplitudeType Tree" % (mcn + 2))
            new.append(procline)
            mcn += 2

    order += '\n'.join(new)
    return mcn, order


def run_batch(curorder, curtests):
    if DEBUG:
        curorder = "NJetReturnAccuracy 2\n" + curorder
    order = ORDER_TPL % curorder
    mcn = 0
    seen = []
    for t in curtests:
        test = t['test']
        if test not in seen:
            mcn, order = add_to_order(mcn, order, test)
            seen.append(test)
    if not njet_init(order):
        print("Skipping batch due to errors")
        return

    test_data = []
    for t in curtests:
        proc = t['proc']
        test = t['test']
        params = test['params']
        if proc:
            data = [d for d in test['data'] if d['name'] == proc]
        else:
            data = test['data']
        if not data:
            print("Warning: can't find %s" % proc)
            continue

        test_data.append([params, data])
    
    return test_data, order


def run_tests(mods, tests):
    cmporder_tmp = [order_global(m) for m in mods]

    def cmporder(x, y):
        return cmp(cmporder_tmp.index(x[0]), cmporder_tmp.index(y[0]))

    sortmods = sorted([(order_global(m), m) for m in mods], key=cmp_to_key(cmporder))
    curorder = ''
    curtests = []
    for order, m in sortmods:
        if order != curorder:
            if curorder:
                run_batch(curorder, curtests)
            curorder = order
            curtests = [t for t in tests if t['mod'] == m]
        else:
            curtests.extend([t for t in tests if t['mod'] == m])
    return curorder, curtests


def order_global(mod):
    order = ['IRregularisation %s' % mod.scheme]
    if mod.renormalized:
        order.append('Extra NJetRenormalize yes')
    else:
        order.append('Extra NJetRenormalize no')
    order.append('Extra SetParameter qcd(nf) %d' % mod.Nf)
    order.append(mod.extraorder)
    order = '\n'.join(order).rstrip(' \n')
    order = re.sub(r'\n\n+', r'\n', order)
    order = re.sub(r'\s\s+', r' ', order)
    return order


def action_run(param_file):
    '''
    Rewritten action_run taking param file as input
    example input: 'NJ_2J'
    '''

    modname, testname, proc = (param_file.split(':') + ['', ''])[:3]
    m = getattr(runparams, modname, None)
    mods = []
    tests = []
    
    mods.append(m)
    testname = m.groups
    for t in testname:
        tests.append({'mod' : m,
                      'test' : getattr(m, t),
                      'testname' : t,
                      'proc' : proc})
    
    return mods, tests
