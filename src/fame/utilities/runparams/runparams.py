#!/usr/bin/python

info = 'Default run parameters which get customised later on.'
scheme = 'CDR'
renormalized = True
Nf = 5

# MCSymmetrizeFinal = sym factors for indentical final state particles
# HelAvgInitial = avg initial state helicities
# ColAvgInitial = avg initial state colours

extraorder = """
Extra MCSymmetrizeFinal       yes
Extra HelAvgInitial           yes
Extra ColAvgInitial           yes
Extra NJetOmit16PiSq          yes

SetParameter mass(23)         91.18810
SetParameter mass(24)         80.419
SetParameter Width(23)        2.441404
SetParameter Width(24)        2.0476

SetParameter sw2              0.22224648578577766

"""

groups = ["All"]

from math import pi

All = {
'params' : {
    'aspow' : 0,
    'aepow' : 2,
    'mur' : 91.188,
    'as' : 0.118,
    'ae' : 1.0 / 1.325070e+02,
    'mode': 'PLAIN',
    },
}
