
import sys
import morse
import numpy as np
import cwmodel
import matplotlib.pyplot as plt

cw = cwmodel.CWDetectorTranslator(8000)
gen = cwmodel.TranslationGenerator()
data = gen[0]
pred = cw.trans_model.predict(data[0])
for i,t in enumerate(pred):
    print(i, cwmodel.bins2msg(data[1][i]))  # truth
    print(i, cwmodel.bins2msg(t))           # prediction
    print()


