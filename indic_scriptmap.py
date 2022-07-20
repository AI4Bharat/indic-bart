#########
# Notes: 
# 1. You will need to install the Indic NLP Library from https://github.com/anoopkunchukuttan/indic_nlp_library
# 2. You will need to install the Indic NLP Resources from https://github.com/anoopkunchukuttan/indic_nlp_resources

# Usage: python indic_scriptmap.py <input_file> <output_file> <source_language> <target_language>
# where <input_file> is the input file containing lines with the original script, and <output_file> is where the file with the script mapped content into the target language will be written.
# <source_language> is the language of the original script, and <target_language> is the language of the script to be mapped into.
# Example: python indic_scriptmap.py input.txt output.txt ta hi
# This will map the script in the input.txt file from Tamil to Hindi.

# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME="path/to/indic_nlp_library"

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES="path/to/indic_nlp_resources"

import sys
infname=sys.argv[1]
outfname=sys.argv[2]

inlang=sys.argv[3]
outlang=sys.argv[4]

sys.path.append('{}'.format(INDIC_NLP_LIB_HOME))

import re
import os
from tqdm import tqdm
import shutil

from indicnlp import common
from indicnlp import loader
common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator as script_conv

loader.load()

if len(sys.argv) < 5:
	print("Usage: python indic_scriptmap.py <input_file> <output_file> <source_language> <target_language>")
	print("where <input_file> is the input file containing lines with the original script, and <output_file> is where the file with the script mapped content into the target language will be written.")
	print("<source_language> is the language of the original script, and <target_language> is the language of the script to be mapped into.")
	print("Example: python indic_scriptmap.py input.txt output.txt ta hi")
	print("This will map the script in the input.txt file from Tamil to Hindi.")
	exit()

print()
print(infname)
print(outfname)
print(inlang)
print(outlang)

with open(outfname,'w',encoding='utf-8') as outfile, \
	 open(infname,'r',encoding='utf-8') as infile:

	for line in infile:
		outline = script_conv.transliterate(line.strip(),inlang,outlang)
		outfile.write(outline+'\n')
