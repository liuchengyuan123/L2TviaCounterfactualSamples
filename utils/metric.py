'''
Description: BLEU and Rouge metric.
'''
import os
import subprocess
import re
import io
import time
import logging
import pyrouge


TOTAL_BAR_LENGTH = 100.
last_time = time.time()
begin_time = last_time
# print(os.popen('stty size', 'r').read())

def bleu_score(labels_file, predictions_path):
    # bleu_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'multi-bleu.perl')
    bleu_script = './multi-bleu.perl'
    try:
      with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
        bleu_out = subprocess.check_output(
            [bleu_script, labels_file],
            stdin=predictions_file,
            stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        # print(bleu_score)
        return float(bleu_score)

    except subprocess.CalledProcessError as error:
      if error.output is not None:
        msg = error.output.strip()
        logging.warning(
            "{} script returned non-zero exit code: {}".format(bleu_script, msg))
      return None

def check_res(res):
    if res.strip() == "":
        return False
    for token in res.strip():
        if token.isalpha():
            return True

    return False

def rouge_score(r) -> str:
    logging.getLogger('global').setLevel(logging.WARNING)
    output = r.convert_and_evaluate()
    logging.getLogger('global').setLevel(logging.DEBUG)
    result_dict = r.output_to_dict(output)
    return output, result_dict