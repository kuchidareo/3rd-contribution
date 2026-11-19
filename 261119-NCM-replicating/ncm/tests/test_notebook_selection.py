"""Execute notebook training dispatch without torch or data downloads."""
import json
from pathlib import Path
import unittest


class SelectionTests(unittest.TestCase):
    def test_ncm_only_never_starts_ar(self):
        notebook = Path(__file__).resolve().parents[1] / 'notebooks/02_train_tinystories_kaggle.ipynb'
        cells = json.loads(notebook.read_text())['cells']
        source = ''.join(cells[5]['source'])
        dispatch = source[source.index("if 'ar' in TRAIN_MODELS:"):]
        calls = []
        context = dict(TRAIN_MODELS=['ncm'], AR_RESUME=None, NCM_RESUME=None,
                       train_model=lambda *args: calls.append(args))
        exec(dispatch, context)
        exec(''.join(cells[6]['source']), context)
        self.assertEqual(calls, [('ncm', None)])

    def test_ar_only_never_starts_ncm(self):
        notebook = Path(__file__).resolve().parents[1] / 'notebooks/02_train_tinystories_kaggle.ipynb'
        cells = json.loads(notebook.read_text())['cells']
        source = ''.join(cells[5]['source'])
        calls = []
        context = dict(TRAIN_MODELS=['ar'], AR_RESUME='saved.pt', NCM_RESUME=None,
                       train_model=lambda *args: calls.append(args))
        exec(source[source.index("if 'ar' in TRAIN_MODELS:"):], context)
        exec(''.join(cells[6]['source']), context)
        self.assertEqual(calls, [('ar', 'saved.pt')])
