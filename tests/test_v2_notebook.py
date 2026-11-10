"""Offline checks for the self-contained Colab v2 notebook (no GPU or model downloads)."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image

NOTEBOOK = Path(__file__).resolve().parents[1] / 'Qwen_Image_2_1_v2.ipynb'


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))


def source(cell):
    return ''.join(cell['source'])


def make_utils(notebook):
    setup = source(notebook['cells'][2])
    line = next(line for line in setup.splitlines() if line.startswith('SHARED_UTILS_CODE = '))
    code = ast.literal_eval(line.split(' = ', 1)[1])
    module = types.ModuleType('qwen_studio_utils')
    exec(compile(code, 'qwen_studio_utils.py', 'exec'), module.__dict__)
    return module


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.nb = load_notebook()
        self.utils = make_utils(self.nb)

    def test_notebook_and_embedded_sources_compile(self):
        for cell in self.nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            code = source(cell)
            # Colab shell escapes are not Python syntax, but the remaining source is.
            code = '\n'.join(line for line in code.splitlines() if not line.startswith('!'))
            compile(code, 'cell', 'exec')
        titles = [source(cell).splitlines()[0] for cell in self.nb['cells']]
        self.assertFalse(any('Text-to-Image Generation' in title for title in titles))
        self.assertTrue(any('Multi-Image Editing' in title for title in titles))

    def test_civit_preset_selects_matching_qwen_version_not_first(self):
        u = self.utils
        versions = [
            {'id': 11, 'baseModel': 'Krea 2', 'files': [{'name': 'krea.safetensors'}]},
            {'id': 22, 'baseModel': 'Qwen 2.1', 'files': [{'name': u.ANYTHING2REAL_LORA_FILE}]},
        ]
        response = mock.Mock(); response.json.return_value = {'modelVersions': versions}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(u.requests, 'get', return_value=response), \
             mock.patch.object(u, 'verify_safetensors_file', side_effect=[(False, 0, 'missing'), (True, 304, 'ok'), (True, 304, 'ok')]), \
             mock.patch.object(u.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run:
            u.download_lora_multisource(u.PRESET_LORA_REGISTRY[u.CIVIT_ANYTHING2REAL_LORA_FILE]['url'], tmp, u.CIVIT_ANYTHING2REAL_LORA_FILE)
            self.assertEqual(run.call_args.args[0][-1], 'https://civitai.com/api/download/models/22')
        response.json.return_value = {'modelVersions': versions[:1]}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(u.requests, 'get', return_value=response), \
             mock.patch.object(u, 'verify_safetensors_file', return_value=(False, 0, 'missing')), \
             mock.patch.object(u.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Qwen 2.1'):
                u.download_lora_multisource('https://civitai.com/models/2808492/anything2realcharacters', tmp, u.CIVIT_ANYTHING2REAL_LORA_FILE)
            run.assert_not_called()

    def test_huggingface_blob_url_is_rewritten(self):
        u = self.utils
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(u, 'verify_safetensors_file', side_effect=[(False, 0, 'missing'), (True, 2, 'ok'), (True, 2, 'ok')]), \
             mock.patch.object(u.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run:
            u.download_lora_multisource('https://huggingface.co/user/model/blob/main/a.safetensors', tmp, 'a.safetensors')
            self.assertEqual(run.call_args.args[0][-1], 'https://huggingface.co/user/model/resolve/main/a.safetensors')

    def test_history_errors_and_empty_results_are_not_success(self):
        u = self.utils
        for history in ({'status': {'status_str': 'error', 'messages': ['failed']}}, {'outputs': {}}):
            response = mock.Mock(); response.read.return_value = json.dumps({'pid': history}).encode()
            with mock.patch.object(u.urllib.request, 'urlopen', return_value=response):
                with self.assertRaises(RuntimeError):
                    u._fetch_history_outputs('pid')

    def test_no_silent_old_images_on_empty_edit_input(self):
        u = self.utils
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(u, 'get_engine_dir', return_value=tmp):
            os.makedirs(os.path.join(tmp, 'input', 'batch_edit_images'))
            Image.new('RGB', (32, 64)).save(os.path.join(tmp, 'input', 'batch_edit_images', 'old.png'))
            with self.assertRaises(ValueError):
                u.load_multiple_raw_images('paths_or_urls', '', 'batch_edit_images', reuse_cached=False)

    def test_load_folder_and_zip_in_stable_order(self):
        u = self.utils
        import io
        import zipfile
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(u, 'get_engine_dir', return_value=tmp):
            folder = Path(tmp) / 'images'
            folder.mkdir()
            for name in ('b.png', 'a.png'):
                Image.new('RGB', (20, 30)).save(folder / name)
            items = u.load_multiple_raw_images('paths_or_urls', str(folder), reuse_cached=False)
            self.assertEqual([name for name, _ in items], ['a.png', 'b.png'])
            buf = io.BytesIO()
            Image.new('RGB', (15, 25)).save(buf, format='PNG')
            archive_path = Path(tmp) / 'images.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('nested/one.png', buf.getvalue())
                archive.writestr('nested/broken.png', b'not an image')
            items = u.load_multiple_raw_images('paths_or_urls', str(archive_path), reuse_cached=False)
            self.assertEqual([name for name, _ in items], ['one.png'])

    def test_turbo_workflow_keeps_edit_image_and_seed(self):
        u = self.utils
        workflow = {
            'unet_loader': {'inputs': {}, 'class_type': 'UNETLoader'},
            'qwen_cache': {'inputs': {'model': ['unet_loader', 0]}, 'class_type': 'QwenImage21Cache'},
            'text_encode_edit': {'inputs': {'images.image_1': ['load_input_image', 0], 'negative_prompt': 'bad'}, 'class_type': 'TextEncodeQwenImage21'},
            'ksampler': {'inputs': {'seed': 321, 'steps': 20, 'positive': ['text_encode_edit', 0], 'latent_image': ['empty_latent', 0], 'model': ['qwen_cache', 0]}, 'class_type': 'KSampler'},
        }
        loras = [{'name': u.LIGHTNING_LORA_FILE, 'strength_model': 1, 'strength_clip': 1}]
        model, clip = u.build_lora_chain(workflow, loras, ['unet_loader', 0], ['clip_loader', 0], check_disk=False)
        workflow['qwen_cache']['inputs']['model'] = model
        u.apply_acceleration_to_workflow(workflow, loras, steps=4, cfg=1, sampler_name='euler')
        self.assertEqual(workflow['random_noise']['inputs']['noise_seed'], 321)
        self.assertEqual(workflow['text_encode_edit']['inputs']['images.image_1'], ['load_input_image', 0])
        self.assertEqual(workflow['ksampler']['class_type'], 'SamplerCustomAdvanced')
        self.assertNotIn('qwen_cache', workflow)

    def test_edit_batch_runs_each_graph_sequentially_and_zips_successes(self):
        u = self.utils
        cell = next(source(c) for c in self.nb['cells'] if source(c).startswith('# @title 5. Multi-Image'))
        cell = cell.replace('seed = -1 # @param', 'seed = 123 # @param')
        items = [('landscape.png', Image.new('RGB', (400, 200))), ('portrait.png', Image.new('RGB', (200, 400))), ('bad.png', Image.new('RGB', (200, 200)))]
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            def generate(workflow, title, w, h, seed, summary, **kwargs):
                calls.append((copy.deepcopy(workflow), w, h, seed, kwargs))
                if len(calls) == 2:
                    raise RuntimeError('test failure')
                path = os.path.join(tmp, f'result{len(calls)}.png')
                Image.new('RGB', (w, h)).save(path)
                generate.last_saved_paths = [path]
                return [Image.new('RGB', (w, h))]
            generate.last_saved_paths = []
            u.load_multiple_raw_images = lambda **kwargs: items
            u.save_processed_image = lambda img, filename, target_dims: (f'input{img.width}x{img.height}.png', *target_dims)
            u.ensure_loras_downloaded = lambda loras: None
            u.execute_comfy_generation = generate
            u.purge_backend_memory = lambda **kwargs: None
            u.get_output_dir = lambda: tmp
            with mock.patch.dict(sys.modules, {'qwen_studio_utils': u}):
                namespace = {}
                exec(compile(cell, 'batch_edit', 'exec'), namespace)
            self.assertEqual(len(calls), 3)
            self.assertEqual([call[3] for call in calls], [123, 123, 123])
            self.assertGreater(calls[0][1], calls[0][2])
            self.assertLess(calls[1][1], calls[1][2])
            self.assertTrue(all('images.image_2' not in call[0]['text_encode_edit']['inputs'] for call in calls))
            self.assertEqual(len(namespace['batch_failures']), 1)
            import zipfile
            archives = list(Path(tmp).glob('*.zip'))
            self.assertEqual(len(archives), 1)
            with zipfile.ZipFile(archives[0]) as archive:
                self.assertEqual(sorted(archive.namelist()), ['result1.png', 'result3.png'])


if __name__ == '__main__':
    unittest.main()
