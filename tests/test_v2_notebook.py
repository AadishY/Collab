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

    def test_civit_preset_downloads_pinned_version_without_lookup(self):
        u = self.utils
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(u.requests, 'get') as lookup, \
             mock.patch.object(u, 'verify_safetensors_file', side_effect=[(False, 0, 'missing'), (True, 304, 'ok'), (True, 304, 'ok')]), \
             mock.patch.object(u.subprocess, 'run', return_value=types.SimpleNamespace(returncode=0)) as run:
            u.download_lora_multisource(u.PRESET_LORA_REGISTRY[u.CIVIT_ANYTHING2REAL_LORA_FILE]['url'], tmp, u.CIVIT_ANYTHING2REAL_LORA_FILE)
            self.assertEqual(run.call_args.args[0][-1], 'https://civitai.com/api/download/models/3363600')
            lookup.assert_not_called()
            self.assertIn('3363600', u.CIVIT_ANYTHING2REAL_LORA_FILE)

    def test_trigger_insertion_and_four_slot_chain(self):
        u = self.utils
        names = [u.RESTORE_LORA_FILE, u.CIVIT_ANYTHING2REAL_LORA_FILE, u.EXPOSURE_LORA_FILE, u.CONSISTENCY_LORA_FILE]
        active = u.collect_active_loras([(name, '', 0.8) for name in names])
        prompt = u.add_lora_triggers('A PHOTOREALISTIC portrait', active)
        self.assertEqual(prompt.lower().count('photorealistic'), 1)
        self.assertIn('transform the image to realistic photograph', prompt)
        self.assertIn('Transform the image with balanced neutral exposure', prompt)
        self.assertEqual(u.add_lora_triggers(prompt, active), prompt)
        self.assertEqual(u.add_lora_triggers('plain', active, False), 'plain')
        zero = u.collect_active_loras([(u.EXPOSURE_LORA_FILE, '', 0)])
        self.assertEqual(u.add_lora_triggers('plain', zero), 'plain')
        wf = {}
        model, clip = u.build_lora_chain(wf, active, ['unet', 0], ['clip', 0], check_disk=False)
        self.assertEqual(model, ['lora_loader_4', 0])
        self.assertEqual(clip, ['lora_loader_4', 1])
        for i in range(2, 5):
            self.assertEqual(wf[f'lora_loader_{i}']['inputs']['model'], [f'lora_loader_{i-1}', 0])
        with self.assertRaisesRegex(ValueError, 'more than once'):
            u.collect_active_loras([(names[0], '', 1), (names[0], '', 1)])
        with self.assertRaisesRegex(ValueError, 'either Lightning or Viggle'):
            u.collect_active_loras([(u.LIGHTNING_LORA_FILE, '', 1), (u.VIGGLE_TURBO_UNET_FILE, '', 1)])
        with self.assertRaisesRegex(ValueError, 'custom LoRA'):
            u.collect_active_loras([('Custom (Enter filename below)', '', 1)])

    def test_safetensors_validation_rejects_truncated_data(self):
        import struct
        u = self.utils
        header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.safetensors'
            path.write_bytes(struct.pack('<Q', len(header)) + header + b'1234')
            self.assertTrue(u.verify_safetensors_file(str(path), 0)[0])
            path.write_bytes(path.read_bytes()[:-1])
            self.assertFalse(u.verify_safetensors_file(str(path), 0)[0])

    def test_new_presets_are_available_in_all_four_slots(self):
        u = self.utils
        for cell in self.nb['cells'][5:8]:
            src = source(cell)
            for slot in ('primary', 'secondary', 'tertiary', 'quaternary'):
                row = next(line for line in src.splitlines() if line.startswith(slot + '_lora = '))
                for name in (u.CIVIT_ANYTHING2REAL_LORA_FILE, u.RESTORE_LORA_FILE, u.EXPOSURE_LORA_FILE, u.CONSISTENCY_LORA_FILE):
                    self.assertIn(name, row)
            self.assertIn('add_lora_triggers(', src)

    def test_headswap_four_slots_and_inheritance(self):
        u = self.utils
        img = Image.new('RGB', (100, 100))
        u.load_raw_image = lambda **kwargs: img
        u.display_comparison_preview = lambda *args, **kwargs: None
        u.save_processed_image = lambda img, name, **kwargs: (name, 100, 100)
        u.save_cell6_config = lambda cfg: None
        u.ensure_loras_downloaded = lambda loras: None
        seen = []
        u.execute_comfy_generation = lambda wf, *args, **kwargs: seen.append(copy.deepcopy(wf))
        head = source(self.nb['cells'][6])
        head = head.replace('seed = -1 # @param', 'seed = 123 # @param')
        head = head.replace('tertiary_lora = "None" # @param', f'tertiary_lora = "{u.EXPOSURE_LORA_FILE}" # @param')
        head = head.replace('quaternary_lora = "None" # @param', f'quaternary_lora = "{u.CIVIT_ANYTHING2REAL_LORA_FILE}" # @param')
        with mock.patch.dict(sys.modules, {'qwen_studio_utils': u}):
            namespace = {}
            exec(compile(head, 'head', 'exec'), namespace)
        wf = seen[0]
        prompt = wf['text_encode_head']['inputs']['prompt']
        self.assertIn('Transform the image with balanced neutral exposure', prompt)
        self.assertIn('transform the image to realistic photograph', prompt)
        self.assertEqual(prompt.count('head_swap:'), 1)
        self.assertIn('lora_loader_3', wf)
        cfg = namespace['_CELL6_LAST_CONFIG']
        self.assertEqual(cfg['tertiary_lora'], u.EXPOSURE_LORA_FILE)
        self.assertEqual(cfg['quaternary_lora'], u.CIVIT_ANYTHING2REAL_LORA_FILE)
        u.load_or_reuse_reference_image = lambda **kwargs: img
        u.load_multiple_raw_images = lambda **kwargs: [('target', img)]
        captured = []
        u.execute_batch_head_swap_6b = lambda **kwargs: captured.append(kwargs)
        with mock.patch.dict(sys.modules, {'qwen_studio_utils': u}):
            exec(compile(source(self.nb['cells'][7]), 'head_batch', 'exec'), namespace)
        batch = captured[0]
        self.assertEqual(batch['base_seed'], 123)
        self.assertEqual([x['name'] for x in batch['active_loras']], [u.BFS_HEAD_LORA_FILE, u.EXPOSURE_LORA_FILE, u.CIVIT_ANYTHING2REAL_LORA_FILE])
        self.assertIn('Transform the image with balanced neutral exposure', batch['final_prompt'])
        self.assertIn('transform the image to realistic photograph', batch['final_prompt'])

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

    def test_viggle_alias_and_manual_step_count(self):
        u = self.utils
        self.assertNotIn(u.VIGGLE_TURBO_LORA_R256, u.PRESET_LORA_REGISTRY[u.VIGGLE_TURBO_LORA_R128]['aliases'])
        self.assertEqual(u.PRESET_LORA_REGISTRY[u.VIGGLE_TURBO_LORA_R256]['file'], u.VIGGLE_TURBO_LORA_R256)
        active = [{'name': u.VIGGLE_TURBO_UNET_FILE, 'strength_model': 1}]
        steps, cfg, _, info = u.optimize_sampling_for_active_loras(active, 20, 1.1, 'euler', auto_optimize=False)
        self.assertEqual(steps, len(info['sigmas_nodes'].split(',')))
        self.assertEqual(cfg, 1.0)

    def test_sequential_batch_head_workflow_uses_four_loras(self):
        u = self.utils
        img = Image.new('RGB', (100, 120))
        loras = u.collect_active_loras([(name, '', 1) for name in (
            u.BFS_HEAD_LORA_FILE, u.RESTORE_LORA_FILE, u.EXPOSURE_LORA_FILE, u.CONSISTENCY_LORA_FILE)])
        u.ensure_server_alive = lambda: None
        u.purge_backend_memory = lambda **kwargs: None
        u.ensure_loras_downloaded = lambda active: None
        u.get_vram_stats = lambda: {'vram_total_gb': 16, 'vram_free_gb': 12, 'device_name': 'mock'}
        u.display = lambda *args, **kwargs: None
        u.save_processed_image = lambda image, name, **kwargs: (name, 100, 120)
        seen = []
        def generate(wf, *args, **kwargs):
            seen.append(copy.deepcopy(wf))
            return [img]
        generate.last_saved_paths = []
        u.execute_comfy_generation = generate
        with mock.patch.object(u.random, 'sample', return_value=[101, 202]):
            output = u.execute_batch_head_swap_6b(
            target_items=[('one', img), ('two', img)], raw_pic2=img,
            final_prompt=u.add_lora_triggers('head_swap:', loras), negative_prompt='',
            dimension_mode='Match Target Image (Picture 1)', scene_megapixels=1,
            lower_dimensions_for_speed=False, speed_target_resolution='', aspect_ratio='1:1 (Square)',
            custom_width=1024, custom_height=1024, steps=20, base_seed=-1, cfg=1,
            sampler_name='euler', scheduler='simple', active_loras=loras, parallel_2_images=False,
                show_side_by_side_comparison=False, create_zip_bundle=False)
        self.assertEqual(len(output), 2)
        self.assertEqual([wf['ksampler']['inputs']['seed'] for wf in seen], [101, 202])
        for workflow in seen:
            self.assertIn('lora_loader_4', workflow)
            self.assertIn('Transform the image with balanced neutral exposure', workflow['text_encode_head']['inputs']['prompt'])
            self.assertIn('images.image_2', workflow['text_encode_head']['inputs'])

    def test_parallel_head_recovery_keeps_already_completed_output(self):
        u = self.utils
        img = Image.new('RGB', (100, 120))
        u.ensure_server_alive = lambda: None
        u.purge_backend_memory = lambda **kwargs: None
        u.get_vram_stats = lambda: {'vram_total_gb': 16, 'vram_free_gb': 12, 'device_name': 'mock'}
        u.display = lambda *args, **kwargs: None
        u.render_dashboard = lambda *args, **kwargs: None
        u.save_processed_image = lambda image, name, **kwargs: (name, 100, 120)
        ids = iter(['first', 'second'])
        u.queue_prompt = lambda wf: {'prompt_id': next(ids)}
        u._fetch_history_outputs = lambda pid: [img]
        saved = []
        u._save_and_display_result = lambda imgs, *args, **kwargs: saved.append('recovered') or ['recovered.png']
        generated = []
        def generate(wf, *args, **kwargs):
            generated.append(wf)
            return [img]
        generate.last_saved_paths = ['retry.png']
        u.execute_comfy_generation = generate
        class WS:
            def __init__(self):
                self.messages = iter([
                    json.dumps({'type': 'executing', 'data': {'prompt_id': 'first', 'node': None}}),
                    json.dumps({'type': 'execution_error', 'data': {'prompt_id': 'second', 'exception_message': 'failure'}}),
                ])
            def connect(self, *args, **kwargs): pass
            def settimeout(self, *args, **kwargs): pass
            def close(self): pass
            def recv(self): return next(self.messages)
        with mock.patch.object(u.websocket, 'WebSocket', WS):
            output = u.execute_batch_head_swap_6b(
                target_items=[('one', img), ('two', img)], raw_pic2=img, final_prompt='head_swap:', negative_prompt='',
                dimension_mode='Match Target Image (Picture 1)', scene_megapixels=1, lower_dimensions_for_speed=False,
                speed_target_resolution='', aspect_ratio='1:1 (Square)', custom_width=1024, custom_height=1024,
                steps=20, base_seed=123, cfg=1, sampler_name='euler', scheduler='simple', active_loras=[],
                parallel_2_images=True, show_side_by_side_comparison=False, create_zip_bundle=False)
        self.assertEqual(len(output), 2)
        self.assertEqual(saved, ['recovered'])
        self.assertEqual(len(generated), 1)

    def test_edit_random_seeds_are_unique_and_reach_turbo_workflows(self):
        import random
        u = self.utils
        cell = source(self.nb['cells'][5])
        cell = cell.replace('secondary_lora = "None" # @param', f'secondary_lora = "{u.LIGHTNING_LORA_FILE}" # @param')
        img = Image.new('RGB', (100, 120))
        u.load_multiple_raw_images = lambda **kwargs: [(f'{i}.png', img) for i in range(3)]
        u.save_processed_image = lambda image, name, **kwargs: (name, 100, 120)
        u.ensure_loras_downloaded = lambda active: None
        u.purge_backend_memory = lambda **kwargs: None
        generated = []
        def generate(wf, title, w, h, seed, summary, **kwargs):
            generated.append((copy.deepcopy(wf), seed))
            return [img]
        generate.last_saved_paths = []
        u.execute_comfy_generation = generate
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(u, 'get_output_dir', return_value=tmp), \
             mock.patch.object(random, 'sample', return_value=[101, 202, 303]) as sample, \
             mock.patch.dict(sys.modules, {'qwen_studio_utils': u}):
            namespace = {}
            exec(compile(cell, 'random_batch', 'exec'), namespace)
        sample.assert_called_once()
        self.assertEqual([seed for _, seed in generated], [101, 202, 303])
        self.assertEqual(namespace['batch_seeds'], [101, 202, 303])
        for wf, item_seed in generated:
            self.assertEqual(wf['random_noise']['inputs']['noise_seed'], item_seed)
            self.assertEqual(wf['ksampler']['class_type'], 'SamplerCustomAdvanced')

    def test_edit_rejects_invalid_seed_before_loading_images(self):
        u = self.utils
        cell = source(self.nb['cells'][5]).replace('seed = -1 # @param', 'seed = -2 # @param')
        u.load_multiple_raw_images = mock.Mock()
        with mock.patch.dict(sys.modules, {'qwen_studio_utils': u}):
            with self.assertRaisesRegex(ValueError, 'Seed must be'):
                exec(compile(cell, 'invalid_seed', 'exec'), {})
        u.load_multiple_raw_images.assert_not_called()

    def test_edit_batch_runs_each_graph_sequentially_and_zips_successes(self):
        u = self.utils
        cell = next(source(c) for c in self.nb['cells'] if source(c).startswith('# @title 5. Multi-Image'))
        cell = cell.replace('seed = -1 # @param', 'seed = 123 # @param')
        cell = cell.replace('secondary_lora = "None" # @param', f'secondary_lora = "{u.CIVIT_ANYTHING2REAL_LORA_FILE}" # @param')
        cell = cell.replace('tertiary_lora = "None" # @param', f'tertiary_lora = "{u.RESTORE_LORA_FILE}" # @param')
        cell = cell.replace('quaternary_lora = "None" # @param', f'quaternary_lora = "{u.EXPOSURE_LORA_FILE}" # @param')
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
            self.assertEqual([call[0]['ksampler']['inputs']['seed'] for call in calls], [123, 123, 123])
            self.assertGreater(calls[0][1], calls[0][2])
            self.assertLess(calls[1][1], calls[1][2])
            self.assertTrue(all('images.image_2' not in call[0]['text_encode_edit']['inputs'] for call in calls))
            for wf, _, _, _, _ in calls:
                self.assertIn('lora_loader_4', wf)
                self.assertIn('Transform the image with balanced neutral exposure', wf['text_encode_edit']['inputs']['prompt'])
                self.assertIn('transform the image to realistic photograph', wf['text_encode_edit']['inputs']['prompt'])
            self.assertEqual(len(namespace['batch_failures']), 1)
            import zipfile
            archives = list(Path(tmp).glob('*.zip'))
            self.assertEqual(len(archives), 1)
            with zipfile.ZipFile(archives[0]) as archive:
                self.assertEqual(sorted(archive.namelist()), ['result1.png', 'result3.png'])


if __name__ == '__main__':
    unittest.main()
