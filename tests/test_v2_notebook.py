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
    setup = next(source(c) for c in notebook['cells'] if source(c).startswith('# @title 1.'))
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

    def test_anything2real_downloads_from_supplied_google_drive_link(self):
        u = self.utils
        gdown = types.ModuleType('gdown')
        gdown.download = mock.Mock(return_value='/tmp/downloaded')
        link = 'https://drive.google.com/file/d/1RA3DwpCE7ztEYl2cd4XAYJ2lm6-D_LSI/view?usp=sharing'
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.dict(sys.modules, {'gdown': gdown}), \
             mock.patch.object(u, 'verify_safetensors_file', side_effect=[(False, 0, 'missing'), (True, 304, 'ok')]), \
             mock.patch.object(u.subprocess, 'run') as aria:
            u.download_lora_multisource(u.PRESET_LORA_REGISTRY[u.CIVIT_ANYTHING2REAL_LORA_FILE]['url'], tmp, u.CIVIT_ANYTHING2REAL_LORA_FILE)
            gdown.download.assert_called_once_with(link, os.path.join(tmp, u.CIVIT_ANYTHING2REAL_LORA_FILE), fuzzy=True, quiet=False)
            aria.assert_not_called()
        self.assertEqual(u.resolve_lora_selection(u.LEGACY_CIVIT_ANYTHING2REAL_LORA_FILE), u.CIVIT_ANYTHING2REAL_LORA_FILE)

    def test_drive_download_failure_gives_actionable_message(self):
        u = self.utils
        gdown = types.ModuleType('gdown'); gdown.download = mock.Mock(return_value=None)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(sys.modules, {'gdown': gdown}):
            with self.assertRaisesRegex(RuntimeError, 'shared with anyone'):
                u.download_lora_multisource(u.PRESET_LORA_REGISTRY[u.CIVIT_ANYTHING2REAL_LORA_FILE]['url'], tmp, u.CIVIT_ANYTHING2REAL_LORA_FILE)

    def test_default_strengths_and_overrides(self):
        u = self.utils
        expected = {u.DMD2_2STEP_LORA_FILE: 1.0, u.DETAIL_ENHANCER_LORA_FILE: 1.0,
                    u.LIGHTNING_LORA_FILE: 0.7, u.ANYTHING2REAL_LORA_FILE: 0.8,
                    u.CIVIT_ANYTHING2REAL_LORA_FILE: 0.8, u.RESTORE_LORA_FILE: 0.7,
                    u.CONSISTENCY_LORA_FILE: 0.8, u.EXPOSURE_LORA_FILE: 0.8, u.BFS_HEAD_LORA_FILE: 1.0}
        for name, value in expected.items():
            active = u.collect_active_loras([(name, '', -1)])
            self.assertEqual(active[0]['strength_model'], value)
            self.assertEqual(active[0]['strength_clip'], value)
            self.assertEqual(u.collect_active_loras([(name, '', 0.55)])[0]['strength_model'], 0.55)
            self.assertEqual(u.collect_active_loras([(name, '', 0)]), [])
        self.assertEqual(u.collect_active_loras([('Custom (Enter filename below)', 'my.safetensors', -1)])[0]['strength_model'], 1.0)
        with self.assertRaises(ValueError):
            u.collect_active_loras([(u.RESTORE_LORA_FILE, '', -0.5)])

    def test_drive_storage_links_and_prevents_data_loss(self):
        u = self.utils
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / 'engine'
            output = Path(tmp) / 'output'
            root = Path(tmp) / 'drive'
            for sub in ('diffusion_models', 'text_encoders', 'vae', 'loras'):
                (engine / 'models' / sub).mkdir(parents=True)
            (engine / 'models' / 'loras' / 'put_loras_here').touch()
            output.mkdir()
            u.configure_drive_storage(str(engine), str(output), str(root), True, True)
            self.assertTrue((engine / 'models' / 'loras').is_symlink())
            (engine / 'models' / 'loras' / 'new.safetensors').write_bytes(b'test')
            self.assertEqual((root / 'models' / 'loras' / 'new.safetensors').read_bytes(), b'test')
            (output / 'result.png').write_bytes(b'image')
            self.assertEqual((root / 'output' / 'result.png').read_bytes(), b'image')
            u.configure_drive_storage(str(engine), str(output), str(root), True, True)
            with self.assertRaisesRegex(RuntimeError, 'linked elsewhere'):
                u.configure_drive_storage(str(engine), str(output), str(root / 'other'), True, True)
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / 'engine'; output = Path(tmp) / 'output'; root = Path(tmp) / 'drive'
            for sub in ('diffusion_models', 'text_encoders', 'vae', 'loras'):
                (engine / 'models' / sub).mkdir(parents=True)
            important = engine / 'models' / 'vae' / 'existing.safetensors'; important.write_bytes(b'data')
            with self.assertRaisesRegex(RuntimeError, 'contains existing files'):
                u.configure_drive_storage(str(engine), str(output), str(root), True, True)
            self.assertEqual(important.read_bytes(), b'data')
            self.assertFalse((engine / 'models' / 'diffusion_models').is_symlink())

    def test_drive_model_download_uses_sequential_stream(self):
        u = self.utils
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(u, 'is_drive_storage', return_value=True), \
             mock.patch.object(u, 'stream_download') as stream, \
             mock.patch.object(u.subprocess, 'run') as aria, \
             mock.patch.object(u, 'verify_safetensors_file', side_effect=[(False, 0, 'missing'), (True, 2, 'ok'), (True, 2, 'ok')]):
            u.download_lora_multisource('https://huggingface.co/user/model/resolve/main/a.safetensors', tmp, 'a.safetensors')
            stream.assert_called_once()
            aria.assert_not_called()

    def test_base_download_streams_directly_to_drive(self):
        import subprocess
        setup = next(source(c) for c in self.nb['cells'] if source(c).startswith('# @title 1.'))
        tree = ast.parse(setup)
        func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'foolproof_download')
        module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
        stream = mock.Mock()
        verify = mock.Mock(side_effect=[(False, 0, 'missing'), (True, 7000, 'ok')])
        env = {'os': os, 'subprocess': subprocess, 'time': __import__('time'),
               'is_drive_storage': lambda p: True, 'stream_download': stream, 'verify_safetensors_file': verify}
        exec(compile(module, 'setup_function', 'exec'), env)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(subprocess, 'run') as aria:
            result = env['foolproof_download']('https://example.com/model', tmp, 'model.safetensors')
            self.assertEqual(result, ('downloaded', 7000))
            stream.assert_called_once_with('https://example.com/model', os.path.join(tmp, 'model.safetensors'))
            aria.assert_not_called()

    def test_drive_storage_allows_mount_only_and_refuses_silent_switch(self):
        u = self.utils
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / 'engine'; output = Path(tmp) / 'output'; root = Path(tmp) / 'drive'
            u.configure_drive_storage(str(engine), str(output), str(root), False, False)
            self.assertFalse(root.exists())
            u.configure_drive_storage(str(engine), str(output), str(root), True, True)
            with mock.patch.object(u, 'is_drive_storage', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'fresh runtime'):
                    u.configure_drive_storage(str(engine), str(output), None, False, False)
            self.assertTrue((engine / 'models' / 'loras').is_symlink())
            self.assertTrue(output.is_symlink())

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
        for cell in [c for c in self.nb['cells'] if source(c).startswith(('# @title 5.', '# @title 6.', '# @title 6B.'))]:
            src = source(cell)
            for slot in ('primary', 'secondary', 'tertiary', 'quaternary'):
                row = next(line for line in src.splitlines() if line.startswith(slot + '_lora = '))
                for name in (u.CIVIT_ANYTHING2REAL_LORA_FILE, u.DETAIL_ENHANCER_LORA_FILE, u.DMD2_2STEP_LORA_FILE,
                             u.RESTORE_LORA_FILE, u.EXPOSURE_LORA_FILE, u.CONSISTENCY_LORA_FILE):
                    self.assertIn(name, row)
            self.assertIn('add_lora_triggers(', src)
        cell2 = next(source(c) for c in self.nb['cells'] if source(c).startswith('# @title 2.'))
        self.assertIn('DMD2_2STEP_LORA_FILE', cell2)
        self.assertIn('DETAIL_ENHANCER_LORA_FILE', cell2)

    def test_dmd2_2step_default_settings_and_fixed_sigmas(self):
        u = self.utils
        reg = u.PRESET_LORA_REGISTRY[u.DMD2_2STEP_LORA_FILE]
        self.assertEqual(
            reg['url'],
            'https://huggingface.co/linoyts/Qwen-Image-2.1-DMD2-2steps/resolve/main/pytorch_lora_weights_v1_5.safetensors',
        )
        self.assertEqual(reg['steps'], 2)
        self.assertEqual(reg['cfg'], 1.0)
        self.assertEqual(reg['sampler'], 'euler')
        self.assertEqual(reg['nodes'], '1.0, 0.5')
        self.assertFalse(reg['use_dynamic_shifting'])

        active = u.collect_active_loras([(u.DMD2_2STEP_LORA_FILE, '', -1)])
        self.assertEqual(active[0]['strength_model'], 1.0)
        steps, cfg, sampler, accel = u.optimize_sampling_for_active_loras(active, 20, 2.5, 'heun', auto_optimize=True)
        self.assertEqual((steps, cfg, sampler), (2, 1.0, 'euler'))
        self.assertEqual(accel['mode'], 'dmd2_2step')
        self.assertEqual(accel['sigmas_nodes'], '1.0, 0.5')
        self.assertFalse(accel['use_dynamic_shifting'])
        self.assertEqual((accel['base_shift'], accel['max_shift']), (0.0, 0.0))

        # Even with auto_optimize=False, DMD2 2-step uses its 2-step fixed schedule and CFG=1.0
        steps2, cfg2, sampler2, _ = u.optimize_sampling_for_active_loras(active, 20, 2.5, 'heun', auto_optimize=False)
        self.assertEqual((steps2, cfg2, sampler2), (2, 1.0, 'euler'))

        wf = {
            'unet_loader': {'inputs': {}, 'class_type': 'UNETLoader'},
            'qwen_cache': {'inputs': {'model': ['unet_loader', 0]}, 'class_type': 'QwenImage21Cache'},
            'text_encode_edit': {'inputs': {'images.image_1': ['load_input_image', 0], 'negative_prompt': 'bad'}, 'class_type': 'TextEncodeQwenImage21'},
            'ksampler': {'inputs': {'seed': 777, 'steps': 2, 'positive': ['text_encode_edit', 0], 'latent_image': ['empty_latent', 0], 'model': ['qwen_cache', 0]}, 'class_type': 'KSampler'},
        }
        model, clip = u.build_lora_chain(wf, active, ['unet_loader', 0], ['clip_loader', 0], check_disk=False)
        wf['qwen_cache']['inputs']['model'] = model
        self.assertEqual(wf['lora_loader_1']['inputs']['strength_model'], 1.0)
        self.assertEqual(wf['lora_loader_1']['inputs']['strength_clip'], 0.0)
        u.apply_acceleration_to_workflow(wf, active, steps=steps, cfg=cfg, sampler_name=sampler)
        self.assertNotIn('qwen_cache', wf)
        self.assertEqual(wf['text_encode_edit']['inputs']['negative_prompt'], '')
        self.assertEqual(wf['turbo_sigmas']['inputs']['nodes'], '1.0, 0.5')
        self.assertEqual(wf['turbo_sigmas']['inputs']['base_shift'], 0.0)
        self.assertEqual(wf['turbo_sigmas']['inputs']['max_shift'], 0.0)
        self.assertEqual(wf['ksampler']['class_type'], 'SamplerCustomAdvanced')
        self.assertEqual(wf['ksampler']['_meta']['scheduler'], 'dmd2_2step (2 steps)')
        self.assertEqual(u.format_active_loras_badge(active), '1 Active (⚡ DMD2 2-Step)')

        with self.assertRaisesRegex(ValueError, 'either Lightning or Viggle'):
            u.collect_active_loras([(u.DMD2_2STEP_LORA_FILE, '', 1), (u.LIGHTNING_LORA_FILE, '', 1)])
        with self.assertRaisesRegex(ValueError, 'either Lightning or Viggle'):
            u.collect_active_loras([(u.DMD2_2STEP_LORA_FILE, '', 1), (u.VIGGLE_TURBO_UNET_FILE, '', 1)])

    def test_custom_turbo_sigmas_node_fixed_and_shifted_schedules(self):
        u = self.utils
        fake_torch = types.ModuleType('torch')
        fake_torch.float32 = 'float32'
        fake_torch.Tensor = list
        fake_torch.tensor = lambda data, **kwargs: list(data)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(u, 'get_engine_dir', return_value=tmp), \
             mock.patch.object(u.urllib.request, 'urlopen', side_effect=OSError('offline')), \
             mock.patch.dict(sys.modules, {'torch': fake_torch}):
            node_file = Path(tmp) / 'custom_nodes' / 'qwen_turbo_nodes.py'
            node_file.parent.mkdir(parents=True, exist_ok=True)
            node_file.write_text('# outdated content', encoding='utf-8')
            u.ensure_custom_nodes_installed()
            code = node_file.read_text(encoding='utf-8')
            self.assertIn('class QwenTurboSigmas', code)
            mod = types.ModuleType('qwen_turbo_nodes')
            exec(compile(code, str(node_file), 'exec'), mod.__dict__)
            node = mod.QwenTurboSigmas()
            # DMD2 2-Step: fixed sigmas [1.0, 0.5, 0.0] with no dynamic shifting (base_shift=0.0, max_shift=0.0)
            fixed_sigmas, = node.get_sigmas({'samples': None}, nodes='1.0, 0.5', base_shift=0.0, max_shift=0.0)
            self.assertEqual(fixed_sigmas, [1.0, 0.5, 0.0])
            # Lightning 4-Step: dynamic shifted sigmas (base_shift=0.5, max_shift=0.9)
            shifted_sigmas, = node.get_sigmas({'samples': None}, nodes='1.0, 0.75, 0.5, 0.25', base_shift=0.5, max_shift=0.9)
            self.assertEqual(len(shifted_sigmas), 5)
            self.assertEqual(shifted_sigmas[0], 1.0)
            self.assertEqual(shifted_sigmas[-1], 0.0)
            self.assertGreater(shifted_sigmas[2], 0.5)

    def test_detail_enhancer_default_prompt_and_dmd2_combo_in_cell5(self):
        u = self.utils
        expected_prompt = (
            'Enhance this image to high resolution with rich fine details. '
            'Sharpen textures and surfaces, add natural microdetails, and improve clarity '
            'and definition across the image while preserving the original composition, '
            'lighting, atmosphere, and visual style; preserve the original style.'
        )
        self.assertEqual(u.DETAIL_ENHANCER_DEFAULT_PROMPT, expected_prompt)
        self.assertEqual(
            u.PRESET_LORA_REGISTRY[u.DETAIL_ENHANCER_LORA_FILE]['url'],
            'https://huggingface.co/reverentelusarca/elusarcas-qwen-2.1-detail-enhancer-lora/resolve/main/elusarcas-qwen2-1-detailer-v1.safetensors',
        )

        # Selecting Detail Enhancer as primary LoRA in Cell 5 replaces the untouched Anything2Real default prompt
        cell = next(source(c) for c in self.nb['cells'] if source(c).startswith('# @title 5.'))
        cell = cell.replace(
            'primary_lora = "Qwen2.1_Anything2RealCharacters.safetensors" # @param',
            f'primary_lora = "{u.DETAIL_ENHANCER_LORA_FILE}" # @param',
        )
        cell = cell.replace(
            'secondary_lora = "None" # @param',
            f'secondary_lora = "{u.DMD2_2STEP_LORA_FILE}" # @param',
        )
        img = Image.new('RGB', (100, 120))
        u.load_multiple_raw_images = lambda **kwargs: [('sample.png', img)]
        u.save_processed_image = lambda image, name, **kwargs: (name, 100, 120)
        u.ensure_loras_downloaded = lambda active: None
        u.purge_backend_memory = lambda **kwargs: None
        seen = []
        def generate(wf, title, w, h, seed, summary, **kwargs):
            seen.append((copy.deepcopy(wf), summary))
            return [img]
        generate.last_saved_paths = []
        u.execute_comfy_generation = generate
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(u, 'get_output_dir', return_value=tmp), \
             mock.patch.dict(sys.modules, {'qwen_studio_utils': u}):
            namespace = {}
            exec(compile(cell, 'detail_enhancer_dmd2', 'exec'), namespace)

        self.assertEqual(len(seen), 1)
        wf, summary = seen[0]
        self.assertEqual(wf['text_encode_edit']['inputs']['prompt'], expected_prompt)
        self.assertEqual(wf['lora_loader_1']['inputs']['lora_name'], u.DETAIL_ENHANCER_LORA_FILE)
        self.assertEqual(wf['lora_loader_1']['inputs']['strength_model'], 1.0)
        self.assertEqual(wf['lora_loader_2']['inputs']['lora_name'], u.DMD2_2STEP_LORA_FILE)
        self.assertEqual(wf['lora_loader_2']['inputs']['strength_model'], 1.0)
        self.assertEqual(wf['lora_loader_2']['inputs']['strength_clip'], 0.0)
        self.assertEqual(wf['turbo_sigmas']['inputs']['nodes'], '1.0, 0.5')
        self.assertEqual(wf['turbo_sigmas']['inputs']['base_shift'], 0.0)
        self.assertEqual(wf['turbo_sigmas']['inputs']['max_shift'], 0.0)
        self.assertIn('DMD2 2-Step', summary)

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
        head = next(source(c) for c in self.nb['cells'] if source(c).startswith('# @title 6.'))
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
        self.assertEqual(wf['lora_loader_2']['inputs']['strength_model'], 0.8)
        self.assertEqual(wf['lora_loader_3']['inputs']['strength_model'], 0.8)
        cfg = namespace['_CELL6_LAST_CONFIG']
        self.assertEqual(cfg['tertiary_lora'], u.EXPOSURE_LORA_FILE)
        self.assertEqual(cfg['quaternary_lora'], u.CIVIT_ANYTHING2REAL_LORA_FILE)
        u.load_or_reuse_reference_image = lambda **kwargs: img
        u.load_multiple_raw_images = lambda **kwargs: [('target', img)]
        captured = []
        u.execute_batch_head_swap_6b = lambda **kwargs: captured.append(kwargs)
        with mock.patch.dict(sys.modules, {'qwen_studio_utils': u}):
            exec(compile(next(source(c) for c in self.nb['cells'] if source(c).startswith('# @title 6B.')), 'head_batch', 'exec'), namespace)
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
        cell = next(source(c) for c in self.nb['cells'] if source(c).startswith('# @title 5.'))
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
        cell = next(source(c) for c in self.nb['cells'] if source(c).startswith('# @title 5.')).replace('seed = -1 # @param', 'seed = -2 # @param')
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
