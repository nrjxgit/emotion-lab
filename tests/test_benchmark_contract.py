"""Synthetic runner contract checks. No model runtimes or datasets are opened."""
import builtins
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emolab.inference import FaceEmotionAdapter, ModelMetadata, ModelResult
from emolab.benchmark.runner import (FER7, BenchmarkContractError,
                                     BenchmarkExecutionError, run_benchmark)
from training import evaluate_models as cli


@pytest.fixture(autouse=True)
def refuse_real_dependencies(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith(('emolab.adapters', 'training.fer_images', 'torch',
                            'tensorflow', 'onnxruntime', 'hsemotion', 'cv2')):
            raise AssertionError('real model/data dependency forbidden in synthetic checks: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


class Clock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t
    def advance(self, seconds):
        self.t += seconds


class Fake(FaceEmotionAdapter):
    def __init__(self, name='a', *, clock=None, fail_at=None, abstain_at=None,
                 probabilities=None, supported=None):
        self.name = name
        self.clock = clock
        self.fail_at = fail_at
        self.abstain_at = abstain_at
        self.probabilities = {'angry': 1.0} if probabilities is None else probabilities
        self.supported = FER7 if supported is None else supported
        self.seen = []
    def predict_proba(self, image):
        value = int(image.flat[0])
        self.seen.append((value, image.tobytes(), image.flags.writeable))
        if self.clock:
            self.clock.advance(.1)
        if value == self.fail_at:
            raise RuntimeError('synthetic failure')
        if value == self.abstain_at:
            return None
        return ModelResult(self.probabilities, 'angry', 1, 0, 7,
                           ModelMetadata(self.name, 'synthetic-v1', (2, 2, 3), {}, False,
                                         frozenset(self.supported), {'source': 'synthetic'}))


def cohort(n=3):
    return {'images': [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(n)],
            'labels': [0] * n, 'sample_ids': [f'synthetic:{i}' for i in range(n)],
            'requested_count': n, 'synthetic_only': True}


def run(adapters=None, data=None, **kwargs):
    data = cohort() if data is None else data
    return run_benchmark([Fake()] if adapters is None else adapters,
                         data['images'], data['labels'], sample_ids=data['sample_ids'],
                         requested_count=data['requested_count'], return_report=True,
                         synthetic=True, **kwargs)


def test_four_models_share_exact_order_and_once_prepared_crops():
    models = [Fake(name) for name in ('mini-xception', 'vit', 'claude-hsemotion', 'claude-combined')]
    crops = []
    def crop(image):
        crops.append(int(image.flat[0]))
        return image
    report = run(models, crop_fn=crop, requested_models=[m.name for m in models])
    assert crops == [0, 1, 2]
    assert report['complete'] is True
    for model in models:
        assert model.seen == models[0].seen
        assert [x[0] for x in model.seen] == [0, 1, 2]
        assert all(not x[2] for x in model.seen)
        assert report['results'][model.name]['attempted_sample_ids'] == cohort()['sample_ids']
    assert len(report['sample_manifest_sha256']) == 64


def test_adapter_cannot_reenable_writes_or_change_next_models_inputs():
    class Mutating(Fake):
        def predict_proba(self, image):
            with pytest.raises(ValueError):
                image.setflags(write=True)
            return super().predict_proba(image)
    first, second = Mutating('first'), Fake('second')
    assert run([first, second])['complete']
    assert first.seen == second.seen


def test_structured_execution_refuses_missing_ids():
    data = cohort()
    with pytest.raises(BenchmarkContractError, match='explicit sample IDs'):
        run_benchmark([Fake()], data['images'], data['labels'], return_report=True)


@pytest.mark.parametrize('change', [
    {'sample_ids': ['duplicate'] * 3}, {'sample_ids': ['a', 'b']},
    {'sample_ids': ['a', '', 'c']}, {'sample_ids': ['a', None, 'c']},
    {'requested_count': 4}, {'requested_count': True},
    {'labels': [0, 0]}, {'labels': [0, 7, 0]}, {'labels': [0., 0., 0.]},
    {'images': [], 'labels': [], 'sample_ids': [], 'requested_count': 0},
])
def test_invalid_cohort_refused_before_prediction(change):
    data = cohort()
    data.update(change)
    adapter = Fake()
    with pytest.raises(BenchmarkContractError):
        run([adapter], data)
    assert not adapter.seen


@pytest.mark.parametrize('adapters,requested', [
    ([Fake('a'), Fake('a')], None), ([Fake('a')], ['a', 'a']),
    ([Fake('a')], ['b']),
])
def test_duplicate_or_extra_model_identity_refused(adapters, requested):
    with pytest.raises(BenchmarkContractError):
        run(adapters, requested_models=requested)
    assert not any(a.seen for a in adapters)


def test_failed_and_missing_model_cannot_look_complete():
    failed, good = Fake('failed', fail_at=1), Fake('good')
    report = run([failed, good], requested_models=['failed', 'missing', 'good'])
    assert not report['complete']
    row = report['results']['failed']
    assert row['attempted_sample_ids'] == ['synthetic:0', 'synthetic:1']
    assert row['unprocessed_sample_ids'] == ['synthetic:2']
    assert 'accuracy' not in row
    assert report['results']['missing']['status'] == 'missing'
    assert report['results']['good']['complete']
    assert [x[0] for x in good.seen] == [0, 1, 2]
    with pytest.raises(BenchmarkExecutionError) as caught:
        run_benchmark([Fake(fail_at=0)], cohort()['images'], [0, 0, 0])
    assert not caught.value.report['complete']


def test_explicit_abstention_is_completed_execution_with_separate_coverage():
    report = run([Fake(abstain_at=1)])
    row = report['results']['a']
    assert report['complete']
    assert row['abstained_sample_ids'] == ['synthetic:1']
    assert row['coverage'] == pytest.approx(2 / 3)
    assert row['accuracy'] == 1


@pytest.mark.parametrize('kwargs', [
    {'probabilities': {'angry': float('nan')}},
    {'probabilities': {'angry': .5}}, {'probabilities': {'angry': -1}},
    {'probabilities': {'angry': 1.0, 'contempt': .1}},
    {'supported': ['angry']},
])
def test_invalid_common_distribution_is_failed_not_scored(kwargs):
    row = run([Fake(**kwargs)])['results']['a']
    assert not row['complete']
    assert 'accuracy' not in row


def test_deterministic_clock_defines_distinct_timing_boundaries():
    clock = Clock()
    def crop(image):
        clock.advance(.2)
        return image
    report = run([Fake('a', clock=clock), Fake('b', clock=clock)], cohort(2),
                 crop_fn=crop, clock=clock)
    assert report['total_wall_ms'] == pytest.approx(800)
    for row in report['results'].values():
        assert row['model_prediction_wall_ms'] == pytest.approx(200)
        assert row['observed_inference_sum_ms'] == pytest.approx(200)
        assert row['latency_ms'] == pytest.approx(100)
        for item in row['items']:
            assert item['observed_inference_ms'] == pytest.approx(100)
            assert item['adapter_reported_inference_ms'] == 7


def test_failed_inference_duration_is_retained():
    clock = Clock()
    row = run([Fake(clock=clock, fail_at=1)], clock=clock)['results']['a']
    assert not row['complete']
    assert row['items'][1]['observed_inference_ms'] == pytest.approx(100)
    assert row['observed_inference_sum_ms'] == pytest.approx(200)


@pytest.mark.parametrize('skipped', [
    [{'sample_id': 'synthetic:0', 'reason': 'overlap'}],
    [{'sample_id': 'outside'}],
    [{'sample_id': 'outside', 'reason': '   '}],
    [{'sample_id': 'outside', 'reason': 'x'}] * 2,
])
def test_exclusions_need_unique_disjoint_ids_and_reasons(skipped):
    with pytest.raises(BenchmarkContractError):
        run(skipped_samples=skipped)


def test_provenance_and_exclusions_are_snapshotted():
    config = {'seed': 505}
    references = {'a': {'model_id': 'fake-a', 'checkpoint_reference': 'synthetic://a',
                        'checkpoint_checksum': None, 'checkpoint_verified': False}}
    skipped = [{'sample_id': 'outside', 'reason': 'predefined synthetic exclusion'}]
    report = run(configuration=config, model_references=references, skipped_samples=skipped)
    config['seed'] = 0
    references['a']['model_id'] = 'changed'
    skipped[0]['reason'] = 'changed'
    assert report['configuration']['seed'] == 505
    assert report['model_references']['a']['model_id'] == 'fake-a'
    assert report['skipped_samples'][0]['reason'] == 'predefined synthetic exclusion'


def invoke(tmp_path, models, factories, loader=None, **kwargs):
    out = tmp_path / 'new-version' / 'report.json'
    args = ['--dataset', 'synthetic', '--out', str(out)]
    for name in models:
        args += ['--model', name]
    exit_code = cli.main(args, adapter_factories=factories,
                         dataset_loaders={'synthetic': loader or (lambda **kw: cohort())}, **kwargs)
    return exit_code, json.loads(out.read_text()), out


def test_cli_explicit_model_request_is_authoritative_and_records_provenance(tmp_path):
    names = ['mini-xception', 'vit', 'claude-hsemotion', 'claude-combined']
    made = []
    def factory(name):
        made.append(name)
        return Fake(name)
    factories = {name: lambda name=name: factory(name) for name in names}
    factories['must-not-default'] = lambda: pytest.fail('unrequested model constructed')
    code, report, _ = invoke(tmp_path, names, factories,
                            model_references={name: {'checkpoint_reference': 'synthetic://' + name}
                                              for name in names})
    assert code == 0 and report['complete'] and report['synthetic_only']
    assert made == names == report['requested_models']
    assert report['sample_ids'] == cohort()['sample_ids']
    assert report['configuration']['seed'] == 505


@pytest.mark.parametrize('models', [['unknown'], ['a', 'a']])
def test_cli_bad_models_fail_before_loader_or_factories(tmp_path, models):
    def forbidden(**kwargs):
        pytest.fail('data/model accessed for invalid request')
    code, report, _ = invoke(tmp_path, models, {'a': forbidden}, loader=forbidden)
    assert code == 1 and not report['complete']


def test_cli_factory_failure_and_shortfall_have_failure_exit(tmp_path):
    def broken():
        raise RuntimeError('synthetic unavailable')
    code, report, _ = invoke(tmp_path, ['broken', 'good'], {'broken': broken, 'good': lambda: Fake('good')})
    assert code == 1 and not report['complete']
    assert 'synthetic unavailable' in report['results']['broken']['error']
    data = cohort()
    data['requested_count'] = 4
    code, report, _ = invoke(tmp_path / 'shortfall', ['a'], {'a': Fake}, loader=lambda **kw: data)
    assert code == 1 and report['requested_count'] == 4 and report['actual_unique_count'] == 3


def test_cli_invalid_id_type_still_saves_explicit_failure(tmp_path):
    data = cohort()
    data['sample_ids'] = [['unhashable'], 'b', 'c']
    code, report, _ = invoke(tmp_path, ['a'], {'a': Fake}, loader=lambda **kw: data)
    assert code == 1 and not report['complete']
    assert report['actual_unique_count'] is None


def test_default_output_uses_new_versioned_paths_and_preserves_previous(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    factories = {'a': Fake}
    loaders = {'synthetic': lambda **kw: cohort()}
    args = ['--model', 'a', '--dataset', 'synthetic']
    assert cli.main(args, adapter_factories=factories, dataset_loaders=loaders) == 0
    first = next((tmp_path / 'eval_results/runner-contract-v1').glob('*/report.json'))
    saved = first.read_bytes()
    assert cli.main(args, adapter_factories=factories, dataset_loaders=loaders) == 0
    assert len(list((tmp_path / 'eval_results/runner-contract-v1').glob('*/report.json'))) == 2
    assert first.read_bytes() == saved


def test_existing_output_and_exclusive_creation_preserve_bytes(tmp_path):
    path = tmp_path / 'old.json'
    original = b'previous result\r\n'
    path.write_bytes(original)
    def forbidden(**kwargs):
        pytest.fail('input accessed before overwrite refusal')
    assert cli.main(['--out', str(path)], adapter_factories={}, dataset_loaders={'x': forbidden}) == 2
    with pytest.raises(FileExistsError):
        cli._write_report(path, {'replacement': True})
    assert path.read_bytes() == original
    bad = tmp_path / 'nonfinite.json'
    with pytest.raises(ValueError):
        cli._write_report(bad, {'bad': float('nan')})
    assert not bad.exists()


def test_unsupported_cached_route_fails_before_model_or_data_access(tmp_path):
    def forbidden():
        pytest.fail('real route/model touched')
    out = tmp_path / 'unsupported.json'
    code = cli.main(['--dataset', 'cremad-dfa', '--model', 'a', '--out', str(out)],
                    adapter_factories={'a': forbidden})
    assert code == 1
    assert not json.loads(out.read_text())['complete']


def test_seeded_subset_ids_per_class_counts_and_shortfall_are_explicit(tmp_path):
    calls = []
    def fake_loader(path):
        assert path == 'synthetic://generated-fer7'
        calls.append(path)
        return (np.zeros((14, 2, 2), dtype=np.uint8), np.repeat(np.arange(7), 2),
                ['0.jpg', '1.jpg'] * 7)
    def loader(subset, seed):
        result = cli._load_fer2013_test(subset, seed, loader=fake_loader,
                                       source_dir='synthetic://generated-fer7')
        result['synthetic_only'] = True
        return result
    data = loader(1, 505)
    assert data['sample_ids'] == loader(1, 505)['sample_ids']
    assert len(data['sample_ids']) == len(set(data['sample_ids'])) == 7
    assert len(data['skipped_samples']) == 7
    assert data['requested_count'] == 7
    assert all(v == {'available': 2, 'selected': 1, 'requested': 1}
               for v in data['configuration']['per_class_counts'].values())
    assert run([Fake()], data, skipped_samples=data['skipped_samples'])['complete']
    out = tmp_path / 'shortfall.json'
    code = cli.main(['--dataset', 'synthetic', '--subset', '3', '--model', 'a', '--out', str(out)],
                    adapter_factories={'a': Fake}, dataset_loaders={'synthetic': loader})
    report = json.loads(out.read_text())
    assert code == 1 and not report['complete']
    assert report['requested_count'] == 21 and report['actual_unique_count'] == 14
    assert all(v['available'] == 2 and v['requested'] == 3
               for v in report['configuration']['per_class_counts'].values())
