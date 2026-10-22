"""Collection-policy tests run only manufactured mini-repos, never real discovery."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def mini_repo(tmp_path):
    root=tmp_path/"SYNTHETIC-mini-repo";root.mkdir()
    for name in ("pytest.ini","conftest.py"):
        (root/name).write_bytes((ROOT/name).read_bytes())
    (root/"tests").mkdir();(root/"outputs"/"nested").mkdir(parents=True)
    allowed="def test_allowed():\n    assert True\n"
    (root/"tests"/"test_allowed.py").write_text(allowed,encoding="ascii")
    (root/"tests"/"test_v092_legitimate.py").write_text(allowed,encoding="ascii")
    (root/"test_ordinary_root.py").write_text(allowed,encoding="ascii")
    poison="from pathlib import Path\nPath(__file__).with_suffix('.IMPORTED').write_text('POISON')\nraise RuntimeError('POISON IMPORT FORBIDDEN')\n"
    (root/"test_v092_poison.py").write_text(poison,encoding="ascii")
    (root/"test_v092_syntax.py").write_text("'unterminated\n",encoding="ascii")
    (root/"outputs"/"nested"/"test_poison.py").write_text(poison,encoding="ascii")
    return root


@pytest.mark.parametrize("arguments,count,blocked",[
    ([],2,False),(["."],3,False),(["tests/test_v092_legitimate.py"],1,False),
    (["test_ordinary_root.py"],1,False),(["test_v092_poison.py"],0,True),
    (["test_v092_syntax.py"],0,True),(["outputs"],0,True),
    (["outputs/nested/test_poison.py"],0,True),
    (["test_v092_poison.py::test_unused"],0,True),
    ([".\\outputs\\nested\\test_poison.py"],0,True),
])
def test_collection_never_imports_poison(mini_repo,arguments,count,blocked):
    env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    for key in ("PYTEST_ADDOPTS","PYTEST_PLUGINS","PYTHONPATH"):env.pop(key,None)
    command=[sys.executable,"-B","-m","pytest","-p","no:cacheprovider","--collect-only","-q",*arguments]
    result=subprocess.run(command,cwd=mini_repo,env=env,capture_output=True,text=True,timeout=30)
    text=result.stdout+result.stderr
    assert not list(mini_repo.rglob("*.IMPORTED")),text
    assert "POISON IMPORT FORBIDDEN" not in text and "SyntaxError" not in text,text
    if blocked:
        assert result.returncode==4 and "O02 collection blocked" in text,text
    else:
        assert result.returncode==0 and f"{count} test" in text,text
        if not arguments:
            assert "test_ordinary_root.py" not in text
            assert "tests/test_allowed.py::test_allowed" in text
            assert "tests/test_v092_legitimate.py::test_allowed" in text


def test_resolved_anchored_case_and_separator_policy(mini_repo):
    spec=importlib.util.spec_from_file_location("synthetic_collection_policy",mini_repo/"conftest.py")
    hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
    for path in (mini_repo/"test_v092_poison.py",mini_repo/"TEST_V092_UPPER.PY",mini_repo/"OUTPUTS"/"anything.py",str(mini_repo)+"\\outputs\\nested\\test_poison.py"):
        assert hook.pytest_ignore_collect(path,None) is True
    for path in (mini_repo/"tests"/"test_v092_legitimate.py",mini_repo/"test_ordinary_root.py",mini_repo/"outputs_backup"/"test_a.py",mini_repo.parent/"outputs"/"test_a.py",mini_repo/"outputs"/".."/"test_ordinary_root.py"):
        assert hook.pytest_ignore_collect(path,None) is None


@pytest.mark.skipif(os.name!="nt",reason="Windows native case-insensitive explicit-path check")
@pytest.mark.parametrize("argument",["TEST_V092_POISON.PY","OUTPUTS\\NESTED\\TEST_POISON.PY"])
def test_windows_explicit_case_variants(mini_repo,argument):
    test_collection_never_imports_poison(mini_repo,[argument],0,True)
