"""Attempt log for labeled evaluation data (all attempts recorded, none fabricated)."""

import urllib.request, ssl, json, os

OUT = []


def attempt(name, fn):
    try:
        detail = fn()
        OUT.append({"name": name, "ok": True, "detail": detail})
        print("OK  ", name, detail)
    except Exception as e:
        OUT.append(
            {"name": name, "ok": False, "detail": f"{type(e).__name__}: {str(e)[:200]}"}
        )
        print("FAIL", name, str(e)[:200])


def kaggle_fer2013():
    # Kaggle API needs credentials; just check CLI presence + public listing reachability
    import shutil

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "kaggle CLI not installed; user must: pip install kaggle, add API token, kaggle datasets download -d msambare/fer2013"
        )
    return "kaggle CLI present"


def jaffe_page():
    req = urllib.request.Request(
        "http://www.kasrl.org/jaffe.html", headers={"User-Agent": "research-audit"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        html = r.read().decode("utf-8", "ignore")
    if "JAFFE" not in html:
        raise RuntimeError("unexpected page content")
    return f"kasrl page reachable ({len(html)} bytes); download requires request form per site"


if __name__ == "__main__":
    attempt("kaggle-fer2013", kaggle_fer2013)
    attempt("jaffe-kasrl", jaffe_page)
    os.makedirs("eval_results", exist_ok=True)
    json.dump(OUT, open("eval_results/data_attempts.json", "w"), indent=1)
    print("wrote eval_results/data_attempts.json")
