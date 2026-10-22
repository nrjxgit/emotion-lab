"""Second-round labeled-data attempts (https + public test faces)."""

import urllib.request, json, os

OUT = []
if os.path.exists("eval_results/data_attempts.json"):
    OUT = json.load(open("eval_results/data_attempts.json"))


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


def jaffe_https():
    req = urllib.request.Request(
        "https://www.kasrl.org/jaffe.html", headers={"User-Agent": "research-audit"}
    )
    with urllib.request.urlopen(req, timeout=25) as r:
        html = r.read().decode("utf-8", "ignore")
    if "JAFFE" not in html:
        raise RuntimeError("unexpected content")
    return "kasrl reachable via https; download requires request form"


def oarriaga_test_images():
    req = urllib.request.Request(
        "https://api.github.com/repos/oarriaga/face_classification/contents/images",
        headers={"User-Agent": "research-audit"},
    )
    with urllib.request.urlopen(req, timeout=25) as r:
        items = json.load(r)
    names = [i["name"] for i in items if i["type"] == "file"]
    return (
        f"public test faces: {names[:10]} (unlabeled demo images, not a benchmark set)"
    )


if __name__ == "__main__":
    attempt("jaffe-https", jaffe_https)
    attempt("oarriaga-images", oarriaga_test_images)
    json.dump(OUT, open("eval_results/data_attempts.json", "w"), indent=1)
