"""Bounded, explicitly SYNTHETIC per-clip analysis. No models or directory scans.

Public API: read_analysis(root, 'per-clip.jsonl', manifest=..., independent pins).
Root must be an explicit absolute local directory named SYNTHETIC-*. No links,
reparse ancestors, traversal, arbitrary filenames or discovered files are admitted.
Only the named JSONL is read, once, from a bounded snapshot. Manifest is supplied
in memory with full expected SHA256; every row binds its complete cohort/model/
protocol/support declaration. Expected file SHA256 binds exact parsed bytes.

Manifest schema synthetic-per-clip-manifest/1 has exactly synthetic_only=true,
exposure=MARKER, seed=505, complete=true, protocol_sha256, models{name:[support]},
clips[{id,actor,label}], class_support{six truths:count}, plus schema. Identifiers
are manufactured SYNTHETIC-* strings. At most 8 models, 256 actors, 10000 total
model/clip rows. Row schema synthetic-per-clip/1 has the same markers/protocol,
manifest_sha256, model/id/actor/label/supported_classes/status/probabilities/
uncertainty/complete. Status predicted requires a finite FER7 unit vector with
zero mass on unsupported classes and uncertainty=1-max(p). no_face,
decode_failure and failed require both quantities null. Failed rows invalidate
model completeness and NEVER become predictions. Missing rows are rejected.

Returns labelled descriptive counts only. No authenticity, accuracy, calibration,
rights, real performance or model-readiness claim. A SYNTHETIC directory name and
caller pins are consistency checks, not an authenticated data-admission boundary.
"""
from collections import Counter,defaultdict
from itertools import combinations
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import re
import stat

import numpy as np

MARKER="SYNTHETIC / NOT REAL MODEL PERFORMANCE"
SEED=505
LABELS=("angry","disgust","fear","happy","sad","surprise","neutral")
TRUTHS=("angry","disgust","fear","happy","sad","neutral")
STATUSES=("predicted","no_face","decode_failure","failed")
COLUMNS=LABELS+STATUSES[1:]
MAX_BYTES=2*1024**2
MAX_ROWS=10000
MAX_LINE_BYTES=16384


class SyntheticAnalysisError(ValueError):
    """Invalid/unsafe/incomplete manufactured input; never repair/coerce."""


def _keys(value,expected):
    if not isinstance(value,dict) or set(value)!=set(expected):
        raise SyntheticAnalysisError("missing/unknown schema fields")


def _pin(value):
    if not isinstance(value,str) or not re.fullmatch("[0-9a-f]{64}",value):
        raise SyntheticAnalysisError("full lowercase SHA256 required")


def _canonical(value):
    try:return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    except (TypeError,ValueError) as exc:raise SyntheticAnalysisError("invalid manifest JSON") from exc


def _identifier(value):
    if not isinstance(value,str) or not value.startswith("SYNTHETIC-") or len(value)>160 or any(ord(c)<32 for c in value):
        raise SyntheticAnalysisError("manufactured SYNTHETIC identifier required")


def _markers(value):
    if value["synthetic_only"] is not True or value["exposure"]!=MARKER or type(value["seed"]) is not int or value["seed"]!=SEED or value["complete"] is not True:
        raise SyntheticAnalysisError("complete synthetic-only exposure/seed required")


def _support(value):
    if not isinstance(value,list) or not value or any(not isinstance(c,str) or c not in LABELS for c in value) or len(set(value))!=len(value):
        raise SyntheticAnalysisError("unique named class support required")


def _manifest(value,expected,max_rows):
    _pin(expected);raw=_canonical(value)
    if len(raw)>MAX_BYTES or hashlib.sha256(raw).hexdigest()!=expected:raise SyntheticAnalysisError("manifest pin/size mismatch")
    value=json.loads(raw)
    _keys(value,{"schema","synthetic_only","exposure","seed","complete","protocol_sha256","models","clips","class_support"})
    _markers(value);_pin(value["protocol_sha256"])
    if value["schema"]!="synthetic-per-clip-manifest/1":raise SyntheticAnalysisError("manifest schema mismatch")
    models=value["models"];clips=value["clips"]
    if not isinstance(models,dict) or not 1<=len(models)<=8:raise SyntheticAnalysisError("bounded declared models required")
    for model,support in models.items():_identifier(model);_support(support)
    if not isinstance(clips,list) or not clips or len(clips)*len(models)>max_rows:raise SyntheticAnalysisError("bounded complete cohort required")
    identities={};actors=set()
    for clip in clips:
        _keys(clip,{"id","actor","label"});_identifier(clip["id"]);_identifier(clip["actor"])
        if clip["label"] not in TRUTHS or clip["id"] in identities:raise SyntheticAnalysisError("duplicate/invalid clip identity or truth")
        identities[clip["id"]]=clip;actors.add(clip["actor"])
    if len(actors)>256:raise SyntheticAnalysisError("actor limit exceeded")
    _keys(value["class_support"],TRUTHS)
    expected_counts=Counter(clip["label"] for clip in clips)
    if any(type(n) is not int or n<0 or n!=expected_counts[label] for label,n in value["class_support"].items()):
        raise SyntheticAnalysisError("class/support mismatch")
    return value,identities


def _checked_path(root,input_name):
    if not isinstance(root,(str,Path)):raise SyntheticAnalysisError("explicit synthetic root required")
    root=Path(root)
    if not root.is_absolute() or not root.name.startswith("SYNTHETIC-") or str(root).startswith(("\\\\","//")) or ".." in root.parts or any(":" in part for part in root.parts[1:]):
        raise SyntheticAnalysisError("explicit absolute local SYNTHETIC root required")
    if input_name!="per-clip.jsonl":raise SyntheticAnalysisError("only explicit per-clip.jsonl basename admitted")
    path=root/input_name
    for part in (*reversed(path.parents),path):
        try:info=part.lstat()
        except OSError as exc:raise SyntheticAnalysisError("missing synthetic input/root") from exc
        if stat.S_ISLNK(info.st_mode) or getattr(info,"st_file_attributes",0)&0x400:
            raise SyntheticAnalysisError("path following/reparse denied")
        if part==root and not stat.S_ISDIR(info.st_mode):raise SyntheticAnalysisError("synthetic root is not a directory")
    return path


def _snapshot(path,max_bytes,expected):
    before=path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=max_bytes:raise SyntheticAnalysisError("input type/byte limit")
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    with path.open("rb") as stream:
        if identity(os.fstat(stream.fileno()))!=identity(before):raise SyntheticAnalysisError("input identity changed")
        raw=stream.read(max_bytes+1)
        if identity(os.fstat(stream.fileno()))!=identity(before):raise SyntheticAnalysisError("input changed during read")
    if identity(path.lstat())!=identity(before) or len(raw)>max_bytes:raise SyntheticAnalysisError("input path/size changed")
    if hashlib.sha256(raw).hexdigest()!=expected:raise SyntheticAnalysisError("input file pin mismatch")
    return raw


def _json_line(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise SyntheticAnalysisError("duplicate JSON field")
            result[key]=value
        return result
    def constant(value):raise SyntheticAnalysisError("nonfinite JSON constant")
    try:return json.loads(raw,object_pairs_hook=pairs,parse_constant=constant)
    except (ValueError,UnicodeDecodeError) as exc:raise SyntheticAnalysisError("invalid JSONL row") from exc


def _row(value,manifest,identities,manifest_sha256):
    _keys(value,{"schema","synthetic_only","exposure","seed","complete","manifest_sha256","protocol_sha256","model","id","actor","label","supported_classes","status","probabilities","uncertainty"})
    _markers(value)
    if value["schema"]!="synthetic-per-clip/1" or value["manifest_sha256"]!=manifest_sha256 or value["protocol_sha256"]!=manifest["protocol_sha256"]:
        raise SyntheticAnalysisError("row protocol/manifest/schema mismatch")
    for key in ("model","id","actor"):_identifier(value[key])
    model=value["model"];clip=identities.get(value["id"])
    if model not in manifest["models"] or clip is None or any(value[k]!=clip[k] for k in ("actor","label")):
        raise SyntheticAnalysisError("mixed model/cohort or clip/actor/truth identity")
    if value["supported_classes"]!=manifest["models"][model]:raise SyntheticAnalysisError("row support differs from manifest")
    status=value["status"]
    if status not in STATUSES:raise SyntheticAnalysisError("unknown item status")
    vector=value["probabilities"];uncertainty=value["uncertainty"]
    if status!="predicted":
        if vector is not None or uncertainty is not None:raise SyntheticAnalysisError("non-predicted/failed items cannot carry predictions")
    else:
        if not isinstance(vector,list) or len(vector)!=7 or any(type(p) not in (float,int) or not math.isfinite(p) or not 0<=p<=1 for p in vector) or not math.isclose(sum(vector),1.,rel_tol=0,abs_tol=1e-6):
            raise SyntheticAnalysisError("finite unit FER7 vector required")
        if any(p!=0 and label not in value["supported_classes"] for label,p in zip(LABELS,vector)):
            raise SyntheticAnalysisError("positive probability on unsupported class")
        if type(uncertainty) not in (float,int) or not math.isfinite(uncertainty) or not 0<=uncertainty<=1 or not math.isclose(uncertainty,1-max(vector),rel_tol=0,abs_tol=1e-6):
            raise SyntheticAnalysisError("finite uncertainty=1-max(p) required")
    return value


def _table(**values):
    return {"synthetic_only":True,"exposure":MARKER,"seed":SEED,**values}


def _outcome(row):
    """Ties select the first class in LABELS, exactly as numpy.argmax does."""
    return LABELS[int(np.argmax(row["probabilities"]))] if row["status"]=="predicted" else row["status"]


def _summary(rows):
    confusion=np.zeros((6,10),dtype=np.int64);statuses={s:0 for s in STATUSES}
    buckets={"low_[0,1/3)":0,"medium_[1/3,2/3)":0,"high_[2/3,1]":0,"unavailable":0}
    for row in rows:
        statuses[row["status"]]+=1
        confusion[TRUTHS.index(row["label"]),COLUMNS.index(_outcome(row))]+=1
        uncertainty=row["uncertainty"]
        bucket="unavailable" if uncertainty is None else "low_[0,1/3)" if uncertainty<1/3 else "medium_[1/3,2/3)" if uncertainty<2/3 else "high_[2/3,1]"
        buckets[bucket]+=1
    n=len(rows);predicted=statuses["predicted"]
    return _table(denominator=n,prediction_count=predicted,coverage=predicted/n if n else None,
        model_complete=statuses["failed"]==0,status_counts=statuses,
        truth_order=list(TRUTHS),prediction_order=list(COLUMNS),confusion=confusion.tolist(),
        class_support={label:int(confusion[i].sum()) for i,label in enumerate(TRUTHS)},
        uncertainty=_table(denominator=n,defined_denominator=predicted,definition="1-max(probability); thirds of [0,1]",counts=buckets))


def read_analysis(synthetic_root,input_name,*,manifest,expected_manifest_sha256,expected_file_sha256,max_bytes=MAX_BYTES,max_rows=MAX_ROWS):
    """Read exactly one complete declared synthetic text file; return labelled counts."""
    if type(max_bytes) is not int or not 0<max_bytes<=MAX_BYTES or type(max_rows) is not int or not 0<max_rows<=MAX_ROWS:
        raise SyntheticAnalysisError("limits must stay within fixed hard bounds")
    _pin(expected_file_sha256)
    manifest,identities=_manifest(manifest,expected_manifest_sha256,max_rows)
    path=_checked_path(synthetic_root,input_name)
    raw=_snapshot(path,max_bytes,expected_file_sha256);lines=raw.splitlines()
    if not raw.endswith(b"\n") or len(lines)>max_rows:raise SyntheticAnalysisError("incomplete/oversize JSONL rows")
    models={name:{} for name in manifest["models"]}
    for line in lines:
        if not line or len(line)>MAX_LINE_BYTES:raise SyntheticAnalysisError("empty/oversize row")
        row=_row(_json_line(line),manifest,identities,expected_manifest_sha256)
        group=models[row["model"]]
        if row["id"] in group:raise SyntheticAnalysisError("duplicate clip ID within model")
        group[row["id"]]=row
    if any(set(rows)!=set(identities) for rows in models.values()):raise SyntheticAnalysisError("incomplete model/cohort rows")
    results={}
    for name,index in models.items():
        rows=[index[clip["id"]] for clip in manifest["clips"]]
        actors=defaultdict(list);classes={label:[] for label in TRUTHS}
        for row in rows:actors[row["actor"]].append(row);classes[row["label"]].append(row)
        results[name]={**_summary(rows),"supported_classes":manifest["models"][name],
            "by_actor":{actor:_summary(items) for actor,items in actors.items()},
            "by_class":{label:_summary(items) for label,items in classes.items()}}
    paired=[]
    for a,b in combinations(models,2):
        both=disagree=one=neither=outcome_disagree=0
        for identity in identities:
            left=models[a][identity];right=models[b][identity]
            lp=left["status"]=="predicted";rp=right["status"]=="predicted"
            if lp and rp:both+=1;disagree+=_outcome(left)!=_outcome(right)
            elif lp or rp:one+=1
            else:neither+=1
            outcome_disagree+=_outcome(left)!=_outcome(right)
        paired.append(_table(models=[a,b],cohort_denominator=len(identities),both_predicted_denominator=both,
            prediction_disagreement_count=disagree,prediction_disagreement_rate=disagree/both if both else None,
            one_predicted_count=one,neither_predicted_count=neither,outcome_disagreement_count=outcome_disagree,
            outcome_disagreement_rate=outcome_disagree/len(identities)))
    return _table(schema="synthetic-error-analysis/1",protocol_sha256=manifest["protocol_sha256"],
        manifest_sha256=expected_manifest_sha256,input_sha256=hashlib.sha256(raw).hexdigest(),
        cohort_size=len(identities),row_count=len(lines),analysis_complete=all(r["model_complete"] for r in results.values()),models=results,paired=paired)


def _validate_report(report):
    """Validate the complete bounded public report shape before HTML rendering.

    This checks structure and count/rate consistency, not input authenticity.
    Summary groups have a fixed depth; caller dictionaries are never traversed
    recursively. Only read_analysis output or equally valid synthetic reports
    are admitted. Booleans are not accepted as counts or numeric rates.
    """
    markers={"synthetic_only","exposure","seed"}
    summary_keys={"denominator","prediction_count","coverage","model_complete",
        "status_counts","truth_order","prediction_order","confusion","class_support","uncertainty"}
    def require(condition):
        if not condition:raise SyntheticAnalysisError("invalid synthetic report structure/counts")
    def labelled(value,keys):
        _keys(value,markers|keys)
        require(value["synthetic_only"] is True and value["exposure"]==MARKER
                and type(value["seed"]) is int and value["seed"]==SEED)
    def count(value,upper=MAX_ROWS):
        require(type(value) is int and 0<=value<=upper)
    def rate(value,numerator,denominator):
        if denominator==0:require(value is None)
        else:require(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
                     and math.isclose(value,numerator/denominator,rel_tol=0,abs_tol=1e-12))
    def summary(value,extra=frozenset()):
        labelled(value,summary_keys|set(extra))
        n=value["denominator"];predicted=value["prediction_count"]
        count(n);count(predicted,n)
        _keys(value["status_counts"],STATUSES)
        for c in value["status_counts"].values():count(c,n)
        require(sum(value["status_counts"].values())==n and value["status_counts"]["predicted"]==predicted)
        require(type(value["model_complete"]) is bool and value["model_complete"]==(value["status_counts"]["failed"]==0))
        rate(value["coverage"],predicted,n)
        require(type(value["truth_order"]) is list and value["truth_order"]==list(TRUTHS)
                and type(value["prediction_order"]) is list and value["prediction_order"]==list(COLUMNS))
        matrix=value["confusion"]
        require(type(matrix) is list and len(matrix)==len(TRUTHS))
        for row in matrix:
            require(type(row) is list and len(row)==len(COLUMNS))
            for c in row:count(c,n)
        require(sum(map(sum,matrix))==n)
        require(sum(sum(row[:7]) for row in matrix)==predicted)
        for index,status in enumerate(STATUSES[1:],7):
            require(sum(row[index] for row in matrix)==value["status_counts"][status])
        _keys(value["class_support"],TRUTHS)
        for i,label in enumerate(TRUTHS):
            count(value["class_support"][label],n)
            require(value["class_support"][label]==sum(matrix[i]))
        u=value["uncertainty"]
        labelled(u,{"denominator","defined_denominator","definition","counts"})
        require(type(u["denominator"]) is int and u["denominator"]==n
                and type(u["defined_denominator"]) is int and u["defined_denominator"]==predicted
                and u["definition"]=="1-max(probability); thirds of [0,1]")
        _keys(u["counts"],{"low_[0,1/3)","medium_[1/3,2/3)","high_[2/3,1]","unavailable"})
        for c in u["counts"].values():count(c,n)
        require(sum(u["counts"].values())==n and u["counts"]["unavailable"]==n-predicted)
        return n
    labelled(report,{"schema","protocol_sha256","manifest_sha256","input_sha256",
        "cohort_size","row_count","analysis_complete","models","paired"})
    require(report["schema"]=="synthetic-error-analysis/1")
    for key in ("protocol_sha256","manifest_sha256","input_sha256"):_pin(report[key])
    n=report["cohort_size"];count(n);require(n>0)
    models=report["models"]
    require(type(models) is dict and 1<=len(models)<=8)
    count(report["row_count"]);require(report["row_count"]==n*len(models))
    for name,value in models.items():
        _identifier(name);require(summary(value,{"supported_classes","by_actor","by_class"})==n)
        _support(value["supported_classes"])
        actors=value["by_actor"];classes=value["by_class"]
        require(type(actors) is dict and 1<=len(actors)<=256)
        for actor in actors:_identifier(actor)
        _keys(classes,TRUTHS)
        for groups in (actors,classes):
            require(sum(summary(group) for group in groups.values())==n)
            for i in range(len(TRUTHS)):
                for j in range(len(COLUMNS)):
                    require(sum(group["confusion"][i][j] for group in groups.values())==value["confusion"][i][j])
        for i,label in enumerate(TRUTHS):
            require(classes[label]["denominator"]==value["class_support"][label])
            require(all(sum(row)==0 for j,row in enumerate(classes[label]["confusion"]) if j!=i))
    require(type(report["analysis_complete"]) is bool
            and report["analysis_complete"]==all(v["model_complete"] for v in models.values()))
    pairs=report["paired"];expected=list(combinations(models,2))
    require(type(pairs) is list and len(pairs)==len(expected))
    for pair,names in zip(pairs,expected):
        labelled(pair,{"models","cohort_denominator","both_predicted_denominator",
            "prediction_disagreement_count","prediction_disagreement_rate","one_predicted_count",
            "neither_predicted_count","outcome_disagreement_count","outcome_disagreement_rate"})
        require(type(pair["models"]) is list and pair["models"]==list(names))
        count(pair["cohort_denominator"]);require(pair["cohort_denominator"]==n)
        for key in ("both_predicted_denominator","one_predicted_count","neither_predicted_count","outcome_disagreement_count"):
            count(pair[key],n)
        both=pair["both_predicted_denominator"]
        require(both+pair["one_predicted_count"]+pair["neither_predicted_count"]==n)
        count(pair["prediction_disagreement_count"],both)
        require(pair["prediction_disagreement_count"]<=pair["outcome_disagreement_count"])
        rate(pair["prediction_disagreement_rate"],pair["prediction_disagreement_count"],both)
        rate(pair["outcome_disagreement_rate"],pair["outcome_disagreement_count"],n)


def render_html(report):
    """Escaped read-only HTML, every table explicitly labelled as synthetic."""
    _validate_report(report)
    esc=lambda value:escape(str(value),quote=True)
    def table(title,headings,rows):
        return '<table data-synthetic-only="true"><caption>'+esc(MARKER+" - "+title)+'</caption><thead><tr>'+''.join('<th>'+esc(c)+'</th>' for c in headings)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>'+MARKER+'</title></head><body><h1>'+MARKER+'</h1><p>synthetic_only=true; seed 505; counts only, no real model performance.</p>']
    for name,result in report["models"].items():
        parts.append('<h2>'+esc(name)+'</h2>')
        parts.append(table("cohort/status/coverage",["denominator","predictions","coverage","complete","no_face","decode_failure","failed"],[[result["denominator"],result["prediction_count"],result["coverage"],result["model_complete"],*[result["status_counts"][s] for s in STATUSES[1:]]]]))
        parts.append(table("truth / predicted class or missing status",["truth",*result["prediction_order"]],[[label,*counts] for label,counts in zip(result["truth_order"],result["confusion"])]))
        for grouping in ("by_actor","by_class"):
            parts.append(table(grouping,["identity","denominator","predictions","coverage","failed"],[[key,value["denominator"],value["prediction_count"],value["coverage"],value["status_counts"]["failed"]] for key,value in result[grouping].items()]))
        u=result["uncertainty"]
        parts.append(table("uncertainty buckets",["bucket","count","all denominator","defined denominator"],[[key,count,u["denominator"],u["defined_denominator"]] for key,count in u["counts"].items()]))
    parts.append(table("paired model disagreement",["models","cohort denominator","both predicted denominator","prediction disagreements","rate","one predicted","neither predicted","outcome disagreements"],[[" vs ".join(pair["models"]),pair["cohort_denominator"],pair["both_predicted_denominator"],pair["prediction_disagreement_count"],pair["prediction_disagreement_rate"],pair["one_predicted_count"],pair["neither_predicted_count"],pair["outcome_disagreement_count"]] for pair in report["paired"]]))
    return ''.join(parts)+"</body></html>"
