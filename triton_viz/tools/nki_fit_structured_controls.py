"""Export structural-family Level-A points from mapped control artifacts."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from triton_viz.tools.nki_cost_model import ComputeCalibration
from triton_viz.tools.nki_region_ir import structural_family
from triton_viz.tools.nki_trace_dump import _annotate_fusion_signature

FIELDS = ["family", "engine", "dtype", "free_dim", "effective_count",
          "instruction_count", "fixed_ns", "case", "compiler_version"]

def collect(roots, level_b, include_prefixes=()):
    rows=[]
    for trace in sorted(t for root in roots for t in root.glob("*/trace.jsonl")):
        case=trace.parent; ap=case/"hardware/source_mapping/audit.json"; mp=case/"hardware/source_mapping/instruction_mapping.csv"
        if include_prefixes and not case.name.startswith(tuple(include_prefixes)): continue
        if not ap.is_file() or not mp.is_file(): continue
        events=[json.loads(x) for x in trace.read_text().splitlines() if x.strip()]; _annotate_fusion_signature(events)
        groups={}
        for event in events:
            if event.get("fusion_group") is not None: groups.setdefault(int(event["fusion_group"]),[]).append(event)
        audit=json.loads(ap.read_text()); mappings=list(csv.DictReader(mp.open()))
        summary=json.loads((case/"hardware/explorer_summary.json").read_text()) if (case/"hardware/explorer_summary.json").is_file() else {}
        profile=next(iter(summary.values()),{})
        for group,members in groups.items():
            ir=members[0]["region_ir"]; family=structural_family(ir); free=int(ir["free_dim"])
            for engine,streams in (("vector",2),("scalar",1)):
                ea=audit["engines"][engine]; active=float(ea["regions"].get(str(group),0)); one=level_b.instruction_ns(engine,ir["dtype"],streams,free)
                selected=[r for r in mappings if r["engine"]==engine and r["fusion_group"]==str(group)
                          and r["opcode"] not in {"DRAIN","NOTIFY","EVENT_SEMAPHORE","EVENT_SEMAPHORE_RANGE_CLEAR","SET_ORDERING_MODE"}]
                rows.append({"family":family,"engine":engine,"dtype":ir["dtype"],"free_dim":free,
                             "effective_count":active/one if one else 0,"instruction_count":len(selected),
                             "fixed_ns":max(0,float(ea["explorer_active_ns"])-float(ea["mapped_active_ns"])) if group==0 else 0,
                             "case":case.name,"compiler_version":profile.get("compiler_version","")})
    return rows

def collect_legacy(paths):
    """Import mapped softmax points without making its full signature a key."""
    rows=[]
    for path in paths:
        for row in csv.DictReader(path.open()):
            rows.append({"family":"reduction_transcendental","engine":row["engine"],
                         "dtype":row["dtype"],"free_dim":int(float(row["free_dim"])),
                         "effective_count":float(row["effective_instruction_count"]),
                         "instruction_count":int(float(row["hardware_instruction_count"])),
                         "fixed_ns":0.0,"case":Path(row.get("case_dir",path.stem)).name,
                         "compiler_version":"legacy-mapped"})
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("roots",nargs="+",type=Path);p.add_argument("--compute-calibration-csv",type=Path,required=True);p.add_argument("--legacy-level-a-csv",nargs="*",type=Path,default=[]);p.add_argument("--include-case-prefix",nargs="*",default=[]);p.add_argument("--output",type=Path,required=True);a=p.parse_args(argv)
    rows=collect(a.roots,ComputeCalibration.from_csv(a.compute_calibration_csv),a.include_case_prefix)+collect_legacy(a.legacy_level_a_csv);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    print(f"Wrote {len(rows)} structured control points");return 0
if __name__=="__main__":raise SystemExit(main())
